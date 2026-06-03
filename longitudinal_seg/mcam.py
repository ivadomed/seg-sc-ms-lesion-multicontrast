"""
Architecture (Figure 2):
  Inputs:
    f_t^i       — feature map from current time point encoder
    f_{t-1}^i  — feature map from previous time point encoder
    f_SEM_{t-1} — Shape Extractor Module output (encoded prior mask)

  Pipeline:
    1. Patch-embed f_t^i   -> P_t          (LayerNorm included)
    2. Patch-embed f_{t-1}^i -> P_{t-1}    (LayerNorm included)
    3. Fuse f_SEM into P_{t-1}:
         reshape f_SEM, broadcast-add to P_{t-1},
         ReLU -> Conv2d -> P_fused_{t-1}
    4. Mamba block on P_t          -> P_t_mamba
    5. Mamba block on P_fused_{t-1} -> P_{t-1}_mamba
    6. Cross-attention (query=P_t_mamba, key/value=P_{t-1}_mamba)
    7. Unpack embedding -> f_att^i  (spatial feature map)
    8. Residual: f_CAM_t^i = ReLU(f_att^i + f_t^i)

Author: Pierre-Louis Benveniste
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba  # pip install mamba-ssm (requires CUDA)


# ---------------------------------------------------------------------------
# Patch Embedding  (3-D spatial -> sequence)
# ---------------------------------------------------------------------------
class PatchEmbedding3D(nn.Module):
    """
    Projects 3-D feature maps using true 3D patches to keep sequence length sane.
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        # Using a strided 3D convolution downsamples the sequence length by patch_size^3
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        # x: [B, C, D, H, W]
        x = self.proj(x) # [B, E, D', H', W']
        D_new, H_new, W_new = x.shape[2:]
        
        # Flatten spatial dimensions into sequence length N
        tokens = x.flatten(2).transpose(1, 2) # [B, N, E]
        tokens = self.norm(tokens)
        return tokens, (D_new, H_new, W_new) # Pass the new downsampled spatial shape


# ---------------------------------------------------------------------------
# Unpack Embedding  (sequence -> 3-D spatial)
# ---------------------------------------------------------------------------
class UnpackEmbedding(nn.Module):
    """Inverse of PatchEmbedding3D: [B, N, E] -> [B, C_out, D', H', W']."""
    def __init__(self, embed_dim: int, out_channels: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(embed_dim, out_channels)
        # Optional: Use ConvTranspose3d if you want to upsample back to original resolution,
        # but since MCAM maps [C -> C], a simple reshape might suffice depending on where it fits.

    def forward(self, tokens: torch.Tensor, downsampled_spatial_shape, original_spatial_shape) -> torch.Tensor:
        D_n, H_n, W_n = downsampled_spatial_shape
        x = self.proj(tokens)               # [B, N, C_out]
        x = x.transpose(1, 2)              # [B, C_out, N]
        x = x.reshape(x.shape[0], x.shape[1], D_n, H_n, W_n)
        
        # Upsample back to the size expected by the UNet decoder skip connection
        if x.shape[2:] != original_spatial_shape:
            x = F.interpolate(x, size=original_spatial_shape, mode='trilinear', align_corners=False)
        return x


# ---------------------------------------------------------------------------
# Mamba-enhanced Cross-Attention Module (M-CAM)
# ---------------------------------------------------------------------------
class MCAM(nn.Module):
    """
    Mamba-enhanced Cross-Attention Module.

    Fuses features from two time points and an optional prior segmentation
    mask (via SEM) to produce a temporally-aware feature map.

    Args:
        in_channels:   spatial feature channels of f_t^i and f_{t-1}^i
                       if f_t is of shape [B, C, D, H, W], then in_channels = C
        embed_dim:     internal patch-embedding / Mamba dimension E

        num_heads:     number of heads for multi-head cross-attention
        sem_channels:  latent channels produced by SEM
                       (set to 0 if not using SEM / mask unavailable)
        mamba_kwargs:  forwarded to MambaBlock (d_state, d_conv, expand)
    """

    def __init__(self, in_channels: int, embed_dim: int = 128, num_heads: int = 2, sem_channels: int = 32, mamba_kwargs: dict | None = None,):
        super().__init__()

        # --- Patch embeddings ---
        self.patch_embed_t   = PatchEmbedding3D(in_channels, embed_dim)
        self.patch_embed_tm1 = PatchEmbedding3D(in_channels, embed_dim)
        # --- SEM fusion ---
        self.sem_proj      = nn.Linear(sem_channels, embed_dim)
        self.sem_fuse_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # --- Mamba blocks ---
        self.mamba_t   = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        self.mamba_tm1 = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)

        # --- Cross-attention: query from t, key/value from t-1 ---
        # batch_first=True so we can feed (B, N, E) directly, matching the
        # Mamba block convention. Without this, MHA treats the batch dim as
        # the sequence dim and silently scrambles the attention.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True,
        )

        # --- Unpack back to spatial ---
        self.unpack = UnpackEmbedding(embed_dim, in_channels)

        # --- Final activation ---
        self.relu = nn.ReLU(inplace=True)

    # ------------------------------------------------------------------
    def _fuse_sem(self, P_tm1: torch.Tensor, f_sem: torch.Tensor, spatial_shape,) -> torch.Tensor:
        """
        Implement the SEM-fusion block from Section 2.2:
          - reshape f_SEM from [B, sem_C, D, H, W] to [B, 3D, H, W]
          - expand P_{t-1} from [B, N, E] to [B, 1, N, E] and broadcast-add
          - ReLU -> Conv2d -> [B, N, E]
        """
        B, N, E = P_tm1.shape
        D, H, W = spatial_shape
        sem_C   = f_sem.shape[1]

        # Resize f_sem to match spatial dimensions of f_t if needed (e.g. if SEM output is at a different resolution)
        if f_sem.shape[2:] != (D, H, W):
            f_sem = F.interpolate(f_sem, size=(D, H, W), mode='trilinear', align_corners=False)

        # Step 1 — flatten f_SEM spatially to get one token per voxel.
        # f_sem is [B, sem_C, D, H, W]; we want [B, N, sem_C] where N = D*H*W.
        # Permute channels last, then flatten the three spatial dims.
        sem_seq = f_sem.permute(0, 2, 3, 4, 1)  # [B, D, H, W, sem_C]
        sem_seq = sem_seq.reshape(B, N, sem_C)   # [B, N, sem_C]
 
        # Step 2 — project sem_C -> E  (fixed linear, initialised in __init__)
        sem_seq   = self.sem_proj(sem_seq)                       # [B, N, E]
 
        # Step 3 — broadcast-add (both are now [B, N, E])
        P_fused   = P_tm1 + sem_seq                              # [B, N, E]
 
        # Step 4 — ReLU
        P_fused   = F.relu(P_fused, inplace=False)               # [B, N, E]
 
        # Step 5 — Conv2d to restore/refine embedding shape.
        # We treat [B, N, E] as [B, E, 1, N] so a 1×1 Conv2d acts like a
        # channel-mixing linear while preserving the sequence structure.
        P_fused   = P_fused.transpose(1, 2).unsqueeze(2)         # [B, E, 1, N]
        P_fused   = self.sem_fuse_conv(P_fused)                   # [B, E, 1, N]
        P_fused   = P_fused.squeeze(2).transpose(1, 2)            # [B, N, E]

        return P_fused

    # ------------------------------------------------------------------
    def forward(self, f_t: torch.Tensor, f_tm1: torch.Tensor, f_sem: torch.Tensor | None = None, ) -> torch.Tensor:
        """
        Args:
            f_t   : [B, C, D, H, W]  current time-point feature map
            f_tm1 : [B, C, D, H, W]  previous time-point feature map
            f_sem : [B, sem_C, D, H, W] SEM output (optional)

        Returns:
            f_CAM : [B, C, D, H, W]  temporally-enhanced feature map
        """
        # 1. Patch embedding
        P_t,   spatial_t   = self.patch_embed_t(f_t)     # [B, N, E]       
        # print("P_t.shape:", P_t.shape) 
        P_tm1, spatial_tm1 = self.patch_embed_tm1(f_tm1)  # [B, N, E]
        # print("P_tm1.shape:", P_tm1.shape)

        # 2. Reshape f_sem and fuse into P_{t-1} and apply ReLU + Conv2d.
        #    NB: spatial_tm1 is the *downsampled* spatial shape of P_{t-1}
        #    (after the strided patch embed), which is what N corresponds to.
        #    Passing f_t.shape[2:] here would try to flatten f_sem at the
        #    original resolution into N tokens at the downsampled resolution
        #    and crash on the reshape.
        P_fused_tm1 = self._fuse_sem(P_tm1, f_sem, spatial_tm1)
        # print("P_fused_tm1.shape:", P_fused_tm1.shape)

        # 3. Mamba blocks capture long-range dependencies
        P_t_m   = self.mamba_t(P_t)            # [B, N, E]
        # print("P_t_m.shape:", P_t_m.shape)
        P_tm1_m = self.mamba_tm1(P_fused_tm1)  # [B, N, E]
        # print("P_tm1_m.shape:", P_tm1_m.shape)
        
        # 4. Cross-attention: current time-point queries into previous
        #    Q = current, K = V = previous (temporal alignment)
        # need_weights=False is required so that nn.MultiheadAttention routes
        # to F.scaled_dot_product_attention (FlashAttention / mem-efficient
        # backend). Otherwise the N×N matmul materialises and OOMs at the
        # finest MCAM level (N can be tens of thousands of tokens).
        f_att, _ = self.cross_attn(query=P_t_m, key=P_tm1_m, value=P_tm1_m, need_weights=False)                                       # [B, N, E]
        # print("f_att.shape:", f_att.shape)

        # 5. Unpack to spatial feature map
        f_att_spatial = self.unpack(f_att, spatial_t, f_t.shape[2:])   # [B, C, D, H, W]
        # print("f_att_spatial.shape:", f_att_spatial.shape)

        # 6. Residual connection with input feature map + ReLU
        f_cam = self.relu(f_att_spatial + f_t)

        return f_cam