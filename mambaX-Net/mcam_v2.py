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
    Flattens a 3-D feature map [B, C, D, H, W] into a patch sequence
    [B, N, E] via a learned linear projection + rearangement + LayerNorm.

    N = D * H * W  (patch size = 1 voxel, matching the paper's description
    of a linear layer + rearrangement).
    """

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        """
        x    : [B, C, D, H, W]
        return: tokens [B, N, E],  spatial shape (D, H, W) for unpacking
        """
        B, C, D, H, W = x.shape
        # [B, C, N] -> [B, N, C]
        tokens = x.flatten(2).transpose(1, 2)   # [B, N, C]
        tokens = self.norm(self.proj(tokens))    # [B, N, E]
        return tokens


# ---------------------------------------------------------------------------
# Mamba block
# ---------------------------------------------------------------------------
class MambaBlock(nn.Module):
    """Thin wrapper around mamba-ssm's Mamba for sequence [B, N, E]."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, E]  ->  [B, N, E]"""
        return self.mamba(self.norm(x))


# ---------------------------------------------------------------------------
# Unpack Embedding  (sequence -> 3-D spatial)
# ---------------------------------------------------------------------------
class UnpackEmbedding(nn.Module):
    """Inverse of PatchEmbedding3D: [B, N, E] -> [B, C_out, D, H, W]."""

    def __init__(self, embed_dim: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_channels)

    def forward(self, tokens: torch.Tensor, spatial_shape) -> torch.Tensor:
        D, H, W = spatial_shape
        x = self.proj(tokens)              # [B, N, C_out]
        x = x.transpose(1, 2)             # [B, C_out, N]
        x = x.reshape(x.shape[0], x.shape[1], D, H, W)
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
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

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

        # Reshape f_sem: [B, sem_C, D, H, W] -> [B, sem_C*D, H, W]
        # The paper says "reshape to (B × 3D × H × W)";
        # we generalise to sem_C*D to handle arbitrary channel counts.
        f_sem_reshaped = f_sem.reshape(B, sem_C * D, H, W)   # [B, sem_C*D, H, W]
        print("f_sem_reshaped.shape:", f_sem_reshaped.shape)

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
        P_t = self.patch_embed_t(f_t)     # [B, N, E]       
        print("P_t.shape:", P_t.shape) 
        P_tm1 = self.patch_embed_tm1(f_tm1)  # [B, N, E]
        print("P_tm1.shape:", P_tm1.shape)

        # 2. Reshape f_sem and fuse into P_{t-1} and apply ReLU + Conv2d
        P_fused_tm1 = self._fuse_sem(P_tm1, f_sem, f_t.shape[2:])
        print("P_fused_tm1.shape:", P_fused_tm1.shape)

        # 3. Mamba blocks capture long-range dependencies
        P_t_m   = self.mamba_t(P_t)            # [B, N, E]
        print("P_t_m.shape:", P_t_m.shape)
        P_tm1_m = self.mamba_tm1(P_fused_tm1)  # [B, N, E]
        print("P_tm1_m.shape:", P_tm1_m.shape)
        
        # 4. Cross-attention: current time-point queries into previous
        #    Q = current, K = V = previous (temporal alignment)
        f_att, _ = self.cross_attn(query=P_t_m, key=P_tm1_m, value=P_tm1_m)                                       # [B, N, E]
        print("f_att.shape:", f_att.shape)

        # 5. Unpack to spatial feature map
        f_att_spatial = self.unpack(f_att, f_t.shape[2:])   # [B, C, D, H, W]
        print("f_att_spatial.shape:", f_att_spatial.shape)

        # 6. Residual connection with input feature map + ReLU
        f_cam = self.relu(f_att_spatial + f_t)

        return f_cam