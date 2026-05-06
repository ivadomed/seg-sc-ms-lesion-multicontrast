import torch
import torch.nn as nn
# from mamba_ssm import Mamba # Requires mamba-ssm installation


class MCAM(nn.Module):
    def __init__(self, channels, num_patches, embedding_dim):
        super(MCAM, self).__init__()
        self.patch_embed = nn.Linear(channels, embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim)
        
        # Mamba blocks for temporal/spatial reasoning
        self.mamba_t = Mamba(d_model=embedding_dim, d_state=16, d_conv=4, expansion=2)
        self.mamba_prev = Mamba(d_model=embedding_dim, d_state=16, d_conv=4, expansion=2)
        
        # Cross-Attention to align I_t and I_{t-1}
        self.cross_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8)
        
        self.conv2d_refine = nn.Conv2d(1, 1, kernel_size=1) # Simplified for reshaping
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_t, f_prev, f_sem):
        # Flatten and embed spatial features into patches
        B, C, D, H, W = f_t.shape
        f_t_flat = f_t.view(B, C, -1).transpose(1, 2) # (B, N, E)
        f_prev_flat = f_prev.view(B, C, -1).transpose(1, 2)
        
        p_t = self.ln(self.patch_embed(f_t_flat))
        p_prev = self.ln(self.patch_embed(f_prev_flat))
        
        # SEM fusion (Element-wise addition with broadcasting after reshape)
        # SEM input is reshaped to align with patch embeddings
        f_sem_reshaped = f_sem.view(B, 1, -1, p_prev.size(-1)) 
        p_fused_prev = self.relu(p_prev.unsqueeze(1) + f_sem_reshaped)
        p_fused_prev = self.conv2d_refine(p_fused_prev).squeeze(1)
        
        # Mamba processing
        m_t = self.mamba_t(p_t)
        m_prev = self.mamba_prev(p_fused_prev)
        
        # Cross-attention alignment
        attn_out, _ = self.cross_attn(m_t.transpose(0, 1), m_prev.transpose(0, 1), m_prev.transpose(0, 1))
        f_att = attn_out.transpose(0, 1).view(B, C, D, H, W)
        
        # Residual connection
        return self.relu(f_t + f_att)