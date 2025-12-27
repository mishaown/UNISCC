"""
Temporal Difference Transformer (TDT)
Bidirectional cross-temporal attention for explicit change modeling

v3.0 Update:
- forward() now returns dict with 'diff', 'feat_t0_enhanced', 'feat_t1_enhanced'
- Enhanced features are used by DualSemanticHead for predicting both sem_A and sem_B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union


class CrossTemporalAttention(nn.Module):
    """Cross-temporal attention: query from one time, key/value from another"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor):
        """
        Args:
            query: [B, N, C] features from source time
            key_value: [B, N, C] features from target time
        Returns:
            output: [B, N, C] attended features
        """
        B, N, C = query.shape
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention and project
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out


class TDTBlock(nn.Module):
    """Single TDT block with bidirectional cross-temporal attention"""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        # Layer norms
        self.norm1_t0 = nn.LayerNorm(dim)
        self.norm1_t1 = nn.LayerNorm(dim)
        self.norm2_t0 = nn.LayerNorm(dim)
        self.norm2_t1 = nn.LayerNorm(dim)
        
        # Cross-temporal attention (forward and backward)
        self.attn_forward = CrossTemporalAttention(dim, num_heads, dropout)
        self.attn_backward = CrossTemporalAttention(dim, num_heads, dropout)
        
        # MLP
        mlp_dim = int(dim * mlp_ratio)
        self.mlp_t0 = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.mlp_t1 = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor):
        """
        Bidirectional cross-temporal attention
        
        Args:
            feat_t0, feat_t1: [B, N, C] features from t0 and t1
        Returns:
            Updated features for both times
        """
        # Forward attention: t0 queries, t1 keys/values
        attn_out = self.attn_forward(self.norm1_t0(feat_t0), self.norm1_t1(feat_t1))
        feat_t0 = feat_t0 + attn_out
        feat_t0 = feat_t0 + self.mlp_t0(self.norm2_t0(feat_t0))
        
        # Backward attention: t1 queries, t0 keys/values
        attn_out = self.attn_backward(self.norm1_t1(feat_t1), self.norm1_t0(feat_t0))
        feat_t1 = feat_t1 + attn_out
        feat_t1 = feat_t1 + self.mlp_t1(self.norm2_t1(feat_t1))
        
        return feat_t0, feat_t1


class TemporalDifferenceTransformer(nn.Module):
    """
    TDT: Explicit temporal modeling through bidirectional cross-attention
    Captures "what appeared" (t0→t1) and "what disappeared" (t1→t0)

    v3.0: Returns enhanced features for both time steps in addition to diff,
    enabling DualSemanticHead to predict both before and after semantics.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim

        # Learnable temporal embeddings
        self.temporal_embed_t0 = nn.Parameter(torch.zeros(1, 1, dim))
        self.temporal_embed_t1 = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.temporal_embed_t0, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed_t1, std=0.02)

        # TDT blocks
        self.blocks = nn.ModuleList([
            TDTBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Output projection for diff
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        # v3.0: Additional projections for enhanced features
        self.norm_t0 = nn.LayerNorm(dim)
        self.norm_t1 = nn.LayerNorm(dim)
        self.proj_t0 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.proj_t1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(
        self,
        feat_t0: torch.Tensor,
        feat_t1: torch.Tensor
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute temporal difference features AND enhanced temporal features.

        Args:
            feat_t0, feat_t1: [B, C, H', W'] features from encoder

        Returns:
            dict with:
                - 'diff': [B, C, H, W] temporal difference features
                - 'feat_t0_enhanced': [B, C, H, W] enhanced t0 features (for sem_A)
                - 'feat_t1_enhanced': [B, C, H, W] enhanced t1 features (for sem_B)
        """
        # Reshape spatial to sequence: [B, C, H, W] -> [B, N, C]
        B, C, H, W = feat_t0.shape
        feat_t0_seq = feat_t0.flatten(2).transpose(1, 2)  # [B, H*W, C]
        feat_t1_seq = feat_t1.flatten(2).transpose(1, 2)

        # Add temporal embeddings
        N = feat_t0_seq.shape[1]
        feat_t0_seq = feat_t0_seq + self.temporal_embed_t0.expand(B, N, -1)
        feat_t1_seq = feat_t1_seq + self.temporal_embed_t1.expand(B, N, -1)

        # Store original for difference computation
        orig_t0 = feat_t0_seq
        orig_t1 = feat_t1_seq

        # Apply TDT blocks (bidirectional cross-temporal attention)
        for block in self.blocks:
            feat_t0_seq, feat_t1_seq = block(feat_t0_seq, feat_t1_seq)

        # Compute difference: what changed
        diff_forward = feat_t0_seq - orig_t0  # What appeared/changed (t0→t1)
        diff_backward = feat_t1_seq - orig_t1  # What disappeared/changed (t1→t0)

        # Adaptive fusion of bidirectional differences
        diff = (diff_forward + diff_backward) / 2

        # Output projection for diff
        diff = self.proj(self.norm(diff))

        # v3.0: Project enhanced temporal features
        feat_t0_enhanced = self.proj_t0(self.norm_t0(feat_t0_seq))
        feat_t1_enhanced = self.proj_t1(self.norm_t1(feat_t1_seq))

        # Reshape back to spatial: [B, N, C] -> [B, C, H, W]
        diff = diff.transpose(1, 2).view(B, C, H, W)
        feat_t0_enhanced = feat_t0_enhanced.transpose(1, 2).view(B, C, H, W)
        feat_t1_enhanced = feat_t1_enhanced.transpose(1, 2).view(B, C, H, W)

        return {
            'diff': diff,
            'feat_t0_enhanced': feat_t0_enhanced,
            'feat_t1_enhanced': feat_t1_enhanced
        }
