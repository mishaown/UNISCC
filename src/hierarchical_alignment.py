"""
Hierarchical Alignment Module for UniSCC v5.0

Per-scale feature alignment with skip connections.
Aligns t0 features to t1 reference frame at each pyramid level independently.

Key features:
- Cross-attention based alignment at each scale
- Confidence estimation for alignment quality
- Skip connection: output = conf * aligned + (1-conf) * original
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class ScaleAligner(nn.Module):
    """
    Cross-attention based alignment for a single scale.

    Uses cross-attention where:
    - Query: t1 features (reference frame)
    - Key/Value: t0 features (to be aligned)

    Args:
        dim: Feature dimension (256 for all pyramid levels)
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections for cross-attention
        # Q from t1 (reference), K/V from t0 (to align)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # Layer norms
        self.norm_t0 = nn.LayerNorm(dim)
        self.norm_t1 = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        # FFN for refinement
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feat_t0: torch.Tensor,
        feat_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        Align t0 features to t1 reference frame.

        Args:
            feat_t0: [B, C, H, W] before-change features
            feat_t1: [B, C, H, W] after-change features (reference)

        Returns:
            aligned_t0: [B, C, H, W] aligned before-change features
        """
        B, C, H, W = feat_t0.shape

        # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        t0_seq = feat_t0.flatten(2).transpose(1, 2)
        t1_seq = feat_t1.flatten(2).transpose(1, 2)

        # Normalize
        t0_norm = self.norm_t0(t0_seq)
        t1_norm = self.norm_t1(t1_seq)

        # Cross-attention: Q from t1, K/V from t0
        Q = self.q_proj(t1_norm).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(t0_norm).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(t0_norm).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to get aligned features
        aligned = (attn @ V).transpose(1, 2).reshape(B, -1, C)
        aligned = self.out_proj(aligned)

        # Residual with original t0
        aligned = t0_seq + aligned

        # FFN refinement
        aligned = aligned + self.ffn(self.norm_out(aligned))

        # Reshape back to spatial: [B, H*W, C] -> [B, C, H, W]
        aligned = aligned.transpose(1, 2).reshape(B, C, H, W)

        return aligned


class HierarchicalAlignmentV5(nn.Module):
    """
    Hierarchical Alignment Module for UniSCC v5.0.

    Performs per-scale alignment with confidence-weighted skip connections.
    Each pyramid level is aligned independently, preserving scale-specific details.

    Args:
        dim: Feature dimension (256 for all pyramid levels)
        num_heads: Number of attention heads
        num_scales: Number of pyramid levels (4: P2, P3, P4, P5)
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_scales: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dim = dim
        self.num_scales = num_scales
        self.scale_names = ['P2', 'P3', 'P4', 'P5'][:num_scales]

        # Per-scale alignment modules
        self.aligners = nn.ModuleDict({
            name: ScaleAligner(dim, num_heads, dropout)
            for name in self.scale_names
        })

        # Per-scale confidence estimators
        # Takes concatenated [aligned_t0, feat_t1] and outputs confidence map
        self.confidence_nets = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(dim * 2, dim // 2, 3, padding=1),
                nn.GroupNorm(16, dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 2, dim // 4, 3, padding=1),
                nn.GroupNorm(8, dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, 1, 1),
                nn.Sigmoid()
            )
            for name in self.scale_names
        })

        self._init_weights()

    def _init_weights(self):
        """Initialize confidence nets to output ~0.5 initially."""
        for name in self.scale_names:
            # Initialize last conv bias to 0 (sigmoid(0) = 0.5)
            nn.init.zeros_(self.confidence_nets[name][-2].weight)
            nn.init.zeros_(self.confidence_nets[name][-2].bias)

    def forward(
        self,
        pyramid_t0: Dict[str, torch.Tensor],
        pyramid_t1: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Align t0 pyramid to t1 pyramid at each scale.

        Args:
            pyramid_t0: dict {'P2': [B,256,64,64], 'P3': [B,256,32,32], ...}
            pyramid_t1: dict {'P2': [B,256,64,64], 'P3': [B,256,32,32], ...}

        Returns:
            aligned_pyramid: dict of aligned t0 features per scale
            confidence_pyramid: dict of confidence maps per scale [B, 1, H, W]
        """
        aligned_pyramid = {}
        confidence_pyramid = {}

        for name in self.scale_names:
            feat_t0 = pyramid_t0[name]
            feat_t1 = pyramid_t1[name]

            # Align t0 to t1 at this scale
            aligned_t0 = self.aligners[name](feat_t0, feat_t1)

            # Estimate alignment confidence
            concat = torch.cat([aligned_t0, feat_t1], dim=1)
            confidence = self.confidence_nets[name](concat)

            # Skip connection: blend aligned and original based on confidence
            # High confidence -> use aligned, Low confidence -> preserve original
            output = confidence * aligned_t0 + (1 - confidence) * feat_t0

            aligned_pyramid[name] = output
            confidence_pyramid[name] = confidence

        return aligned_pyramid, confidence_pyramid


class EfficientHierarchicalAlignment(nn.Module):
    """
    Memory-efficient hierarchical alignment using shared weights.

    Shares alignment weights across scales but uses scale-specific
    confidence estimation. Reduces parameter count significantly.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_scales: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dim = dim
        self.num_scales = num_scales
        self.scale_names = ['P2', 'P3', 'P4', 'P5'][:num_scales]

        # Shared alignment module
        self.shared_aligner = ScaleAligner(dim, num_heads, dropout)

        # Scale-specific confidence estimation
        self.confidence_net = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 2, 3, padding=1),
            nn.GroupNorm(16, dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # Scale embeddings to differentiate scales
        self.scale_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, dim, 1, 1))
            for name in self.scale_names
        })

        for name in self.scale_names:
            nn.init.trunc_normal_(self.scale_embeddings[name], std=0.02)

    def forward(
        self,
        pyramid_t0: Dict[str, torch.Tensor],
        pyramid_t1: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Efficient alignment with shared weights."""
        aligned_pyramid = {}
        confidence_pyramid = {}

        for name in self.scale_names:
            feat_t0 = pyramid_t0[name]
            feat_t1 = pyramid_t1[name]

            # Add scale embedding
            scale_embed = self.scale_embeddings[name]
            feat_t0_scaled = feat_t0 + scale_embed
            feat_t1_scaled = feat_t1 + scale_embed

            # Align using shared module
            aligned_t0 = self.shared_aligner(feat_t0_scaled, feat_t1_scaled)

            # Estimate confidence
            concat = torch.cat([aligned_t0, feat_t1], dim=1)
            confidence = self.confidence_net(concat)

            # Skip connection
            output = confidence * aligned_t0 + (1 - confidence) * feat_t0

            aligned_pyramid[name] = output
            confidence_pyramid[name] = confidence

        return aligned_pyramid, confidence_pyramid
