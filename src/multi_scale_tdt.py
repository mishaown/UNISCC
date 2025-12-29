"""
Multi-Scale Temporal Difference Transformer (TDT) for UniSCC v5.0

Computes temporal differences at each pyramid level and fuses them
into a unified change representation.

Architecture:
    1. Per-scale TDT: Separate TDT for each pyramid level
    2. Upsample: All scales upsampled to target resolution (256x256)
    3. Fusion: Concatenate and fuse to 512 channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class LightweightTDTBlock(nn.Module):
    """
    Lightweight TDT block for per-scale processing.

    Uses efficient cross-temporal attention to model changes.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Cross-temporal attention: t0 -> t1
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Lightweight MLP
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feat_t0: torch.Tensor,
        feat_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-temporal features.

        Args:
            feat_t0: [B, N, C] aligned t0 features (sequence)
            feat_t1: [B, N, C] t1 features (sequence)

        Returns:
            diff: [B, N, C] difference features
        """
        B, N, C = feat_t0.shape

        # Normalize
        t0_norm = self.norm1(feat_t0)
        t1_norm = self.norm1(feat_t1)

        # Cross-attention: t0 queries, t1 keys/values
        # This captures "what in t1 corresponds to each position in t0"
        Q = self.q_proj(t0_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(t1_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(t1_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Get attended t1 features
        attended_t1 = (attn @ V).transpose(1, 2).reshape(B, N, C)
        attended_t1 = self.out_proj(attended_t1)

        # Compute difference
        diff = attended_t1 - feat_t0

        # MLP refinement
        diff = diff + self.mlp(self.norm2(diff))

        return diff


class ScaleTDT(nn.Module):
    """
    TDT for a single pyramid scale.

    Stacks multiple TDT blocks for deeper temporal modeling.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            LightweightTDTBlock(dim, num_heads, mlp_ratio=2.0, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        feat_t0: torch.Tensor,
        feat_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal difference at this scale.

        Args:
            feat_t0: [B, C, H, W] aligned t0 features
            feat_t1: [B, C, H, W] t1 features

        Returns:
            diff: [B, C, H, W] difference features
        """
        B, C, H, W = feat_t0.shape

        # Flatten to sequence
        t0_seq = feat_t0.flatten(2).transpose(1, 2)  # [B, H*W, C]
        t1_seq = feat_t1.flatten(2).transpose(1, 2)

        # Apply TDT blocks
        diff = t0_seq
        for block in self.blocks:
            diff = block(diff, t1_seq)

        diff = self.norm(diff)

        # Reshape back to spatial
        diff = diff.transpose(1, 2).reshape(B, C, H, W)

        return diff


class MultiScaleTDT(nn.Module):
    """
    Multi-Scale Temporal Difference Transformer for UniSCC v5.0.

    Computes temporal differences at each pyramid level and fuses them
    into a unified change representation.

    Architecture:
        1. Per-scale TDT modules process each pyramid level
        2. All outputs upsampled to target resolution (256x256)
        3. Concatenate along channels: [256*4] = 1024 channels
        4. Fuse to output dimension (512 channels)

    Args:
        dim: Feature dimension per scale (256)
        num_heads: Attention heads
        num_layers: TDT depth per scale
        output_dim: Fused output dimension (512)
        target_size: Target spatial size (256)
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        output_dim: int = 512,
        target_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.target_size = target_size
        self.scale_names = ['P2', 'P3', 'P4', 'P5']

        # Per-scale TDT modules
        self.tdt_modules = nn.ModuleDict({
            name: ScaleTDT(dim, num_heads, num_layers, dropout)
            for name in self.scale_names
        })

        # Fusion network: [256*4 channels] -> [512 channels]
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, output_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )

        # Scale-specific refinement after upsampling
        self.scale_refine = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )
            for name in self.scale_names
        })

    def forward(
        self,
        aligned_pyramid_t0: Dict[str, torch.Tensor],
        pyramid_t1: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale temporal differences.

        Args:
            aligned_pyramid_t0: dict of aligned t0 features per scale
            pyramid_t1: dict of t1 features per scale

        Returns:
            dict with:
                - 'fused_diff': [B, 512, 256, 256] fused change features
                - 'diff_pyramid': dict of per-scale diffs (for caption decoder)
        """
        diff_pyramid = {}
        upsampled_diffs = []

        for name in self.scale_names:
            feat_t0 = aligned_pyramid_t0[name]
            feat_t1 = pyramid_t1[name]

            # Compute difference at this scale
            diff = self.tdt_modules[name](feat_t0, feat_t1)
            diff_pyramid[name] = diff

            # Upsample to target size
            if diff.shape[-1] != self.target_size:
                diff_up = F.interpolate(
                    diff,
                    size=(self.target_size, self.target_size),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                diff_up = diff

            # Refine after upsampling
            diff_up = self.scale_refine[name](diff_up)
            upsampled_diffs.append(diff_up)

        # Concatenate all scales
        concat = torch.cat(upsampled_diffs, dim=1)  # [B, 256*4, 256, 256]

        # Fuse to output dimension
        fused_diff = self.fusion(concat)  # [B, 512, 256, 256]

        return {
            'fused_diff': fused_diff,
            'diff_pyramid': diff_pyramid
        }


class EfficientMultiScaleTDT(nn.Module):
    """
    Memory-efficient Multi-Scale TDT using shared weights.

    Shares TDT weights across scales with scale-specific embeddings.
    Significantly reduces parameter count.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        output_dim: int = 512,
        target_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.target_size = target_size
        self.scale_names = ['P2', 'P3', 'P4', 'P5']

        # Shared TDT module
        self.shared_tdt = ScaleTDT(dim, num_heads, num_layers, dropout)

        # Scale embeddings
        self.scale_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, dim, 1, 1))
            for name in self.scale_names
        })

        for name in self.scale_names:
            nn.init.trunc_normal_(self.scale_embeddings[name], std=0.02)

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, output_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        aligned_pyramid_t0: Dict[str, torch.Tensor],
        pyramid_t1: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Efficient multi-scale TDT with shared weights."""
        diff_pyramid = {}
        upsampled_diffs = []

        for name in self.scale_names:
            feat_t0 = aligned_pyramid_t0[name]
            feat_t1 = pyramid_t1[name]

            # Add scale embedding
            scale_embed = self.scale_embeddings[name]
            feat_t0_scaled = feat_t0 + scale_embed
            feat_t1_scaled = feat_t1 + scale_embed

            # Compute difference using shared module
            diff = self.shared_tdt(feat_t0_scaled, feat_t1_scaled)
            diff_pyramid[name] = diff

            # Upsample to target size
            if diff.shape[-1] != self.target_size:
                diff_up = F.interpolate(
                    diff,
                    size=(self.target_size, self.target_size),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                diff_up = diff

            upsampled_diffs.append(diff_up)

        # Fuse
        concat = torch.cat(upsampled_diffs, dim=1)
        fused_diff = self.fusion(concat)

        return {
            'fused_diff': fused_diff,
            'diff_pyramid': diff_pyramid
        }
