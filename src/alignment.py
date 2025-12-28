"""
Bitemporal Image Alignment Module

Aligns features from before (t0) and after (t1) images to handle
spatial misalignments in satellite imagery caused by:
- Different acquisition angles
- Sensor positioning differences
- Atmospheric distortions
- Registration errors

Implements cross-attention based alignment which learns to find
correspondences between temporal features without explicit
geometric transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class CrossAttentionAlignment(nn.Module):
    """
    Cross-Attention based feature alignment for bitemporal images.

    Uses cross-attention to find correspondences between t0 and t1 features,
    then warps t0 features to align with t1 (reference frame).

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_relative_pos: Use relative position encoding
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_relative_pos: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_relative_pos = use_relative_pos

        # Query from t1 (reference), Key/Value from t0 (to be aligned)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Relative position bias (optional but helps with spatial relationships)
        if use_relative_pos:
            self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 64, 64))
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # FFN for refinement
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        feat_t0: torch.Tensor,
        feat_t1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align t0 features to t1 reference frame.

        Args:
            feat_t0: [B, C, H, W] before-change features
            feat_t1: [B, C, H, W] after-change features (reference)

        Returns:
            feat_t0_aligned: [B, C, H, W] aligned before-change features
            attention_weights: [B, num_heads, H*W, H*W] attention maps
        """
        B, C, H, W = feat_t0.shape

        # Reshape to sequence: [B, H*W, C]
        t0_seq = feat_t0.flatten(2).transpose(1, 2)
        t1_seq = feat_t1.flatten(2).transpose(1, 2)

        # Normalize
        t0_norm = self.norm1(t0_seq)
        t1_norm = self.norm1(t1_seq)

        # Cross-attention: Q from t1, K/V from t0
        Q = self.q_proj(t1_norm).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(t0_norm).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(t0_norm).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale

        # Add relative position bias
        if self.use_relative_pos and H * W <= 64 * 64:
            rel_bias = self.rel_pos_bias[:, :H*W, :H*W]
            attn = attn + rel_bias.unsqueeze(0)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to get aligned features
        aligned = (attn_weights @ V).transpose(1, 2).reshape(B, -1, C)
        aligned = self.out_proj(aligned)

        # Residual connection with original t0
        aligned = t0_seq + aligned

        # FFN refinement
        aligned = aligned + self.ffn(self.norm2(aligned))

        # Reshape back to spatial
        feat_t0_aligned = aligned.transpose(1, 2).reshape(B, C, H, W)

        return feat_t0_aligned, attn_weights


class DeformableAlignment(nn.Module):
    """
    Deformable convolution based alignment for bitemporal images.

    Learns spatial offsets to warp t0 features to match t1.
    More efficient than attention for high-resolution features.

    Args:
        dim: Feature dimension
        num_groups: Number of offset groups
    """

    def __init__(self, dim: int = 512, num_groups: int = 8):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups

        # Offset prediction from concatenated features
        self.offset_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, 2 * num_groups * 9, 3, padding=1)  # 2D offsets for 3x3
        )

        # Modulation mask
        self.mask_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, num_groups * 9, 3, padding=1),
            nn.Sigmoid()
        )

        # Feature transformation after alignment
        self.align_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=num_groups)

        self._init_weights()

    def _init_weights(self):
        # Initialize offsets to zero (identity transform initially)
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)

    def forward(
        self,
        feat_t0: torch.Tensor,
        feat_t1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align t0 features using deformable sampling.

        Args:
            feat_t0: [B, C, H, W] before-change features
            feat_t1: [B, C, H, W] after-change features (reference)

        Returns:
            feat_t0_aligned: [B, C, H, W] aligned features
            offsets: [B, 2*G*9, H, W] learned offsets
        """
        B, C, H, W = feat_t0.shape

        # Concatenate for offset prediction
        concat = torch.cat([feat_t0, feat_t1], dim=1)

        # Predict offsets and mask
        offsets = self.offset_conv(concat)
        mask = self.mask_conv(concat)

        # Apply deformable sampling (simplified grid sample version)
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat_t0.device),
            torch.linspace(-1, 1, W, device=feat_t0.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Average offsets across groups for grid sampling
        offset_x = offsets[:, 0::2, :, :].mean(dim=1, keepdim=True)
        offset_y = offsets[:, 1::2, :, :].mean(dim=1, keepdim=True)
        offset_grid = torch.cat([offset_x, offset_y], dim=1)
        offset_grid = offset_grid.permute(0, 2, 3, 1) * 0.1  # Scale offsets

        # Warp features
        warped_grid = grid + offset_grid
        feat_t0_aligned = F.grid_sample(
            feat_t0, warped_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        # Apply mask and residual
        mask_avg = mask.mean(dim=1, keepdim=True)
        feat_t0_aligned = feat_t0_aligned * mask_avg + feat_t0 * (1 - mask_avg)

        # Final convolution
        feat_t0_aligned = self.align_conv(feat_t0_aligned)

        return feat_t0_aligned, offsets


class HierarchicalAlignment(nn.Module):
    """
    Multi-scale hierarchical alignment module.

    Performs alignment at multiple scales for robust correspondence
    matching, from coarse to fine.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        num_scales: Number of alignment scales
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        num_scales: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_scales = num_scales

        # Multi-scale alignment modules
        self.aligners = nn.ModuleList([
            CrossAttentionAlignment(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                use_relative_pos=True
            )
            for _ in range(num_scales)
        ])

        # Scale-specific projections
        self.down_projs = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, stride=2, padding=1) if i > 0 else nn.Identity()
            for i in range(num_scales)
        ])

        self.up_projs = nn.ModuleList([
            nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1) if i > 0 else nn.Identity()
            for i in range(num_scales)
        ])

        # Fusion
        self.fusion = nn.Conv2d(dim * num_scales, dim, 1)

    def forward(
        self,
        feat_t0: torch.Tensor,
        feat_t1: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Multi-scale alignment.

        Args:
            feat_t0: [B, C, H, W] before-change features
            feat_t1: [B, C, H, W] after-change features

        Returns:
            aligned: [B, C, H, W] aligned features
            info: Dictionary with intermediate results
        """
        B, C, H, W = feat_t0.shape
        aligned_features = []
        attention_maps = []

        for i in range(self.num_scales):
            # Downsample if needed
            t0_scale = self.down_projs[i](feat_t0) if i > 0 else feat_t0
            t1_scale = self.down_projs[i](feat_t1) if i > 0 else feat_t1

            for j in range(i):
                t0_scale = F.avg_pool2d(t0_scale, 2)
                t1_scale = F.avg_pool2d(t1_scale, 2)

            # Align at this scale
            aligned_scale, attn = self.aligners[i](t0_scale, t1_scale)
            attention_maps.append(attn)

            # Upsample back to original resolution
            for j in range(i):
                aligned_scale = F.interpolate(
                    aligned_scale, scale_factor=2, mode='bilinear', align_corners=True
                )
            aligned_scale = self.up_projs[i](aligned_scale) if i > 0 else aligned_scale

            # Ensure same size
            if aligned_scale.shape[-2:] != (H, W):
                aligned_scale = F.interpolate(aligned_scale, size=(H, W), mode='bilinear', align_corners=True)

            aligned_features.append(aligned_scale)

        # Fuse multi-scale alignments
        fused = torch.cat(aligned_features, dim=1)
        aligned = self.fusion(fused)

        return aligned, {'attention_maps': attention_maps}


class FeatureAlignment(nn.Module):
    """
    Main alignment module for UniSCC.

    Combines cross-attention alignment with residual connections
    for robust bitemporal feature alignment.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        alignment_type: 'cross_attention', 'deformable', or 'hierarchical'
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        alignment_type: str = 'cross_attention',
        dropout: float = 0.1
    ):
        super().__init__()
        self.alignment_type = alignment_type

        if alignment_type == 'cross_attention':
            self.aligner = CrossAttentionAlignment(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout
            )
        elif alignment_type == 'deformable':
            self.aligner = DeformableAlignment(dim=dim)
        elif alignment_type == 'hierarchical':
            self.aligner = HierarchicalAlignment(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown alignment type: {alignment_type}")

        # Confidence estimation for alignment quality
        self.confidence = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        feat_t0: torch.Tensor,
        feat_t1: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Align bitemporal features.

        Args:
            feat_t0: [B, C, H, W] before-change features
            feat_t1: [B, C, H, W] after-change features

        Returns:
            Dictionary with:
                - feat_t0_aligned: Aligned before-change features
                - feat_t1: After-change features (unchanged)
                - confidence: Alignment confidence map
                - diff: Difference features after alignment
        """
        # Align t0 to t1
        if self.alignment_type == 'hierarchical':
            feat_t0_aligned, info = self.aligner(feat_t0, feat_t1)
        else:
            feat_t0_aligned, _ = self.aligner(feat_t0, feat_t1)

        # Compute alignment confidence
        concat = torch.cat([feat_t0_aligned, feat_t1], dim=1)
        confidence = self.confidence(concat)

        # Weighted alignment based on confidence
        feat_t0_final = confidence * feat_t0_aligned + (1 - confidence) * feat_t0

        # Compute difference after alignment
        diff = feat_t1 - feat_t0_final

        return {
            'feat_t0_aligned': feat_t0_final,
            'feat_t1': feat_t1,
            'confidence': confidence,
            'diff': diff
        }
