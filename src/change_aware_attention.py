"""
Change-Aware Attention Module for UniSCC v5.0

Generates spatial attention map indicating "where changes occurred"
and enhances features with channel attention.

Outputs:
    - change_features: Enhanced features for CD head
    - change_attention: Spatial attention map for caption decoder guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SpatialAttention(nn.Module):
    """
    Spatial attention branch.

    Generates a 2D attention map highlighting changed regions.
    Uses progressive channel reduction to create a single-channel map.
    """

    def __init__(self, in_channels: int = 512):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate spatial attention map.

        Args:
            x: [B, C, H, W] input features

        Returns:
            [B, 1, H, W] spatial attention map in [0, 1]
        """
        return self.attention(x)


class ChannelAttention(nn.Module):
    """
    Channel attention branch (SE-like).

    Learns which feature channels are most important for change detection.
    Uses global average pooling followed by FC layers.
    """

    def __init__(self, in_channels: int = 512, reduction: int = 16):
        super().__init__()

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate channel attention weights.

        Args:
            x: [B, C, H, W] input features

        Returns:
            [B, C, 1, 1] channel attention weights in [0, 1]
        """
        return self.attention(x)


class ChangeAwareAttention(nn.Module):
    """
    Change-Aware Attention Module for UniSCC v5.0.

    Combines spatial and channel attention to:
    1. Identify WHERE changes occurred (spatial attention)
    2. Identify WHAT type of change (channel attention)

    The spatial attention map is also used by the caption decoder
    to focus on changed regions during generation.

    Args:
        dim: Input feature dimension (512 from fused TDT output)
        reduction: Channel reduction ratio for channel attention
    """

    def __init__(self, dim: int = 512, reduction: int = 16):
        super().__init__()

        self.dim = dim

        # Spatial attention: where to look
        self.spatial_attn = SpatialAttention(dim)

        # Channel attention: what to look for
        self.channel_attn = ChannelAttention(dim, reduction)

        # Feature refinement after attention
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, fused_diff: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply change-aware attention.

        Args:
            fused_diff: [B, 512, 256, 256] fused difference features from TDT

        Returns:
            dict with:
                - 'change_features': [B, 512, H, W] enhanced features
                - 'change_attention': [B, 1, H, W] spatial attention map
        """
        # Compute attention maps
        spatial_attn = self.spatial_attn(fused_diff)  # [B, 1, H, W]
        channel_attn = self.channel_attn(fused_diff)  # [B, C, 1, 1]

        # Apply both attentions: feature * channel_weight * spatial_weight
        attended = fused_diff * channel_attn * spatial_attn

        # Residual connection
        change_features = fused_diff + attended

        # Refine
        change_features = self.refine(change_features)

        return {
            'change_features': change_features,
            'change_attention': spatial_attn
        }


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) variant.

    Alternative attention mechanism that applies channel and spatial
    attention sequentially rather than in parallel.
    """

    def __init__(self, dim: int = 512, reduction: int = 16):
        super().__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False)
        )

        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply CBAM attention."""
        # Channel attention
        avg_out = self.channel_mlp(self.avg_pool(x))
        max_out = self.channel_mlp(self.max_pool(x))
        channel_attn = torch.sigmoid(avg_out + max_out)
        x = x * channel_attn

        # Spatial attention
        avg_feat = x.mean(dim=1, keepdim=True)
        max_feat = x.max(dim=1, keepdim=True)[0]
        spatial_feat = torch.cat([avg_feat, max_feat], dim=1)
        spatial_attn = self.spatial_conv(spatial_feat)
        x = x * spatial_attn

        return {
            'change_features': x,
            'change_attention': spatial_attn
        }


class MultiHeadChangeAttention(nn.Module):
    """
    Multi-head variant of change-aware attention.

    Uses multiple attention heads to capture different aspects of change,
    then aggregates them for the final attention map.
    """

    def __init__(self, dim: int = 512, num_heads: int = 4):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Per-head spatial attention
        self.head_attns = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, self.head_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_dim, self.head_dim // 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_dim // 4, 1, 1),
                nn.Sigmoid()
            )
            for _ in range(num_heads)
        ])

        # Aggregation: combine head attentions
        self.aggregate = nn.Sequential(
            nn.Conv2d(num_heads, num_heads, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_heads, 1, 1),
            nn.Sigmoid()
        )

        # Channel attention
        self.channel_attn = ChannelAttention(dim)

        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, fused_diff: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply multi-head change attention."""
        # Compute per-head attention
        head_attns = [attn(fused_diff) for attn in self.head_attns]
        head_stack = torch.cat(head_attns, dim=1)  # [B, num_heads, H, W]

        # Aggregate heads
        spatial_attn = self.aggregate(head_stack)  # [B, 1, H, W]

        # Channel attention
        channel_attn = self.channel_attn(fused_diff)  # [B, C, 1, 1]

        # Apply attentions
        attended = fused_diff * channel_attn * spatial_attn

        # Residual and refine
        change_features = self.refine(fused_diff + attended)

        return {
            'change_features': change_features,
            'change_attention': spatial_attn
        }
