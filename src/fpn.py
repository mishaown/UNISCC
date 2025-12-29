"""
Feature Pyramid Network (FPN) for UniSCC v5.0

Converts Swin-B multi-stage outputs to unified 256-channel pyramid.
Implements top-down pathway with lateral connections.

Architecture:
    Input: Swin-B stages [128, 256, 512, 1024] channels at resolutions [64, 32, 16, 8]
    Output: Unified pyramid {P2:64², P3:32², P4:16², P5:8²} all with 256 channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class FeaturePyramidNetwork(nn.Module):
    """
    FPN for UniSCC v5.0 multi-scale architecture.

    Converts Swin-B encoder outputs to a feature pyramid with unified channel dimension.
    Uses top-down pathway with lateral connections for multi-scale feature fusion.

    Args:
        in_channels_list: Input channels from Swin stages [C1, C2, C3, C4]
                         Default: [128, 256, 512, 1024] for Swin-B
        out_channels: Unified output channels for all pyramid levels (default: 256)
        use_gn: Use GroupNorm instead of BatchNorm (default: True)
        num_groups: Number of groups for GroupNorm (default: 32)
    """

    def __init__(
        self,
        in_channels_list: List[int] = [128, 256, 512, 1024],
        out_channels: int = 256,
        use_gn: bool = True,
        num_groups: int = 32
    ):
        super().__init__()

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_levels = len(in_channels_list)

        # Lateral connections: 1x1 conv to unify channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels_list
        ])

        # Output convolutions: 3x3 conv to reduce aliasing after addition
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups, out_channels) if use_gn else nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(self.num_levels)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize conv weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Build feature pyramid from multi-stage encoder features.

        Args:
            features: List of [C1, C2, C3, C4] features from encoder stages
                     - C1: [B, 128, 64, 64]
                     - C2: [B, 256, 32, 32]
                     - C3: [B, 512, 16, 16]
                     - C4: [B, 1024, 8, 8]

        Returns:
            dict: Feature pyramid with keys 'P2', 'P3', 'P4', 'P5'
                  All outputs have shape [B, 256, Hi, Wi]
        """
        assert len(features) == self.num_levels, \
            f"Expected {self.num_levels} features, got {len(features)}"

        # Apply lateral connections
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Top-down pathway: start from highest level and add upsampled features
        # Process from top (smallest) to bottom (largest)
        for i in range(self.num_levels - 1, 0, -1):
            # Upsample higher level to match lower level size
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
            # Add to lower level
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply output convolutions to reduce aliasing
        outputs = [
            output_conv(laterals[i])
            for i, output_conv in enumerate(self.output_convs)
        ]

        # Return as dict with standard FPN naming
        # P2 is highest resolution, P5 is lowest
        return {
            'P2': outputs[0],  # [B, 256, 64, 64]
            'P3': outputs[1],  # [B, 256, 32, 32]
            'P4': outputs[2],  # [B, 256, 16, 16]
            'P5': outputs[3],  # [B, 256, 8, 8]
        }


class LightweightFPN(nn.Module):
    """
    Lightweight FPN variant with fewer parameters.

    Uses depthwise separable convolutions for efficiency.
    Suitable for memory-constrained scenarios.
    """

    def __init__(
        self,
        in_channels_list: List[int] = [128, 256, 512, 1024],
        out_channels: int = 256
    ):
        super().__init__()

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_levels = len(in_channels_list)

        # Lateral connections: 1x1 conv
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False)
            for in_ch in in_channels_list
        ])

        # Depthwise separable output convolutions
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                # Depthwise conv
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                # Pointwise conv
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(self.num_levels)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build lightweight feature pyramid."""
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down
        for i in range(self.num_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )

        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        return {
            'P2': outputs[0],
            'P3': outputs[1],
            'P4': outputs[2],
            'P5': outputs[3],
        }
