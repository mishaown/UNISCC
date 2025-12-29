"""
Multi-Task Change Detection Head for UniSCC v5.0

Predicts both semantic classification and change magnitude.

Outputs:
    - cd_logits: [B, K, H, W] semantic class predictions
    - magnitude: [B, 1, H, W] change intensity (0-1)

Uses hierarchical prompts with per-scale decoders and learnable
scale fusion weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ScaleDecoder(nn.Module):
    """
    Decoder for a single scale that upsamples to target resolution.

    Args:
        in_channels: Input channels (512 from change_features)
        out_channels: Output channels (256 to match prompts)
        scale_factor: Upsampling factor (1, 2, 4, or 8)
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 256,
        scale_factor: int = 1
    ):
        super().__init__()

        self.scale_factor = scale_factor

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, target_size: int = 256) -> torch.Tensor:
        """
        Decode and upsample to target resolution.

        Args:
            x: [B, C, H, W] input features
            target_size: Target spatial size (default 256)

        Returns:
            [B, out_channels, target_size, target_size]
        """
        x = self.decoder(x)

        if x.shape[-1] != target_size:
            x = F.interpolate(
                x,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )

        return x


class MultiTaskCDHead(nn.Module):
    """
    Multi-Task Change Detection Head for UniSCC v5.0.

    Predicts both semantic classification and change magnitude using
    hierarchical prompts with per-scale decoders.

    Architecture:
        1. Per-scale decoders process change_features
        2. Cosine similarity with scale-specific prompts
        3. Learnable scale fusion for final logits
        4. Separate magnitude prediction branch

    Args:
        in_channels: Input channels from change_features (512)
        hidden_channels: Decoder hidden channels (256)
        num_classes: Number of semantic classes (7 for SECOND-CC, 3 for LEVIR-MCI)
        num_scales: Number of pyramid scales (4)
        target_size: Output spatial size (256)
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        num_classes: int = 7,
        num_scales: int = 4,
        target_size: int = 256
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_scales = num_scales
        self.target_size = target_size
        self.scale_names = ['P2', 'P3', 'P4', 'P5'][:num_scales]

        # Per-scale decoders
        self.decoders = nn.ModuleDict({
            name: ScaleDecoder(in_channels, hidden_channels, scale_factor=2**i)
            for i, name in enumerate(self.scale_names)
        })

        # Feature projection to match prompt dimension
        self.feature_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)

        # Learnable scale fusion weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        # Temperature for similarity scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

        # Magnitude prediction branch
        self.magnitude_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        change_features: torch.Tensor,
        hierarchical_prompts: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-task forward pass.

        Args:
            change_features: [B, 512, 256, 256] from ChangeAwareAttention
            hierarchical_prompts: dict with 'prompts_P2', 'prompts_P3', etc.
                                 Each tensor is [K, 256]

        Returns:
            dict with:
                - 'cd_logits': [B, K, 256, 256] semantic classification
                - 'magnitude': [B, 1, 256, 256] change intensity
        """
        B = change_features.shape[0]
        scale_logits = []

        # Process each scale
        for i, name in enumerate(self.scale_names):
            # Decode features for this scale
            decoded = self.decoders[name](change_features, self.target_size)
            decoded = self.feature_proj(decoded)  # [B, 256, 256, 256]

            # Normalize features
            features_norm = F.normalize(decoded, dim=1)  # [B, 256, 256, 256]

            # Get prompts for this scale
            prompts = hierarchical_prompts[f'prompts_{name}']  # [K, 256]
            prompts_norm = F.normalize(prompts, dim=1)  # [K, 256]

            # Cosine similarity: [B, 256, 256, 256] x [K, 256] -> [B, K, 256, 256]
            logits = torch.einsum('bdhw,kd->bkhw', features_norm, prompts_norm)
            logits = logits * self.logit_scale.exp()

            scale_logits.append(logits)

        # Fuse multi-scale logits with learned weights
        weights = F.softmax(self.scale_weights, dim=0)
        cd_logits = sum(w * logits for w, logits in zip(weights, scale_logits))

        # Predict magnitude
        magnitude = self.magnitude_head(change_features)

        return {
            'cd_logits': cd_logits,
            'magnitude': magnitude
        }


class LightweightMultiTaskCDHead(nn.Module):
    """
    Lightweight variant with shared decoder weights.

    Uses a single shared decoder with scale embeddings instead of
    per-scale decoders. Reduces parameter count significantly.
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        num_classes: int = 7,
        num_scales: int = 4,
        target_size: int = 256
    ):
        super().__init__()

        self.num_scales = num_scales
        self.target_size = target_size
        self.scale_names = ['P2', 'P3', 'P4', 'P5'][:num_scales]

        # Shared decoder
        self.shared_decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Scale embeddings
        self.scale_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
            for name in self.scale_names
        })

        for name in self.scale_names:
            nn.init.trunc_normal_(self.scale_embeddings[name], std=0.02)

        # Fusion weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        # Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

        # Magnitude head
        self.magnitude_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        change_features: torch.Tensor,
        hierarchical_prompts: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Lightweight multi-task forward."""
        B = change_features.shape[0]

        # Shared decoding
        decoded = self.shared_decoder(change_features)

        scale_logits = []
        for name in self.scale_names:
            # Add scale embedding
            scale_decoded = decoded + self.scale_embeddings[name]

            # Normalize
            features_norm = F.normalize(scale_decoded, dim=1)

            # Get prompts
            prompts = hierarchical_prompts[f'prompts_{name}']
            prompts_norm = F.normalize(prompts, dim=1)

            # Similarity
            logits = torch.einsum('bdhw,kd->bkhw', features_norm, prompts_norm)
            logits = logits * self.logit_scale.exp()

            scale_logits.append(logits)

        # Fuse
        weights = F.softmax(self.scale_weights, dim=0)
        cd_logits = sum(w * l for w, l in zip(weights, scale_logits))

        # Magnitude
        magnitude = self.magnitude_head(change_features)

        return {
            'cd_logits': cd_logits,
            'magnitude': magnitude
        }


class MultiTaskCDHeadWithBoundary(nn.Module):
    """
    Extended Multi-Task CD Head with boundary prediction.

    Adds a boundary prediction branch for improved edge detection.
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        num_classes: int = 7,
        num_scales: int = 4,
        target_size: int = 256
    ):
        super().__init__()

        # Base multi-task head
        self.base_head = MultiTaskCDHead(
            in_channels, hidden_channels, num_classes, num_scales, target_size
        )

        # Boundary prediction branch
        self.boundary_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        change_features: torch.Tensor,
        hierarchical_prompts: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward with boundary prediction."""
        outputs = self.base_head(change_features, hierarchical_prompts)

        # Add boundary prediction
        outputs['boundary'] = self.boundary_head(change_features)

        return outputs
