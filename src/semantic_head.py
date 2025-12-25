"""
Semantic Change Head with Shared Semantic Space

Unified change detection head that outputs a single change map:
- SECOND-CC: 7-class semantic segmentation (what the area became after change)
- LEVIR-MCI: 3-class change type (no_change, building, road)

Both use cosine similarity with learned semantic prompts for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SemanticDecoder(nn.Module):
    """Decoder that upsamples features to full resolution."""

    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__()

        # Progressive upsampling: 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            self._make_block(in_channels, hidden_channels),
            self._make_block(hidden_channels, hidden_channels),
            self._make_block(hidden_channels, hidden_channels // 2),
            self._make_block(hidden_channels // 2, hidden_channels // 4),
            self._make_block(hidden_channels // 4, hidden_channels // 4),
        )

        self.out_channels = hidden_channels // 4

    def _make_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        # Ensure exact 256x256 output
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x


class SemanticChangeHead(nn.Module):
    """
    Unified Semantic Change Head using Shared Semantic Space.

    For both datasets, outputs a single change map:
    - SECOND-CC (mode='scd'): 7-class semantic map
    - LEVIR-MCI (mode='bcd'): 3-class change type

    Uses cosine similarity with semantic prompts for classification.
    This creates alignment between visual features and semantic concepts.
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        prompt_dim: int = 512,
        num_semantic_classes: int = 7,  # SECOND-CC semantic classes
        num_change_classes: int = 3,    # LEVIR-MCI change classes
        temperature: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.prompt_dim = prompt_dim
        self.num_semantic_classes = num_semantic_classes
        self.num_change_classes = num_change_classes
        self.temperature = temperature

        # Shared decoder for feature upsampling
        self.decoder = SemanticDecoder(in_channels, hidden_channels)
        decoder_out_dim = self.decoder.out_channels

        # Project decoded features to prompt dimension
        self.feature_proj = nn.Sequential(
            nn.Conv2d(decoder_out_dim, prompt_dim, 1, bias=False),
            nn.BatchNorm2d(prompt_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(prompt_dim, prompt_dim, 1)
        )

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ~14.28

    def forward(
        self,
        features: torch.Tensor,
        semantic_prompts: torch.Tensor,
        mode: str = 'scd',
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: [B, C, H', W'] temporal difference features from TDT
            semantic_prompts: [K, D] semantic prompts from LSP
            mode: 'scd' for SECOND-CC (7 classes), 'bcd' for LEVIR-MCI (3 classes)
            return_embeddings: Whether to return pixel embeddings

        Returns:
            dict with:
                - cd_logits: [B, K, H, W] change detection logits
                - enhanced_features: [B, D, H, W] features for caption decoder
                - pixel_embeddings: [B, D, H, W] (if return_embeddings)
        """
        # Decode and upsample
        decoded = self.decoder(features)  # [B, C', 256, 256]

        # Project to prompt space
        pixel_features = self.feature_proj(decoded)  # [B, D, 256, 256]

        # Compute cosine similarity with semantic prompts
        cd_logits = self._compute_similarity(pixel_features, semantic_prompts)

        outputs = {
            'cd_logits': cd_logits,
            'enhanced_features': pixel_features,
        }

        if return_embeddings:
            outputs['pixel_embeddings'] = pixel_features

        return outputs

    def _compute_similarity(
        self,
        features: torch.Tensor,
        prompts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between pixel features and semantic prompts.

        Args:
            features: [B, D, H, W] pixel features in prompt space
            prompts: [K, D] semantic prompts

        Returns:
            [B, K, H, W] similarity logits (scaled)
        """
        B, D, H, W = features.shape
        K = prompts.shape[0]

        # Normalize features: [B, D, H, W]
        features_norm = F.normalize(features, dim=1)

        # Normalize prompts: [K, D]
        prompts_norm = F.normalize(prompts, dim=1)

        # Compute similarity using einsum: result is [B, K, H, W]
        # For each pixel, compute dot product with each prompt
        similarity = torch.einsum('bdhw,kd->bkhw', features_norm, prompts_norm)

        # Scale by learnable temperature
        similarity = similarity * self.logit_scale.exp()

        return similarity


class SemanticLoss(nn.Module):
    """
    Loss function for Semantic Change Head.

    Simple cross-entropy loss with optional label smoothing.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            logits: [B, K, H, W] prediction logits
            targets: [B, H, W] ground truth labels

        Returns:
            scalar loss
        """
        return self.ce_loss(logits, targets)
