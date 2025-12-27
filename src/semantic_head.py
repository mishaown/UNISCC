"""
Semantic Change Head with Shared Semantic Space

Unified change detection head that outputs a single change map:
- SECOND-CC: 7-class semantic segmentation (what the area became after change)
- LEVIR-MCI: 3-class change type (no_change, building, road)

Both use cosine similarity with learned semantic prompts for classification.

v3.0 Addition: DualSemanticHead
- Predicts both before-change (sem_A) and after-change (sem_B) semantic maps
- Uses separate prompts for A and B from TransitionLSP
- Outputs enhanced features for caption decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


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


class DualSemanticHead(nn.Module):
    """
    Dual Semantic Change Head for predicting both before and after semantics.

    v3.0 component that outputs:
        - sem_A_logits: [B, K, H, W] before-change semantic predictions
        - sem_B_logits: [B, K, H, W] after-change semantic predictions
        - change_mask: [B, H, W] binary mask where sem_A != sem_B (derived)
        - enhanced_features: [B, D, H, W] for caption decoder

    Uses separate prompts for before/after but optionally shared decoder architecture.

    Args:
        in_channels: Input feature channels from TDT
        hidden_channels: Hidden channels in decoder
        prompt_dim: Dimension of semantic prompts
        num_classes: Number of semantic classes
        temperature: Base temperature for similarity (overridden by learnable scale)
        share_decoder: Whether to share decoder weights between A and B
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        prompt_dim: int = 512,
        num_classes: int = 7,
        temperature: float = 0.1,
        share_decoder: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.prompt_dim = prompt_dim
        self.share_decoder = share_decoder

        # Decoder for before-change features (sem_A)
        self.decoder_A = SemanticDecoder(in_channels, hidden_channels)

        # Decoder for after-change features (sem_B)
        if share_decoder:
            self.decoder_B = self.decoder_A  # Share weights
        else:
            self.decoder_B = SemanticDecoder(in_channels, hidden_channels)

        decoder_out_dim = self.decoder_A.out_channels

        # Projection heads (separate for A and B)
        self.feature_proj_A = nn.Sequential(
            nn.Conv2d(decoder_out_dim, prompt_dim, 1, bias=False),
            nn.BatchNorm2d(prompt_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(prompt_dim, prompt_dim, 1)
        )

        self.feature_proj_B = nn.Sequential(
            nn.Conv2d(decoder_out_dim, prompt_dim, 1, bias=False),
            nn.BatchNorm2d(prompt_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(prompt_dim, prompt_dim, 1)
        )

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ~14.28

        # Change-aware feature fusion for caption decoder
        self.change_fusion = nn.Sequential(
            nn.Conv2d(prompt_dim * 2, prompt_dim, 1),
            nn.BatchNorm2d(prompt_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(prompt_dim, prompt_dim, 1)
        )

    def forward(
        self,
        feat_t0: torch.Tensor,
        feat_t1: torch.Tensor,
        diff_features: torch.Tensor,
        prompts_A: torch.Tensor,
        prompts_B: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for dual semantic prediction.

        Args:
            feat_t0: [B, C, H', W'] enhanced features for time 0 (from TDT)
            feat_t1: [B, C, H', W'] enhanced features for time 1 (from TDT)
            diff_features: [B, C, H', W'] difference features (from TDT)
            prompts_A: [K, D] before-change semantic prompts
            prompts_B: [K, D] after-change semantic prompts
            return_embeddings: Whether to return pixel embeddings

        Returns:
            dict with:
                - sem_A_logits: [B, K, H, W] before-change semantic logits
                - sem_B_logits: [B, K, H, W] after-change semantic logits
                - change_mask: [B, H, W] derived binary change mask
                - enhanced_features: [B, D, H, W] for caption decoder
                - pred_A: [B, H, W] argmax predictions for A
                - pred_B: [B, H, W] argmax predictions for B
        """
        # Decode features for time A (before)
        decoded_A = self.decoder_A(feat_t0)  # [B, C', 256, 256]
        pixel_features_A = self.feature_proj_A(decoded_A)  # [B, D, 256, 256]

        # Decode features for time B (after)
        decoded_B = self.decoder_B(feat_t1)  # [B, C', 256, 256]
        pixel_features_B = self.feature_proj_B(decoded_B)  # [B, D, 256, 256]

        # Compute similarities with respective prompts
        sem_A_logits = self._compute_similarity(pixel_features_A, prompts_A)
        sem_B_logits = self._compute_similarity(pixel_features_B, prompts_B)

        # Derive change mask from predictions
        pred_A = sem_A_logits.argmax(dim=1)  # [B, H, W]
        pred_B = sem_B_logits.argmax(dim=1)  # [B, H, W]
        change_mask = (pred_A != pred_B).float()  # [B, H, W]

        # Create enhanced features for caption decoder
        # Fuse before/after features
        fused_features = self.change_fusion(
            torch.cat([pixel_features_A, pixel_features_B], dim=1)
        )

        outputs = {
            'sem_A_logits': sem_A_logits,
            'sem_B_logits': sem_B_logits,
            'change_mask': change_mask,
            'enhanced_features': fused_features,
            'pred_A': pred_A,
            'pred_B': pred_B,
            # Backward compatibility
            'cd_logits': sem_B_logits,
        }

        if return_embeddings:
            outputs['pixel_embeddings_A'] = pixel_features_A
            outputs['pixel_embeddings_B'] = pixel_features_B

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
        similarity = torch.einsum('bdhw,kd->bkhw', features_norm, prompts_norm)

        # Scale by learnable temperature
        similarity = similarity * self.logit_scale.exp()

        return similarity


class DualSemanticLoss(nn.Module):
    """
    Loss function for Dual Semantic Head.

    Computes:
    1. CE/Focal loss for sem_A predictions
    2. CE/Focal loss for sem_B predictions
    3. Optional: Consistency loss between change mask and ground truth

    Args:
        num_classes: Number of semantic classes
        ignore_index: Index to ignore in loss computation
        label_smoothing: Label smoothing factor
        class_weights: Optional class weights for imbalanced data
        use_focal: Whether to use focal loss
        focal_gamma: Gamma parameter for focal loss
    """

    def __init__(
        self,
        num_classes: int = 7,
        ignore_index: int = 255,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        use_focal: bool = True,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Standard CE loss (used when focal is disabled)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        sem_A_logits: torch.Tensor,
        sem_B_logits: torch.Tensor,
        target_A: torch.Tensor,
        target_B: torch.Tensor,
        change_mask_pred: Optional[torch.Tensor] = None,
        change_mask_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dual semantic loss.

        Args:
            sem_A_logits: [B, K, H, W] predictions for before
            sem_B_logits: [B, K, H, W] predictions for after
            target_A: [B, H, W] ground truth for before
            target_B: [B, H, W] ground truth for after
            change_mask_pred: [B, H, W] predicted change mask (optional)
            change_mask_gt: [B, H, W] ground truth change mask (optional)

        Returns:
            dict with 'total', 'sem_A', 'sem_B', and optionally 'change' losses
        """
        if self.use_focal:
            loss_A = self._focal_loss(sem_A_logits, target_A)
            loss_B = self._focal_loss(sem_B_logits, target_B)
        else:
            loss_A = self.ce_loss(sem_A_logits, target_A)
            loss_B = self.ce_loss(sem_B_logits, target_B)

        total = loss_A + loss_B

        losses = {
            'total': total,
            'sem_A': loss_A,
            'sem_B': loss_B,
        }

        # Optional change mask consistency loss
        if change_mask_pred is not None and change_mask_gt is not None:
            change_loss = F.binary_cross_entropy(
                change_mask_pred.clamp(0, 1),
                change_mask_gt.float().clamp(0, 1)
            )
            losses['change'] = change_loss
            losses['total'] = total + 0.5 * change_loss

        return losses

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss for handling class imbalance.

        Focal loss = -alpha * (1-pt)^gamma * log(pt)

        Args:
            logits: [B, K, H, W] prediction logits
            targets: [B, H, W] ground truth labels

        Returns:
            scalar focal loss
        """
        # Compute cross entropy loss per pixel (no reduction)
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        # Get probabilities
        probs = F.softmax(logits, dim=1)  # [B, K, H, W]

        # Clamp targets for gather (handle ignore_index)
        targets_clamped = targets.clone()
        targets_clamped[targets == self.ignore_index] = 0

        # Get probability of true class
        # targets_clamped: [B, H, W] -> [B, 1, H, W]
        pt = probs.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)  # [B, H, W]

        # Compute focal weight
        focal_weight = (1 - pt) ** self.focal_gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Mask ignored pixels
        valid_mask = (targets != self.ignore_index)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return focal_loss[valid_mask].mean()
