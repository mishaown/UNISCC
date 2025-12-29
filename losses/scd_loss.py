"""
Semantic/Binary Change Detection Loss Functions

Supports:
- SECOND-CC: Multi-class semantic change detection (7 classes)
- LEVIR-MCI: Binary change detection (2 classes)

v5.0: Added magnitude loss and multi-task CD loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Class weights or scalar
        gamma: Focusing parameter (default: 2.0)
        ignore_index: Index to ignore
        reduction: 'none', 'mean', 'sum'
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        smooth: Smoothing factor
        ignore_index: Index to ignore
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        num_classes = inputs.shape[1]
        
        # Softmax probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Create mask for valid pixels
        valid_mask = targets != self.ignore_index
        targets_masked = targets.clone()
        targets_masked[~valid_mask] = 0
        
        # One-hot encode targets
        targets_onehot = F.one_hot(targets_masked, num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()
        
        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).expand_as(probs)
        probs = probs * valid_mask
        targets_onehot = targets_onehot * valid_mask
        
        # Compute Dice per class
        dims = (0, 2, 3)  # Batch, H, W
        intersection = (probs * targets_onehot).sum(dims)
        union = probs.sum(dims) + targets_onehot.sum(dims)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        if self.reduction == 'mean':
            return 1 - dice.mean()
        return 1 - dice.sum()


class BinaryChangeLoss(nn.Module):
    """
    Binary Change Detection Loss for LEVIR-MCI.
    
    Combines BCE and Dice loss for binary segmentation.
    
    Args:
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        pos_weight: Positive class weight for BCE
        ignore_index: Index to ignore
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: Optional[float] = None,
        ignore_index: int = 255
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        else:
            self.pos_weight = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, 1, H, W] or [B, 2, H, W] logits
            targets: [B, H, W] binary labels (0 or 1)
        """
        # Handle 2-class output
        if inputs.shape[1] == 2:
            # Use softmax + CE approach
            return F.cross_entropy(inputs, targets, ignore_index=self.ignore_index)
        
        # Single channel output - use BCE
        inputs = inputs.squeeze(1)
        
        # Create valid mask
        valid_mask = targets != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device)
        
        inputs_valid = inputs[valid_mask]
        targets_valid = targets[valid_mask].float()
        
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(
            inputs_valid, targets_valid,
            pos_weight=self.pos_weight
        )
        
        # Dice Loss
        probs = torch.sigmoid(inputs)
        probs_valid = probs[valid_mask]
        
        intersection = (probs_valid * targets_valid).sum()
        union = probs_valid.sum() + targets_valid.sum()
        dice = 1 - (2 * intersection + 1) / (union + 1)
        
        return self.bce_weight * bce + self.dice_weight * dice


class SCDLoss(nn.Module):
    """
    Semantic Change Detection Loss.
    
    Supports both multi-class (SECOND-CC) and binary (LEVIR-MCI) change detection.
    
    Args:
        num_classes: Number of classes (2 for binary, 7 for SECOND-CC)
        loss_type: 'ce', 'focal', 'dice', 'combined'
        focal_gamma: Gamma for focal loss
        dice_smooth: Smoothing for dice loss
        class_weights: Optional class weights
        ignore_index: Index to ignore
        ce_weight: Weight for CE in combined loss
        focal_weight: Weight for focal in combined loss
        dice_weight: Weight for dice in combined loss
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        loss_type: str = 'combined',
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        focal_weight: float = 1.0,
        dice_weight: float = 0.5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.ignore_index = ignore_index
        self.is_binary = (num_classes == 2)
        
        # Loss weights
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # Binary change loss
        if self.is_binary:
            self.binary_loss = BinaryChangeLoss(ignore_index=ignore_index)
        
        # Multi-class losses
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=ignore_index
        )
        
        self.focal_loss = FocalLoss(
            alpha=self.class_weights,
            gamma=focal_gamma,
            ignore_index=ignore_index
        )
        
        self.dice_loss = DiceLoss(
            smooth=dice_smooth,
            ignore_index=ignore_index
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        
        Returns:
            Total loss
        """
        # Use binary loss for 2-class or single-channel
        if self.is_binary or inputs.shape[1] <= 2:
            return self.binary_loss(inputs, targets)
        
        # Multi-class loss
        if self.loss_type == 'ce':
            return self.ce_loss(inputs, targets)
        
        elif self.loss_type == 'focal':
            return self.focal_loss(inputs, targets)
        
        elif self.loss_type == 'dice':
            return self.dice_loss(inputs, targets)
        
        elif self.loss_type == 'combined':
            ce = self.ce_loss(inputs, targets)
            focal = self.focal_loss(inputs, targets)
            dice = self.dice_loss(inputs, targets)

            total = (
                self.ce_weight * ce +
                self.focal_weight * focal +
                self.dice_weight * dice
            )
            return total

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class MagnitudeLoss(nn.Module):
    """
    Change Magnitude Loss for UniSCC v5.0.

    Predicts change intensity/magnitude at each pixel.
    Uses MSE loss between predicted magnitude and derived ground truth.

    Args:
        ignore_index: Index to ignore in target labels
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        ignore_index: int = 255,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        pred_magnitude: torch.Tensor,
        target_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute magnitude loss.

        Args:
            pred_magnitude: [B, 1, H, W] predicted change intensity (0-1)
            target_labels: [B, H, W] semantic labels

        Returns:
            MSE loss between predicted and derived magnitude
        """
        # Derive ground truth magnitude: 1 where changed (class > 0), 0 otherwise
        # Assuming class 0 is "no change" or "background"
        target_magnitude = (target_labels > 0).float()

        # Create valid mask
        valid_mask = (target_labels != self.ignore_index)

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_magnitude.device, requires_grad=True)

        # Squeeze channel dimension
        pred_magnitude = pred_magnitude.squeeze(1)  # [B, H, W]

        # Compute MSE on valid pixels
        diff_sq = (pred_magnitude - target_magnitude) ** 2

        if self.reduction == 'mean':
            return diff_sq[valid_mask].mean()
        elif self.reduction == 'sum':
            return diff_sq[valid_mask].sum()
        else:
            return diff_sq


class MultiTaskCDLoss(nn.Module):
    """
    Multi-Task Change Detection Loss for UniSCC v5.0.

    Combines:
        1. Classification loss (CE/Focal) for semantic segmentation
        2. Magnitude loss for change intensity prediction

    Args:
        num_classes: Number of semantic classes
        cls_weight: Weight for classification loss
        mag_weight: Weight for magnitude loss
        use_focal: Whether to use focal loss for classification
        focal_gamma: Gamma parameter for focal loss
        ignore_index: Index to ignore
        class_weights: Optional class weights for imbalanced data
    """

    def __init__(
        self,
        num_classes: int = 7,
        cls_weight: float = 1.0,
        mag_weight: float = 0.5,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.mag_weight = mag_weight
        self.use_focal = use_focal
        self.ignore_index = ignore_index

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Classification loss
        if use_focal:
            self.cls_loss = FocalLoss(
                alpha=self.class_weights,
                gamma=focal_gamma,
                ignore_index=ignore_index
            )
        else:
            self.cls_loss = nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=ignore_index
            )

        # Magnitude loss
        self.mag_loss = MagnitudeLoss(ignore_index=ignore_index)

    def forward(
        self,
        cd_logits: torch.Tensor,
        magnitude: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            cd_logits: [B, K, H, W] classification logits
            magnitude: [B, 1, H, W] predicted magnitude
            targets: [B, H, W] ground truth labels

        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary with individual loss values
        """
        # Classification loss
        cls = self.cls_loss(cd_logits, targets)

        # Magnitude loss
        mag = self.mag_loss(magnitude, targets)

        # Combined loss
        total = self.cls_weight * cls + self.mag_weight * mag

        loss_dict = {
            'cls': cls.item(),
            'mag': mag.item(),
            'total': total.item()
        }

        return total, loss_dict


class BoundaryLoss(nn.Module):
    """
    Boundary-Aware Loss for improved edge detection.

    Applies higher weight to pixels near boundaries.

    Args:
        kernel_size: Size of dilation kernel for boundary detection
        boundary_weight: Additional weight for boundary pixels
        ignore_index: Index to ignore
    """

    def __init__(
        self,
        kernel_size: int = 3,
        boundary_weight: float = 2.0,
        ignore_index: int = 255
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index

        # Create dilation kernel
        self.register_buffer(
            'dilation_kernel',
            torch.ones(1, 1, kernel_size, kernel_size)
        )

    def get_boundary_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Extract boundary pixels from labels.

        Args:
            labels: [B, H, W] label tensor

        Returns:
            [B, H, W] binary boundary mask
        """
        B, H, W = labels.shape

        # Convert to one-hot
        labels_valid = labels.clone()
        labels_valid[labels == self.ignore_index] = 0

        # Create boundary mask using morphological operations
        # A pixel is on boundary if dilated != original
        labels_float = labels_valid.float().unsqueeze(1)

        # Dilate
        dilated = F.conv2d(
            labels_float,
            self.dilation_kernel,
            padding=self.kernel_size // 2
        )

        # Erode (dilate with inverted)
        eroded = -F.conv2d(
            -labels_float,
            self.dilation_kernel,
            padding=self.kernel_size // 2
        )

        # Boundary is where dilated != eroded
        boundary = (dilated != eroded).float().squeeze(1)

        return boundary

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary-aware loss.

        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] ground truth labels

        Returns:
            Boundary-weighted cross entropy loss
        """
        # Get boundary mask
        boundary = self.get_boundary_mask(targets)

        # Compute pixel-wise CE loss
        ce_loss = F.cross_entropy(
            inputs, targets,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        # Apply boundary weighting
        weights = 1 + (self.boundary_weight - 1) * boundary
        weighted_loss = ce_loss * weights

        # Create valid mask
        valid_mask = targets != self.ignore_index

        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        return weighted_loss[valid_mask].mean()


class MultiTaskCDLossV5(nn.Module):
    """
    Extended Multi-Task CD Loss for UniSCC v5.0 with boundary awareness.

    Combines:
        1. Classification loss (Focal)
        2. Magnitude loss
        3. Boundary loss (optional)

    Args:
        num_classes: Number of semantic classes
        cls_weight: Weight for classification loss
        mag_weight: Weight for magnitude loss
        boundary_weight: Weight for boundary loss
        use_boundary: Whether to use boundary loss
        focal_gamma: Gamma for focal loss
        ignore_index: Index to ignore
    """

    def __init__(
        self,
        num_classes: int = 7,
        cls_weight: float = 1.0,
        mag_weight: float = 0.5,
        boundary_weight: float = 0.3,
        use_boundary: bool = False,
        focal_gamma: float = 2.0,
        ignore_index: int = 255
    ):
        super().__init__()

        self.cls_weight = cls_weight
        self.mag_weight = mag_weight
        self.boundary_weight = boundary_weight
        self.use_boundary = use_boundary

        # Losses
        self.focal_loss = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)
        self.mag_loss = MagnitudeLoss(ignore_index=ignore_index)

        if use_boundary:
            self.boundary_loss = BoundaryLoss(ignore_index=ignore_index)

    def forward(
        self,
        cd_logits: torch.Tensor,
        magnitude: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss with optional boundary awareness.
        """
        # Classification
        cls = self.focal_loss(cd_logits, targets)

        # Magnitude
        mag = self.mag_loss(magnitude, targets)

        # Total
        total = self.cls_weight * cls + self.mag_weight * mag

        loss_dict = {
            'cls': cls.item(),
            'mag': mag.item(),
        }

        # Optional boundary loss
        if self.use_boundary:
            boundary = self.boundary_loss(cd_logits, targets)
            total = total + self.boundary_weight * boundary
            loss_dict['boundary'] = boundary.item()

        loss_dict['total'] = total.item()

        return total, loss_dict
