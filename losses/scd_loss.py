"""
Semantic/Binary Change Detection Loss Functions

Supports:
- SECOND-CC: Multi-class semantic change detection (7 classes)
- LEVIR-MCI: Binary change detection (2 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union


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
