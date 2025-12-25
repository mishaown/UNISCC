"""
Caption Generation Loss Functions

Cross-entropy based losses for sequence generation with label smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss.
    
    Smooths the target distribution to prevent overconfident predictions.
    
    Args:
        num_classes: Vocabulary size
        smoothing: Label smoothing factor (0.0 = no smoothing)
        ignore_index: Index to ignore (PAD token)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        ignore_index: int = 0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, T, V] or [B*T, V] logits
            targets: [B, T] or [B*T] token indices
        
        Returns:
            Smoothed cross-entropy loss
        """
        # Reshape if needed
        if inputs.dim() == 3:
            B, T, V = inputs.shape
            inputs = inputs.reshape(-1, V)
            targets = targets.reshape(-1)
        
        # Create smoothed distribution
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            
            # Set confidence for true class
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
            
            # Zero out ignore index
            mask = targets == self.ignore_index
            smooth_targets[mask] = 0
        
        # Compute loss
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # Apply mask
        loss = loss.masked_fill(mask, 0.0)
        
        if self.reduction == 'mean':
            num_valid = (~mask).sum()
            return loss.sum() / (num_valid + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CaptionLoss(nn.Module):
    """
    Caption Generation Loss.
    
    Supports standard cross-entropy and label smoothing.
    
    Args:
        vocab_size: Size of vocabulary
        label_smoothing: Label smoothing factor
        ignore_index: Index to ignore (PAD = 0)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        label_smoothing: float = 0.1,
        ignore_index: int = 0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        
        if label_smoothing > 0:
            self.loss_fn = LabelSmoothingLoss(
                num_classes=vocab_size,
                smoothing=label_smoothing,
                ignore_index=ignore_index,
                reduction=reduction
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                reduction=reduction
            )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, T, V] prediction logits
            targets: [B, T] target token indices
            lengths: [B] sequence lengths (optional)
        
        Returns:
            Caption loss
        """
        if self.label_smoothing > 0:
            return self.loss_fn(logits, targets)
        else:
            # Reshape for CrossEntropyLoss
            B, T, V = logits.shape
            logits = logits.reshape(-1, V)
            targets = targets.reshape(-1)
            return self.loss_fn(logits, targets)


class SequenceLoss(nn.Module):
    """
    Length-normalized sequence loss.
    
    Normalizes by sequence length to avoid bias toward shorter sequences.
    
    Args:
        vocab_size: Size of vocabulary
        label_smoothing: Label smoothing factor
        ignore_index: PAD token index
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        label_smoothing: float = 0.1,
        ignore_index: int = 0
    ):
        super().__init__()
        
        self.ignore_index = ignore_index
        
        if label_smoothing > 0:
            self.loss_fn = LabelSmoothingLoss(
                num_classes=vocab_size,
                smoothing=label_smoothing,
                ignore_index=ignore_index,
                reduction='none'
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                reduction='none'
            )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, T, V] prediction logits
            targets: [B, T] target token indices
            lengths: [B] sequence lengths
        
        Returns:
            Length-normalized loss
        """
        B, T, V = logits.shape
        
        # Compute per-token loss
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        
        if hasattr(self.loss_fn, 'num_classes'):
            loss = self.loss_fn(logits_flat, targets_flat)
        else:
            loss = self.loss_fn(logits_flat, targets_flat)
        
        loss = loss.reshape(B, T)
        
        # Sum per sequence and normalize by length
        seq_loss = loss.sum(dim=1)
        normalized_loss = seq_loss / lengths.float().clamp(min=1)
        
        return normalized_loss.mean()
