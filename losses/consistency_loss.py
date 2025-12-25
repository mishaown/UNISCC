"""
Multi-Task Consistency Loss

Enforces consistency between change detection and caption generation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ConsistencyLoss(nn.Module):
    """
    Multi-Task Consistency Loss.
    
    Encourages alignment between:
    1. Visual change features and caption features
    2. Predicted change regions and mentioned change concepts
    
    Args:
        hidden_dim: Feature dimension
        temperature: Temperature for contrastive loss
        loss_type: 'cosine', 'mse', 'kl', or 'contrastive'
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        temperature: float = 0.07,
        loss_type: str = 'cosine'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.loss_type = loss_type
        
        # Projection heads for feature alignment
        self.visual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.caption_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        caption_features: torch.Tensor,
        change_logits: Optional[torch.Tensor] = None,
        caption_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            visual_features: [B, D] or [B, H*W, D] visual change features
            caption_features: [B, D] or [B, T, D] caption features
            change_logits: [B, C, H, W] change detection logits (optional)
            caption_logits: [B, T, V] caption logits (optional)
        
        Returns:
            Consistency loss
        """
        # Pool features if needed
        if visual_features.dim() == 3:
            visual_features = visual_features.mean(dim=1)
        if caption_features.dim() == 3:
            caption_features = caption_features.mean(dim=1)
        
        # Project features
        visual_proj = self.visual_proj(visual_features)
        caption_proj = self.caption_proj(caption_features)
        
        # Normalize
        visual_proj = F.normalize(visual_proj, dim=-1)
        caption_proj = F.normalize(caption_proj, dim=-1)
        
        if self.loss_type == 'cosine':
            # Cosine similarity loss (maximize similarity)
            similarity = (visual_proj * caption_proj).sum(dim=-1)
            loss = 1 - similarity.mean()
        
        elif self.loss_type == 'mse':
            # MSE loss
            loss = F.mse_loss(visual_proj, caption_proj)
        
        elif self.loss_type == 'contrastive':
            # InfoNCE contrastive loss
            B = visual_features.shape[0]
            
            # Compute similarity matrix
            logits = torch.matmul(visual_proj, caption_proj.t()) / self.temperature
            
            # Labels: diagonal elements are positives
            labels = torch.arange(B, device=logits.device)
            
            # Cross-entropy loss (both directions)
            loss_v2c = F.cross_entropy(logits, labels)
            loss_c2v = F.cross_entropy(logits.t(), labels)
            loss = (loss_v2c + loss_c2v) / 2
        
        elif self.loss_type == 'kl':
            # KL divergence (soft alignment)
            visual_dist = F.softmax(visual_proj, dim=-1)
            caption_dist = F.softmax(caption_proj, dim=-1)
            
            # Symmetric KL
            kl_vc = F.kl_div(
                visual_dist.log(), caption_dist, reduction='batchmean'
            )
            kl_cv = F.kl_div(
                caption_dist.log(), visual_dist, reduction='batchmean'
            )
            loss = (kl_vc + kl_cv) / 2
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class SemanticCaptionAlignmentLoss(nn.Module):
    """
    Semantic-Caption Alignment Loss.
    
    Encourages consistency between predicted change classes and
    semantic concepts mentioned in generated captions.
    
    Args:
        num_classes: Number of change classes
        class_names: List of class names for keyword matching
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        class_names: Optional[list] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Default class keywords for matching
        if class_names is None:
            self.class_keywords = {
                0: ['no', 'same', 'identical', 'unchanged'],
                1: ['vegetation', 'grass', 'green'],
                2: ['ground', 'soil', 'bare', 'land'],
                3: ['tree', 'trees', 'forest'],
                4: ['water', 'lake', 'river', 'pond'],
                5: ['building', 'buildings', 'house', 'structure'],
                6: ['playground', 'park', 'recreation'],
            }
        else:
            self.class_keywords = {i: [name] for i, name in enumerate(class_names)}
    
    def extract_mentioned_classes(
        self,
        captions: list,
        vocab
    ) -> torch.Tensor:
        """
        Extract which classes are mentioned in captions.
        
        Args:
            captions: List of caption strings
            vocab: Vocabulary object
        
        Returns:
            [B, num_classes] binary tensor
        """
        B = len(captions)
        mentioned = torch.zeros(B, self.num_classes)
        
        for i, caption in enumerate(captions):
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            
            words = caption.lower().split()
            
            for class_id, keywords in self.class_keywords.items():
                if any(kw in words for kw in keywords):
                    mentioned[i, class_id] = 1.0
        
        return mentioned
    
    def forward(
        self,
        change_logits: torch.Tensor,
        captions: list,
        vocab=None
    ) -> torch.Tensor:
        """
        Args:
            change_logits: [B, C, H, W] change detection logits
            captions: List of caption strings or list of lists
            vocab: Vocabulary object
        
        Returns:
            Alignment loss
        """
        device = change_logits.device
        
        # Get predicted class distribution
        # Average over spatial dimensions
        pred_probs = F.softmax(change_logits, dim=1)
        pred_class_probs = pred_probs.mean(dim=[2, 3])  # [B, C]
        
        # Get mentioned classes from captions
        mentioned = self.extract_mentioned_classes(captions, vocab)
        mentioned = mentioned.to(device)
        
        # Normalize mentioned to distribution
        mentioned_sum = mentioned.sum(dim=1, keepdim=True).clamp(min=1)
        mentioned_dist = mentioned / mentioned_sum
        
        # BCE loss between predicted and mentioned
        loss = F.binary_cross_entropy(
            pred_class_probs,
            mentioned_dist
        )
        
        return loss
