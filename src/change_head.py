"""
Unified Change Detection Head
Handles both semantic (SECOND-CC) and binary (LEVIR-MCI) change detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Single decoder block with upsampling"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class SemanticAttention(nn.Module):
    """Generate spatial attention from change predictions"""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(num_classes, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, predictions: torch.Tensor):
        """
        Args:
            predictions: [B, K, H, W] change predictions
        Returns:
            attention: [B, 1, H, W] spatial attention map
        """
        return self.attention(predictions)


class UnifiedChangeHead(nn.Module):
    """
    Unified change detection head for both SECOND-CC and LEVIR-MCI

    - SECOND-CC: 7 semantic classes (after-change semantics: what the area became)
    - LEVIR-MCI: 3 classes (no_change, building, road)
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        scd_classes: int = 7,  # SECOND-CC semantic classes (after-change)
        bcd_classes: int = 3   # LEVIR-MCI multi-class (no_change, building, road)
    ):
        super().__init__()
        self.scd_classes = scd_classes
        self.bcd_classes = bcd_classes

        # Progressive upsampling decoder
        # From H/32 (8x8) to H (256x256) requires 5 stages of 2x upsampling each
        # 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            DecoderBlock(in_channels, hidden_channels),           # 8 -> 16
            DecoderBlock(hidden_channels, hidden_channels),       # 16 -> 32
            DecoderBlock(hidden_channels, hidden_channels // 2),  # 32 -> 64
            DecoderBlock(hidden_channels // 2, hidden_channels // 4),  # 64 -> 128
            DecoderBlock(hidden_channels // 4, hidden_channels // 4)   # 128 -> 256
        )

        # Classification heads for different modes
        # SCD: Semantic change detection (7 after-change classes for SECOND-CC)
        self.scd_classifier = nn.Conv2d(hidden_channels // 4, scd_classes, 1)

        # BCD: Multi-class change detection (3 classes for LEVIR-MCI)
        self.bcd_classifier = nn.Conv2d(hidden_channels // 4, bcd_classes, 1)

        # Attention modules for feature enhancement
        self.scd_attention = SemanticAttention(scd_classes)
        self.bcd_attention = SemanticAttention(bcd_classes)
        
        # Feature projection for caption decoder
        self.feature_proj = nn.Conv2d(in_channels, in_channels, 1)
    
    def forward(self, features: torch.Tensor, mode: str = 'scd', return_enhanced: bool = True):
        """
        Forward pass

        Args:
            features: [B, C, H', W'] temporal difference features from TDT
            mode: 'scd' for SECOND-CC, 'bcd' for LEVIR-MCI
            return_enhanced: Whether to return attention-enhanced features

        Returns:
            predictions: [B, K, H, W] change predictions (K=7 for scd, K=3 for bcd)
            enhanced_features: [B, C, H', W'] attention-enhanced features (optional)
        """
        # Decode and upsample features
        decoded = self.decoder(features)  # Progressive upsampling

        # Ensure output is exactly 256x256 by using interpolate as final step
        # This handles cases where input resolution varies
        decoded = F.interpolate(decoded, size=(256, 256), mode='bilinear', align_corners=False)

        # Mode-specific classification
        if mode == 'scd':
            predictions = self.scd_classifier(decoded)  # [B, 7, 256, 256] (after-change semantics)
            attention_map = self.scd_attention(predictions)
        else:  # mode == 'bcd'
            predictions = self.bcd_classifier(decoded)  # [B, 3, 256, 256] (no_change, building, road)
            attention_map = self.bcd_attention(predictions)
        
        # Generate enhanced features for caption decoder
        enhanced_features = None
        if return_enhanced:
            # Downsample attention to match feature resolution
            attn_downsampled = F.adaptive_avg_pool2d(attention_map, features.shape[-2:])
            # Apply attention to features
            projected = self.feature_proj(features)
            enhanced_features = projected * (1 + attn_downsampled)
        
        return predictions, enhanced_features