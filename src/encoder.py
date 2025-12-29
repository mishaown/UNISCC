"""
Vision Encoder for UniSCC
Implements Siamese encoder with temporal embeddings

v3.0: Added gradient checkpointing support for memory optimization
v5.0: Added multi-scale pyramid output for FPN architecture
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Union

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")


class TemporalEmbedding(nn.Module):
    """Learnable temporal position embeddings for bi-temporal images"""
    
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.temporal_embed_t0 = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        self.temporal_embed_t1 = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        nn.init.trunc_normal_(self.temporal_embed_t0, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed_t1, std=0.02)
    
    def forward(self, features: torch.Tensor, time_idx: int) -> torch.Tensor:
        """Add temporal embedding to features"""
        embed = self.temporal_embed_t0 if time_idx == 0 else self.temporal_embed_t1
        return features + embed


class SimpleResNetBackbone(nn.Module):
    """Fallback ResNet-like backbone when timm is not available"""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 128, 2, stride=2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, feature_dim, 2, stride=2)
        
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class UniSCCEncoder(nn.Module):
    """
    Siamese encoder for bi-temporal images
    Extracts features with shared weights and temporal embeddings

    v3.0: Added gradient checkpointing for memory optimization
    v5.0: Added multi-scale pyramid output for FPN architecture
    """

    def __init__(
        self,
        backbone: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        feature_dim: int = 512,
        img_size: int = 256,
        use_temporal_embed: bool = True,
        output_pyramid: bool = False  # v5.0: Enable multi-scale output
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_temporal_embed = use_temporal_embed
        self.output_pyramid = output_pyramid
        self.gradient_checkpointing = False  # Can be enabled for memory saving

        # Build backbone
        if TIMM_AVAILABLE and "swin" in backbone.lower():
            # v5.0: Get all stages for pyramid output
            out_indices = [0, 1, 2, 3] if output_pyramid else [3]
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                img_size=img_size,
                out_indices=out_indices
            )
            # Get output channels from all stages
            self.backbone_channels = self.backbone.feature_info.channels()
            backbone_channels = self.backbone_channels[-1]
            self.use_timm = True
        else:
            print(f"Using simple ResNet backbone (timm not available)")
            self.backbone = SimpleResNetBackbone(feature_dim)
            self.backbone_channels = [feature_dim]
            backbone_channels = feature_dim
            self.use_timm = False

        # Project to feature_dim if needed (only for single-scale mode)
        if not output_pyramid and backbone_channels != feature_dim:
            self.proj = nn.Conv2d(backbone_channels, feature_dim, 1)
        else:
            self.proj = nn.Identity()

        # Temporal embeddings (only for single-scale mode)
        if use_temporal_embed and not output_pyramid:
            self.temporal_embed = TemporalEmbedding(feature_dim)
        else:
            self.temporal_embed = None

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing = enable
        # Also enable for Swin backbone if available
        if self.use_timm and hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone (for checkpointing) - returns last stage."""
        if self.use_timm:
            features = self.backbone(x)
            x = features[-1]
            if x.dim() == 4 and x.shape[1] != self.backbone.feature_info.channels()[-1]:
                x = x.permute(0, 3, 1, 2)
        else:
            x = self.backbone(x)
        return x

    def _forward_backbone_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward through backbone and return ALL stages for pyramid mode."""
        if self.use_timm:
            features = self.backbone(x)
            # Process each stage to ensure correct format [B, C, H, W]
            processed = []
            for i, feat in enumerate(features):
                if feat.dim() == 4 and feat.shape[1] != self.backbone_channels[i]:
                    # Convert from [B, H, W, C] to [B, C, H, W]
                    feat = feat.permute(0, 3, 1, 2)
                processed.append(feat)
            return processed
        else:
            x = self.backbone(x)
            return [x]

    def forward_single(self, x: torch.Tensor, time_idx: int):
        """Extract features from single image"""
        # Get features from backbone with optional checkpointing
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self._forward_backbone, x, use_reentrant=False)
        else:
            x = self._forward_backbone(x)

        # Project to target dimension
        x = self.proj(x)

        # Add temporal embedding
        if self.temporal_embed is not None:
            x = self.temporal_embed(x, time_idx)

        return x

    def forward_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features for v5.0 pyramid mode.

        Args:
            x: [B, 3, H, W] input image

        Returns:
            List of features from each stage:
                - Stage 0: [B, 128, 64, 64]
                - Stage 1: [B, 256, 32, 32]
                - Stage 2: [B, 512, 16, 16]
                - Stage 3: [B, 1024, 8, 8]
        """
        if self.gradient_checkpointing and self.training:
            # Checkpoint doesn't work well with returning lists, so don't use it for pyramid
            features = self._forward_backbone_pyramid(x)
        else:
            features = self._forward_backbone_pyramid(x)

        return features

    def forward(self, img_t0: torch.Tensor, img_t1: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass for bi-temporal images

        Args:
            img_t0: [B, 3, H, W] pre-change image
            img_t1: [B, 3, H, W] post-change image

        Returns:
            v4.0 mode (output_pyramid=False):
                dict with 'features_t0' and 'features_t1' [B, C, H', W']

            v5.0 mode (output_pyramid=True):
                dict with 'stages_t0' and 'stages_t1' (lists of multi-scale features)
        """
        if self.output_pyramid:
            # v5.0: Return multi-scale features for FPN
            stages_t0 = self.forward_pyramid(img_t0)
            stages_t1 = self.forward_pyramid(img_t1)
            return {
                'stages_t0': stages_t0,
                'stages_t1': stages_t1
            }
        else:
            # v4.0: Return single-scale features
            feat_t0 = self.forward_single(img_t0, time_idx=0)
            feat_t1 = self.forward_single(img_t1, time_idx=1)
            return {
                'features_t0': feat_t0,
                'features_t1': feat_t1
            }