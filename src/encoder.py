"""
Vision Encoder for UniSCC
Implements Siamese encoder with temporal embeddings
"""

import torch
import torch.nn as nn

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
    """
    
    def __init__(
        self,
        backbone: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        feature_dim: int = 512,
        img_size: int = 256,
        use_temporal_embed: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_temporal_embed = use_temporal_embed
        
        # Build backbone
        if TIMM_AVAILABLE and "swin" in backbone.lower():
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                img_size=img_size,
                out_indices=[3]  # Only get last stage
            )
            # Get output channels from last stage
            backbone_channels = self.backbone.feature_info.channels()[-1]
            self.use_timm = True
        else:
            print(f"Using simple ResNet backbone (timm not available)")
            self.backbone = SimpleResNetBackbone(feature_dim)
            backbone_channels = feature_dim
            self.use_timm = False
        
        # Project to feature_dim if needed
        if backbone_channels != feature_dim:
            self.proj = nn.Conv2d(backbone_channels, feature_dim, 1)
        else:
            self.proj = nn.Identity()
        
        # Temporal embeddings
        if use_temporal_embed:
            self.temporal_embed = TemporalEmbedding(feature_dim)
        else:
            self.temporal_embed = None
    
    def forward_single(self, x: torch.Tensor, time_idx: int):
        """Extract features from single image"""
        # Get features from backbone
        if self.use_timm:
            # timm model - returns list of features
            features = self.backbone(x)
            x = features[-1]  # Use last stage (already filtered to only last stage)
            
            # Swin Transformer returns features in (B, H, W, C) format
            # Need to convert to (B, C, H, W) for Conv2d
            if x.dim() == 4 and x.shape[1] != self.backbone.feature_info.channels()[-1]:
                # Features are in (B, H, W, C) format - permute to (B, C, H, W)
                x = x.permute(0, 3, 1, 2)
        else:
            # Simple backbone
            x = self.backbone(x)
        
        # Project to target dimension
        x = self.proj(x)
        
        # Add temporal embedding
        if self.temporal_embed is not None:
            x = self.temporal_embed(x, time_idx)
        
        return x
    
    def forward(self, img_t0: torch.Tensor, img_t1: torch.Tensor):
        """
        Forward pass for bi-temporal images
        
        Args:
            img_t0: [B, 3, H, W] pre-change image
            img_t1: [B, 3, H, W] post-change image
        
        Returns:
            dict with features_t0 and features_t1 [B, C, H', W']
        """
        feat_t0 = self.forward_single(img_t0, time_idx=0)
        feat_t1 = self.forward_single(img_t1, time_idx=1)
        
        return {
            'features_t0': feat_t0,
            'features_t1': feat_t1
        }