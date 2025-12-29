"""
UniSCC v5.0: Unified Semantic Change Captioning with Multi-Scale Architecture

Key architectural changes from v4.0:
    1. Multi-scale encoder with FPN (4 pyramid levels)
    2. Hierarchical alignment (per-scale with skip connections)
    3. Multi-scale TDT with cross-scale fusion
    4. Change-aware attention module
    5. Hierarchical semantic prompts (scale-specific)
    6. Multi-task CD head (classification + magnitude)
    7. Multi-level caption decoder (attends to all scales)

Expected improvements:
    - CD mIoU: +5-8%
    - Caption BLEU-4: +3-5%
    - Boundary quality: +15%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from dataclasses import dataclass, field

from .encoder import UniSCCEncoder
from .fpn import FeaturePyramidNetwork
from .hierarchical_alignment import HierarchicalAlignmentV5
from .multi_scale_tdt import MultiScaleTDT
from .change_aware_attention import ChangeAwareAttention
from .hierarchical_lsp import HierarchicalSemanticPrompts
from .multi_task_cd_head import MultiTaskCDHead
from .multi_level_caption_decoder import MultiLevelCaptionDecoder


@dataclass
class UniSCCConfigV5:
    """Configuration for UniSCC v5.0 model."""

    # Dataset
    dataset: str = "second_cc"  # "second_cc" or "levir_mci"

    # Encoder
    backbone: str = "swin_base_patch4_window7_224"
    pretrained: bool = True
    img_size: int = 256

    # FPN
    pyramid_channels: int = 256  # Unified pyramid channel dimension

    # Alignment
    alignment_heads: int = 8
    alignment_dropout: float = 0.1

    # TDT
    tdt_layers: int = 2
    tdt_heads: int = 8
    tdt_dropout: float = 0.1

    # Change Attention
    attention_reduction: int = 16

    # Semantic Classes
    num_semantic_classes: int = 7   # SECOND-CC: 7 semantic classes
    num_change_classes: int = 3     # LEVIR-MCI: 3 change classes

    # CD Head
    cd_hidden_channels: int = 256
    use_magnitude: bool = True

    # Caption Decoder
    vocab_size: int = 10000
    decoder_dim: int = 512
    decoder_heads: int = 8
    decoder_layers: int = 6
    decoder_ffn_dim: int = 2048
    decoder_dropout: float = 0.1
    max_caption_length: int = 50

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Legacy compatibility
    scd_classes: int = 7
    bcd_classes: int = 3


class UniSCCV5(nn.Module):
    """
    UniSCC v5.0: Multi-Scale Semantic Change Captioning Model.

    Architecture:
        1. Siamese Encoder (Swin-B) -> Multi-scale features
        2. FPN -> Unified 256-channel pyramid (P2, P3, P4, P5)
        3. Hierarchical Alignment -> Per-scale aligned features
        4. Multi-Scale TDT -> Fused change features (512 channels)
        5. Change-Aware Attention -> Enhanced features + attention map
        6. Hierarchical LSP -> Scale-specific semantic prompts
        7. Multi-Task CD Head -> Classification + magnitude
        8. Multi-Level Caption Decoder -> Change descriptions

    Outputs:
        - cd_logits: [B, K, 256, 256] semantic predictions
        - magnitude: [B, 1, 256, 256] change intensity
        - caption_logits: [B, T, V] or generated_captions: [B, T]
        - change_attention: [B, 1, 256, 256] spatial attention map
    """

    def __init__(self, config: Union[UniSCCConfigV5, dict]):
        super().__init__()

        # Convert dict to config if needed
        if isinstance(config, dict):
            self.config = UniSCCConfigV5(**{
                k: v for k, v in config.items()
                if k in UniSCCConfigV5.__dataclass_fields__
            })
        else:
            self.config = config

        self.dataset = self.config.dataset

        # Determine number of classes
        if self.dataset == 'second_cc':
            self.num_classes = self.config.num_semantic_classes
        else:
            self.num_classes = self.config.num_change_classes

        # 1. Siamese Encoder (with pyramid output)
        self.encoder = UniSCCEncoder(
            backbone=self.config.backbone,
            pretrained=self.config.pretrained,
            img_size=self.config.img_size,
            output_pyramid=True  # Enable multi-scale output
        )

        # Get backbone channel dimensions
        backbone_channels = self.encoder.backbone_channels

        # 2. Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(backbone_channels),
            out_channels=self.config.pyramid_channels
        )

        # 3. Hierarchical Alignment
        self.alignment = HierarchicalAlignmentV5(
            dim=self.config.pyramid_channels,
            num_heads=self.config.alignment_heads,
            num_scales=4,
            dropout=self.config.alignment_dropout
        )

        # 4. Multi-Scale TDT
        self.tdt = MultiScaleTDT(
            dim=self.config.pyramid_channels,
            num_heads=self.config.tdt_heads,
            num_layers=self.config.tdt_layers,
            output_dim=512,  # Fused output dimension
            target_size=256,
            dropout=self.config.tdt_dropout
        )

        # 5. Change-Aware Attention
        self.change_attention = ChangeAwareAttention(
            dim=512,
            reduction=self.config.attention_reduction
        )

        # 6. Hierarchical Semantic Prompts
        self.lsp = HierarchicalSemanticPrompts(
            dataset=self.dataset,
            prompt_dim=self.config.pyramid_channels,
            num_scales=4
        )

        # 7. Multi-Task CD Head
        self.change_head = MultiTaskCDHead(
            in_channels=512,
            hidden_channels=self.config.cd_hidden_channels,
            num_classes=self.num_classes,
            num_scales=4,
            target_size=256
        )

        # 8. Multi-Level Caption Decoder
        self.caption_decoder = MultiLevelCaptionDecoder(
            vocab_size=self.config.vocab_size,
            d_model=self.config.decoder_dim,
            nhead=self.config.decoder_heads,
            num_layers=self.config.decoder_layers,
            dim_feedforward=self.config.decoder_ffn_dim,
            dropout=self.config.decoder_dropout,
            max_length=self.config.max_caption_length,
            num_scales=4,
            prompt_dim=self.config.pyramid_channels,
            pad_token_id=self.config.pad_token_id,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id
        )

    def get_mode(self) -> str:
        """Get detection mode based on dataset."""
        return 'scd' if self.dataset == 'second_cc' else 'bcd'

    def forward(
        self,
        img_t0: torch.Tensor,
        img_t1: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        caption_lengths: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for UniSCC v5.0.

        Args:
            img_t0: [B, 3, H, W] pre-change image
            img_t1: [B, 3, H, W] post-change image
            captions: [B, T] caption tokens (for training)
            caption_lengths: [B] caption lengths
            return_features: Whether to return intermediate features

        Returns:
            dict with:
                - cd_logits: [B, K, 256, 256] semantic predictions
                - magnitude: [B, 1, 256, 256] change intensity
                - caption_logits: [B, T, V] (training)
                - generated_captions: [B, T] (inference)
                - change_attention: [B, 1, 256, 256] spatial attention
                - hierarchical_prompts: dict of scale-specific prompts
                - confidence_pyramid: dict of alignment confidences
        """
        # 1. Multi-scale encoding
        encoder_out = self.encoder(img_t0, img_t1)
        stages_t0 = encoder_out['stages_t0']
        stages_t1 = encoder_out['stages_t1']

        # 2. Build pyramids with FPN
        pyramid_t0 = self.fpn(stages_t0)
        pyramid_t1 = self.fpn(stages_t1)

        # 3. Hierarchical alignment
        aligned_pyramid_t0, confidence_pyramid = self.alignment(pyramid_t0, pyramid_t1)

        # 4. Multi-scale TDT
        tdt_out = self.tdt(aligned_pyramid_t0, pyramid_t1)
        fused_diff = tdt_out['fused_diff']
        diff_pyramid = tdt_out['diff_pyramid']

        # 5. Change-aware attention
        change_attn_out = self.change_attention(fused_diff)
        change_features = change_attn_out['change_features']
        change_attention_map = change_attn_out['change_attention']

        # 6. Get hierarchical prompts
        hierarchical_prompts = self.lsp()

        # 7. Multi-task CD
        cd_out = self.change_head(change_features, hierarchical_prompts)
        cd_logits = cd_out['cd_logits']
        magnitude = cd_out['magnitude']

        # 8. Caption generation
        teacher_forcing = captions is not None and self.training

        if teacher_forcing:
            caption_logits = self.caption_decoder(
                diff_pyramid,
                hierarchical_prompts,
                change_attention_map,
                captions,
                caption_lengths,
                teacher_forcing=True
            )
            generated_captions = None
        else:
            caption_logits = None
            generated_captions = self.caption_decoder.generate(
                diff_pyramid,
                hierarchical_prompts,
                change_attention_map
            )

        # Build output
        outputs = {
            'cd_logits': cd_logits,
            'magnitude': magnitude,
            'caption_logits': caption_logits,
            'generated_captions': generated_captions,
            'change_attention': change_attention_map,
            'hierarchical_prompts': hierarchical_prompts,
            'confidence_pyramid': confidence_pyramid,
        }

        # v4.0 compatibility aliases
        outputs['sem_B_logits'] = cd_logits
        outputs['sem_A_logits'] = cd_logits

        if return_features:
            outputs['features'] = {
                'pyramid_t0': pyramid_t0,
                'pyramid_t1': pyramid_t1,
                'aligned_pyramid_t0': aligned_pyramid_t0,
                'diff_pyramid': diff_pyramid,
                'fused_diff': fused_diff,
                'change_features': change_features,
            }

        return outputs

    def generate_caption(
        self,
        img_t0: torch.Tensor,
        img_t1: torch.Tensor,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Generate captions for image pairs."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(img_t0, img_t1)
            return outputs['generated_captions']

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing for memory optimization."""
        if hasattr(self.encoder, 'set_gradient_checkpointing'):
            self.encoder.set_gradient_checkpointing(enable)


def build_uniscc_v5(config: Union[UniSCCConfigV5, dict]) -> UniSCCV5:
    """Build UniSCC v5.0 model from configuration."""
    return UniSCCV5(config)
