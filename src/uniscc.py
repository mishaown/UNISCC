"""
UniSCC: Unified Semantic Change Captioning

Main model with Shared Semantic Space architecture.
Both change detection and captioning use the same semantic prompts,
creating alignment between visual features, semantic classes, and language.

Unified output for both datasets:
- SECOND-CC: 7-class semantic map (what changed) + captions
- LEVIR-MCI: 3-class change type + captions
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from dataclasses import dataclass

from .encoder import UniSCCEncoder
from .tdt import TemporalDifferenceTransformer
from .lsp import LearnableSemanticPrompts
from .semantic_head import SemanticChangeHead
from .caption_decoder import SemanticCaptionDecoder


@dataclass
class UniSCCConfig:
    """Configuration for UniSCC model."""

    # Dataset
    dataset: str = "second_cc"  # "second_cc" or "levir_mci"

    # Encoder
    backbone: str = "swin_base_patch4_window7_224"
    pretrained: bool = True
    feature_dim: int = 512
    img_size: int = 256
    use_temporal_embed: bool = True

    # TDT
    tdt_layers: int = 3
    tdt_heads: int = 8
    tdt_dropout: float = 0.1

    # Semantic Classes
    num_semantic_classes: int = 7   # SECOND-CC: 7 semantic classes
    num_change_classes: int = 3     # LEVIR-MCI: 3 change classes

    # Change Head
    hidden_channels: int = 256
    temperature: float = 0.1

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

    # LSP
    prompt_dim: int = 512
    learnable_prompts: bool = True

    # Legacy compatibility
    scd_classes: int = 7
    bcd_classes: int = 3


class UniSCC(nn.Module):
    """
    Unified Semantic Change Captioning Model with Shared Semantic Space.

    Architecture:
    1. Encoder: Siamese Swin Transformer
    2. TDT: Temporal Difference Transformer
    3. LSP: Learnable Semantic Prompts (shared space)
    4. SemanticChangeHead: Similarity-based classification using prompts
    5. SemanticCaptionDecoder: Caption generation attending to prompts

    Output (same for both datasets):
        - cd_logits: [B, K, H, W] change detection logits
        - caption_logits/generated_captions: captions
        - semantic_prompts: the prompts used

    SECOND-CC: K=7 (semantic classes)
    LEVIR-MCI: K=3 (change types)
    """

    def __init__(self, config: Union[UniSCCConfig, dict]):
        super().__init__()

        # Convert dict to config if needed
        if isinstance(config, dict):
            self.config = UniSCCConfig(**{
                k: v for k, v in config.items()
                if k in UniSCCConfig.__dataclass_fields__
            })
        else:
            self.config = config

        self.dataset = self.config.dataset

        # Determine number of classes based on dataset
        if self.dataset == 'second_cc':
            self.num_classes = self.config.num_semantic_classes
        else:
            self.num_classes = self.config.num_change_classes

        # 1. Encoder
        self.encoder = UniSCCEncoder(
            backbone=self.config.backbone,
            pretrained=self.config.pretrained,
            feature_dim=self.config.feature_dim,
            img_size=self.config.img_size,
            use_temporal_embed=self.config.use_temporal_embed
        )

        # 2. TDT
        self.tdt = TemporalDifferenceTransformer(
            dim=self.config.feature_dim,
            num_heads=self.config.tdt_heads,
            num_layers=self.config.tdt_layers,
            dropout=self.config.tdt_dropout
        )

        # 3. LSP (Shared Semantic Space)
        self.lsp = LearnableSemanticPrompts(
            dataset=self.dataset,
            prompt_dim=self.config.prompt_dim,
            learnable=self.config.learnable_prompts
        )

        # 4. Semantic Change Head
        self.change_head = SemanticChangeHead(
            in_channels=self.config.feature_dim,
            hidden_channels=self.config.hidden_channels,
            prompt_dim=self.config.prompt_dim,
            num_semantic_classes=self.config.num_semantic_classes,
            num_change_classes=self.config.num_change_classes,
            temperature=self.config.temperature
        )

        # 5. Caption Decoder
        self.caption_decoder = SemanticCaptionDecoder(
            vocab_size=self.config.vocab_size,
            d_model=self.config.decoder_dim,
            nhead=self.config.decoder_heads,
            num_layers=self.config.decoder_layers,
            dim_feedforward=self.config.decoder_ffn_dim,
            dropout=self.config.decoder_dropout,
            max_length=self.config.max_caption_length,
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
        return_features: bool = False,
        force_teacher_forcing: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            img_t0: [B, 3, H, W] pre-change image
            img_t1: [B, 3, H, W] post-change image
            captions: [B, T] caption tokens (for training)
            caption_lengths: [B] caption lengths
            return_features: Whether to return intermediate features
            force_teacher_forcing: Force teacher forcing in eval mode

        Returns:
            dict with:
                - cd_logits: [B, K, H, W] change detection logits
                - caption_logits: [B, T, V] caption logits (training)
                - generated_captions: [B, T] generated tokens (inference)
                - semantic_prompts: [K, D] semantic prompts used
        """
        # 1. Encode both images
        encoder_output = self.encoder(img_t0, img_t1)
        feat_t0 = encoder_output['features_t0']
        feat_t1 = encoder_output['features_t1']

        # 2. Temporal Difference Transformer
        temporal_features = self.tdt(feat_t0, feat_t1)

        # 3. Get semantic prompts from LSP
        semantic_prompts = self.lsp.get_prompts()

        # 4. Semantic Change Detection
        mode = self.get_mode()
        cd_outputs = self.change_head(
            temporal_features,
            semantic_prompts,
            mode=mode,
            return_embeddings=True
        )

        # 5. Caption Generation
        enhanced_features = cd_outputs['enhanced_features']
        teacher_forcing = captions is not None and (self.training or force_teacher_forcing)

        if teacher_forcing:
            caption_logits = self.caption_decoder(
                enhanced_features, semantic_prompts,
                captions, caption_lengths,
                teacher_forcing=True
            )
            generated_captions = None
        else:
            caption_logits = None
            generated_captions = self.caption_decoder.generate(
                enhanced_features, semantic_prompts
            )

        # Build output - unified for both datasets
        outputs = {
            'cd_logits': cd_outputs['cd_logits'],
            'caption_logits': caption_logits,
            'generated_captions': generated_captions,
            'semantic_prompts': semantic_prompts,
        }

        if return_features:
            outputs['features'] = {
                'feat_t0': feat_t0,
                'feat_t1': feat_t1,
                'temporal_features': temporal_features,
                'enhanced_features': enhanced_features,
            }

        return outputs

    def generate_caption(
        self,
        img_t0: torch.Tensor,
        img_t1: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0
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


def build_uniscc(config: Union[UniSCCConfig, dict]) -> UniSCC:
    """Build UniSCC model from configuration."""
    return UniSCC(config)
