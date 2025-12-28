"""
UniSCC: Unified Semantic Change Captioning

v4.0: Aligned Architecture with Single Semantic Head
- Bitemporal alignment module for feature registration
- Single semantic head predicting after-change semantics
- Semantic prompts guide both change detection and captioning
- Classes and captions work together through shared semantic space

Architecture:
    1. Siamese Encoder: Extract features from t0 and t1
    2. Feature Alignment: Align t0 features to t1 reference frame
    3. Temporal Difference: Compute change features from aligned pairs
    4. Semantic Prompts: Learnable class embeddings
    5. Change Detection Head: Predict after-change semantic map
    6. Caption Decoder: Generate change descriptions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from dataclasses import dataclass

from .encoder import UniSCCEncoder
from .tdt import TemporalDifferenceTransformer
from .alignment import FeatureAlignment
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

    # Alignment
    use_alignment: bool = True
    alignment_type: str = "cross_attention"  # cross_attention, deformable, hierarchical
    alignment_heads: int = 8

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
    dual_head: bool = False  # Disabled - use single head
    scd_classes: int = 7
    bcd_classes: int = 3


class UniSCC(nn.Module):
    """
    Unified Semantic Change Captioning Model.

    v4.0 Architecture:
    1. Siamese Encoder: Swin Transformer backbone
    2. Feature Alignment: Cross-attention based alignment for bitemporal registration
    3. TDT: Temporal Difference Transformer for change modeling
    4. LSP: Learnable Semantic Prompts for class embeddings
    5. Change Head: Single head predicting after-change semantics
    6. Caption Decoder: Transformer decoder with semantic guidance

    The model produces:
    - cd_logits: [B, K, H, W] after-change semantic predictions
    - caption_logits/generated_captions: change descriptions
    - alignment_info: alignment confidence and warped features

    K = num_semantic_classes for SECOND-CC (7)
    K = num_change_classes for LEVIR-MCI (3)
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

        # 1. Siamese Encoder
        self.encoder = UniSCCEncoder(
            backbone=self.config.backbone,
            pretrained=self.config.pretrained,
            feature_dim=self.config.feature_dim,
            img_size=self.config.img_size,
            use_temporal_embed=self.config.use_temporal_embed
        )

        # 2. Feature Alignment Module (NEW in v4.0)
        self.use_alignment = self.config.use_alignment
        if self.use_alignment:
            self.alignment = FeatureAlignment(
                dim=self.config.feature_dim,
                num_heads=self.config.alignment_heads,
                alignment_type=self.config.alignment_type,
                dropout=self.config.tdt_dropout
            )

        # 3. Temporal Difference Transformer
        self.tdt = TemporalDifferenceTransformer(
            dim=self.config.feature_dim,
            num_heads=self.config.tdt_heads,
            num_layers=self.config.tdt_layers,
            dropout=self.config.tdt_dropout
        )

        # 4. Learnable Semantic Prompts (single set for after-change)
        self.lsp = LearnableSemanticPrompts(
            dataset=self.dataset,
            prompt_dim=self.config.prompt_dim,
            learnable=self.config.learnable_prompts
        )

        # 5. Semantic Change Detection Head (single head)
        self.change_head = SemanticChangeHead(
            in_channels=self.config.feature_dim,
            hidden_channels=self.config.hidden_channels,
            prompt_dim=self.config.prompt_dim,
            num_semantic_classes=self.config.num_semantic_classes,
            num_change_classes=self.config.num_change_classes,
            temperature=self.config.temperature
        )

        # 6. Caption Decoder
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
        Forward pass for UniSCC v4.0.

        Args:
            img_t0: [B, 3, H, W] pre-change image
            img_t1: [B, 3, H, W] post-change image
            captions: [B, T] caption tokens (for training)
            caption_lengths: [B] caption lengths
            return_features: Whether to return intermediate features
            force_teacher_forcing: Force teacher forcing in eval mode

        Returns:
            dict with:
                - cd_logits: [B, K, H, W] after-change semantic logits
                - caption_logits: [B, T, V] caption logits (training)
                - generated_captions: [B, T] generated tokens (inference)
                - semantic_prompts: [K, D] semantic class embeddings
                - alignment_confidence: [B, 1, H, W] alignment quality map
        """
        # 1. Encode both images
        encoder_output = self.encoder(img_t0, img_t1)
        feat_t0 = encoder_output['features_t0']
        feat_t1 = encoder_output['features_t1']

        # 2. Align features (t0 to t1 reference frame)
        if self.use_alignment:
            align_output = self.alignment(feat_t0, feat_t1)
            feat_t0_aligned = align_output['feat_t0_aligned']
            alignment_confidence = align_output['confidence']
            diff_aligned = align_output['diff']
        else:
            feat_t0_aligned = feat_t0
            alignment_confidence = None
            diff_aligned = feat_t1 - feat_t0

        # 3. Temporal Difference Transformer
        tdt_output = self.tdt(feat_t0_aligned, feat_t1)

        # Handle dict or tensor output from TDT
        if isinstance(tdt_output, dict):
            diff_features = tdt_output['diff']
            feat_t1_enhanced = tdt_output.get('feat_t1_enhanced', feat_t1)
        else:
            diff_features = tdt_output
            feat_t1_enhanced = feat_t1

        # 4. Get semantic prompts
        semantic_prompts = self.lsp()

        # 5. Semantic Change Detection (after-change only)
        mode = self.get_mode()
        cd_outputs = self.change_head(
            diff_features,
            semantic_prompts,
            mode=mode,
            return_embeddings=True
        )

        cd_logits = cd_outputs['cd_logits']
        enhanced_features = cd_outputs['enhanced_features']

        # 6. Caption Generation
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

        # Build output
        outputs = {
            'cd_logits': cd_logits,
            'caption_logits': caption_logits,
            'generated_captions': generated_captions,
            'semantic_prompts': semantic_prompts,
            'alignment_confidence': alignment_confidence,
        }

        # v3.0 compatibility aliases
        outputs['sem_B_logits'] = cd_logits
        outputs['sem_A_logits'] = cd_logits  # Same as sem_B for single head

        if return_features:
            outputs['features'] = {
                'feat_t0': feat_t0,
                'feat_t1': feat_t1,
                'feat_t0_aligned': feat_t0_aligned,
                'diff_features': diff_features,
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

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing for memory optimization."""
        if hasattr(self.encoder, 'set_gradient_checkpointing'):
            self.encoder.set_gradient_checkpointing(enable)
        if hasattr(self.tdt, 'set_gradient_checkpointing'):
            self.tdt.set_gradient_checkpointing(enable)
        if hasattr(self.caption_decoder, 'set_gradient_checkpointing'):
            self.caption_decoder.set_gradient_checkpointing(enable)


def build_uniscc(config: Union[UniSCCConfig, dict]) -> UniSCC:
    """Build UniSCC model from configuration."""
    return UniSCC(config)
