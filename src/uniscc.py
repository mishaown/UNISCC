"""
UniSCC: Unified Semantic Change Captioning

Main model with Shared Semantic Space architecture.
Both change detection and captioning use the same semantic prompts,
creating alignment between visual features, semantic classes, and language.

Unified output for both datasets:
- SECOND-CC: 7-class semantic map (what changed) + captions
- LEVIR-MCI: 3-class change type + captions

v3.0: Dual Semantic Head with Transition Prompts
- Predicts both before (sem_A) and after (sem_B) semantic maps
- Uses transition embeddings for caption generation
- Enables "what changed into what" descriptions
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from dataclasses import dataclass, field

from .encoder import UniSCCEncoder
from .tdt import TemporalDifferenceTransformer
from .lsp import LearnableSemanticPrompts, TransitionLSP
from .semantic_head import SemanticChangeHead, DualSemanticHead
from .caption_decoder import SemanticCaptionDecoder, TransitionCaptionDecoder


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

    # v3.0: Dual Head Configuration
    dual_head: bool = True  # Enable dual semantic head
    share_decoder: bool = True  # Share decoder between A and B

    # v3.0: Transition Configuration
    transition_hidden_dim: int = 256
    use_transition_attention: bool = True

    # v3.0: Loss Configuration
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    sem_A_weight: float = 1.0
    sem_B_weight: float = 1.0
    caption_weight: float = 1.0

    # Legacy compatibility
    scd_classes: int = 7
    bcd_classes: int = 3


class UniSCC(nn.Module):
    """
    Unified Semantic Change Captioning Model with Shared Semantic Space.

    v3.0 Architecture:
    1. Encoder: Siamese Swin Transformer
    2. TDT: Temporal Difference Transformer (returns diff + enhanced features)
    3. TransitionLSP: Dual semantic prompts + transition embeddings
    4. DualSemanticHead: Predicts both sem_A and sem_B
    5. TransitionCaptionDecoder: Caption generation with transition attention

    Output:
        - sem_A_logits: [B, K, H, W] before-change semantic logits
        - sem_B_logits: [B, K, H, W] after-change semantic logits
        - cd_logits: [B, K, H, W] (alias for sem_B_logits, backward compat)
        - change_mask: [B, H, W] derived change mask
        - caption_logits/generated_captions: captions
        - prompts_A, prompts_B: semantic prompts used

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
        self.dual_head = self.config.dual_head

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

        # 2. TDT (v3.0: returns dict with diff + enhanced features)
        self.tdt = TemporalDifferenceTransformer(
            dim=self.config.feature_dim,
            num_heads=self.config.tdt_heads,
            num_layers=self.config.tdt_layers,
            dropout=self.config.tdt_dropout
        )

        # 3. LSP - v3.0: Use TransitionLSP for dual prompts + transitions
        if self.dual_head:
            self.lsp = TransitionLSP(
                dataset=self.dataset,
                prompt_dim=self.config.prompt_dim,
                learnable=self.config.learnable_prompts,
                transition_hidden_dim=self.config.transition_hidden_dim
            )
        else:
            # Fallback to legacy LSP
            self.lsp = LearnableSemanticPrompts(
                dataset=self.dataset,
                prompt_dim=self.config.prompt_dim,
                learnable=self.config.learnable_prompts
            )

        # 4. Change Head - v3.0: Use DualSemanticHead
        if self.dual_head:
            self.change_head = DualSemanticHead(
                in_channels=self.config.feature_dim,
                hidden_channels=self.config.hidden_channels,
                prompt_dim=self.config.prompt_dim,
                num_classes=self.num_classes,
                temperature=self.config.temperature,
                share_decoder=self.config.share_decoder
            )
        else:
            # Fallback to legacy head
            self.change_head = SemanticChangeHead(
                in_channels=self.config.feature_dim,
                hidden_channels=self.config.hidden_channels,
                prompt_dim=self.config.prompt_dim,
                num_semantic_classes=self.config.num_semantic_classes,
                num_change_classes=self.config.num_change_classes,
                temperature=self.config.temperature
            )

        # 5. Caption Decoder - v3.0: Use TransitionCaptionDecoder
        if self.dual_head and self.config.use_transition_attention:
            self.caption_decoder = TransitionCaptionDecoder(
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
        else:
            # Fallback to legacy decoder
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
        Forward pass for UniSCC v3.0.

        Args:
            img_t0: [B, 3, H, W] pre-change image
            img_t1: [B, 3, H, W] post-change image
            captions: [B, T] caption tokens (for training)
            caption_lengths: [B] caption lengths
            return_features: Whether to return intermediate features
            force_teacher_forcing: Force teacher forcing in eval mode

        Returns:
            dict with:
                - sem_A_logits: [B, K, H, W] before-change semantic logits (v3.0)
                - sem_B_logits: [B, K, H, W] after-change semantic logits (v3.0)
                - cd_logits: [B, K, H, W] change detection logits (backward compat)
                - change_mask: [B, H, W] derived change mask (v3.0)
                - caption_logits: [B, T, V] caption logits (training)
                - generated_captions: [B, T] generated tokens (inference)
                - prompts_A, prompts_B: semantic prompts used (v3.0)
        """
        # 1. Encode both images
        encoder_output = self.encoder(img_t0, img_t1)
        feat_t0 = encoder_output['features_t0']
        feat_t1 = encoder_output['features_t1']

        # 2. Temporal Difference Transformer
        # v3.0: TDT returns dict with diff + enhanced features
        tdt_output = self.tdt(feat_t0, feat_t1)

        # Handle both v3.0 dict output and legacy tensor output
        if isinstance(tdt_output, dict):
            diff_features = tdt_output['diff']
            feat_t0_enhanced = tdt_output['feat_t0_enhanced']
            feat_t1_enhanced = tdt_output['feat_t1_enhanced']
        else:
            # Legacy: TDT returns single tensor
            diff_features = tdt_output
            feat_t0_enhanced = feat_t0
            feat_t1_enhanced = feat_t1

        if self.dual_head:
            # v3.0: Dual head with transition prompts
            return self._forward_v3(
                feat_t0, feat_t1, diff_features,
                feat_t0_enhanced, feat_t1_enhanced,
                captions, caption_lengths,
                return_features, force_teacher_forcing
            )
        else:
            # Legacy: Single head forward
            return self._forward_legacy(
                diff_features, captions, caption_lengths,
                return_features, force_teacher_forcing,
                feat_t0, feat_t1
            )

    def _forward_v3(
        self,
        feat_t0: torch.Tensor,
        feat_t1: torch.Tensor,
        diff_features: torch.Tensor,
        feat_t0_enhanced: torch.Tensor,
        feat_t1_enhanced: torch.Tensor,
        captions: Optional[torch.Tensor],
        caption_lengths: Optional[torch.Tensor],
        return_features: bool,
        force_teacher_forcing: bool
    ) -> Dict[str, torch.Tensor]:
        """v3.0 forward with dual semantic head and transition prompts."""

        # 3. Get semantic prompts from TransitionLSP
        lsp_output = self.lsp()
        prompts_A = lsp_output['prompts_A']
        prompts_B = lsp_output['prompts_B']

        # 4. Dual Semantic Change Detection
        cd_outputs = self.change_head(
            feat_t0_enhanced, feat_t1_enhanced, diff_features,
            prompts_A, prompts_B,
            return_embeddings=True
        )

        # 5. Get transition embeddings for caption decoder
        pred_A = cd_outputs['pred_A']  # [B, H, W]
        pred_B = cd_outputs['pred_B']  # [B, H, W]
        transition_embeddings = self.lsp.get_transition_for_caption(pred_A, pred_B)

        # 6. Caption Generation
        enhanced_features = cd_outputs['enhanced_features']
        teacher_forcing = captions is not None and (self.training or force_teacher_forcing)

        if teacher_forcing:
            caption_logits = self.caption_decoder(
                enhanced_features, transition_embeddings,
                captions, caption_lengths,
                teacher_forcing=True
            )
            generated_captions = None
        else:
            caption_logits = None
            generated_captions = self.caption_decoder.generate(
                enhanced_features, transition_embeddings
            )

        # Build output
        outputs = {
            # v3.0 outputs
            'sem_A_logits': cd_outputs['sem_A_logits'],
            'sem_B_logits': cd_outputs['sem_B_logits'],
            'change_mask': cd_outputs['change_mask'],
            'prompts_A': prompts_A,
            'prompts_B': prompts_B,
            # Backward compatibility
            'cd_logits': cd_outputs['cd_logits'],  # Same as sem_B_logits
            'semantic_prompts': prompts_B,
            # Captions
            'caption_logits': caption_logits,
            'generated_captions': generated_captions,
        }

        if return_features:
            outputs['features'] = {
                'feat_t0': feat_t0,
                'feat_t1': feat_t1,
                'feat_t0_enhanced': feat_t0_enhanced,
                'feat_t1_enhanced': feat_t1_enhanced,
                'diff_features': diff_features,
                'enhanced_features': enhanced_features,
                'transition_embeddings': transition_embeddings,
            }

        return outputs

    def _forward_legacy(
        self,
        temporal_features: torch.Tensor,
        captions: Optional[torch.Tensor],
        caption_lengths: Optional[torch.Tensor],
        return_features: bool,
        force_teacher_forcing: bool,
        feat_t0: torch.Tensor,
        feat_t1: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Legacy forward for backward compatibility."""

        # Get semantic prompts from LSP
        semantic_prompts = self.lsp.get_prompts()

        # Semantic Change Detection
        mode = self.get_mode()
        cd_outputs = self.change_head(
            temporal_features,
            semantic_prompts,
            mode=mode,
            return_embeddings=True
        )

        # Caption Generation
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

        # Build output
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

    def set_gradient_checkpointing(self, enable: bool = True):
        """
        Enable or disable gradient checkpointing for memory optimization.

        This trades compute for memory by recomputing activations during backward pass.
        Can reduce memory usage by 30-50% at the cost of ~20% slower training.
        """
        # Enable for encoder
        if hasattr(self.encoder, 'set_gradient_checkpointing'):
            self.encoder.set_gradient_checkpointing(enable)

        # Enable for TDT if it supports it
        if hasattr(self.tdt, 'set_gradient_checkpointing'):
            self.tdt.set_gradient_checkpointing(enable)

        # Enable for caption decoder if it supports it
        if hasattr(self.caption_decoder, 'set_gradient_checkpointing'):
            self.caption_decoder.set_gradient_checkpointing(enable)


def build_uniscc(config: Union[UniSCCConfig, dict]) -> UniSCC:
    """Build UniSCC model from configuration."""
    return UniSCC(config)
