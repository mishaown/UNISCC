"""
UniSCC Model Package - Aligned Semantic Change Captioning

A unified framework for Semantic Change Captioning that jointly performs:
1. Change Detection (semantic or multi-class)
2. Change Captioning

v4.0: Aligned Architecture with Single Semantic Head
- Bitemporal alignment module for feature registration
- Single semantic head predicting after-change semantics
- Semantic prompts guide both change detection and captioning
- Classes and captions work together through shared semantic space

Key Innovations:
1. Feature Alignment: Cross-attention based alignment handles spatial
   misalignments in satellite imagery
2. Shared Semantic Space: Both change detection and captioning use the
   same semantic prompts
3. CLIP-initialized prompts provide strong semantic priors

For SECOND-CC:
    - Predicts after-change semantic map (7 classes)
    - Uses aligned features for accurate change detection

For LEVIR-MCI:
    - Predicts change type directly (3 classes)

Usage:
    from src import UniSCC, UniSCCConfig, build_uniscc

    # Create model with alignment
    config = UniSCCConfig(
        dataset='second_cc',
        vocab_size=10000,
        use_alignment=True,
        alignment_type='cross_attention'
    )
    model = UniSCC(config)

    # Forward pass
    outputs = model(img_t0, img_t1, captions, caption_lengths)
    cd_logits = outputs['cd_logits']  # After-change semantic map
    caption_logits = outputs['caption_logits']
    alignment_conf = outputs['alignment_confidence']  # Alignment quality
"""

from .encoder import UniSCCEncoder
from .tdt import TemporalDifferenceTransformer
from .alignment import (
    FeatureAlignment,
    CrossAttentionAlignment,
    DeformableAlignment,
    HierarchicalAlignment
)
from .lsp import (
    LearnableSemanticPrompts,
    TransitionLSP,
    SECOND_CC_CLASSES,
    LEVIR_MCI_CLASSES
)
from .semantic_head import (
    SemanticChangeHead,
    SemanticLoss,
    DualSemanticHead,
    DualSemanticLoss
)
from .caption_decoder import (
    SemanticCaptionDecoder,
    TransitionCaptionDecoder
)
from .uniscc import UniSCC, UniSCCConfig, build_uniscc

# Backward compatibility
from .caption_decoder import SemanticCaptionDecoder as ChangeGuidedCaptionDecoder

__version__ = '4.0.0'
__author__ = 'UniSCC Team'

__all__ = [
    # Main model
    'UniSCC',
    'UniSCCConfig',
    'build_uniscc',

    # v4.0 Components - Alignment
    'FeatureAlignment',
    'CrossAttentionAlignment',
    'DeformableAlignment',
    'HierarchicalAlignment',

    # Shared Components
    'UniSCCEncoder',
    'TemporalDifferenceTransformer',
    'LearnableSemanticPrompts',
    'SemanticChangeHead',
    'SemanticCaptionDecoder',
    'SemanticLoss',

    # Legacy v3.0 Components
    'TransitionLSP',
    'DualSemanticHead',
    'DualSemanticLoss',
    'TransitionCaptionDecoder',

    # Backward compatibility
    'ChangeGuidedCaptionDecoder',

    # Constants
    'SECOND_CC_CLASSES',
    'LEVIR_MCI_CLASSES',
]


# Quick access configurations
DEFAULT_SECOND_CC_CONFIG = {
    'dataset': 'second_cc',
    'backbone': 'swin_base_patch4_window7_224',
    'pretrained': True,
    'feature_dim': 512,
    'img_size': 256,
    'tdt_layers': 3,
    'tdt_heads': 8,
    'num_semantic_classes': 7,
    'vocab_size': 10000,
    'decoder_dim': 512,
    'decoder_layers': 6,
    'max_caption_length': 50,
    # v4.0 config - alignment
    'use_alignment': True,
    'alignment_type': 'cross_attention',
    'alignment_heads': 8,
}

DEFAULT_LEVIR_MCI_CONFIG = {
    'dataset': 'levir_mci',
    'backbone': 'swin_base_patch4_window7_224',
    'pretrained': True,
    'feature_dim': 512,
    'img_size': 256,
    'tdt_layers': 3,
    'tdt_heads': 8,
    'num_change_classes': 3,
    'vocab_size': 10000,
    'decoder_dim': 512,
    'decoder_layers': 6,
    'max_caption_length': 50,
    # v4.0 config - alignment
    'use_alignment': True,
    'alignment_type': 'cross_attention',
    'alignment_heads': 8,
}


def create_second_cc_model(**kwargs):
    """Create model for SECOND-CC dataset with default configuration."""
    config = DEFAULT_SECOND_CC_CONFIG.copy()
    config.update(kwargs)
    return build_uniscc(config)


def create_levir_mci_model(**kwargs):
    """Create model for LEVIR-MCI dataset with default configuration."""
    config = DEFAULT_LEVIR_MCI_CONFIG.copy()
    config.update(kwargs)
    return build_uniscc(config)


def get_model_info(model: UniSCC) -> dict:
    """Get model information and statistics."""
    return {
        'dataset': model.dataset,
        'mode': model.get_mode(),
        'num_parameters': model.get_num_parameters(trainable_only=False),
        'trainable_parameters': model.get_num_parameters(trainable_only=True),
        'components': {
            'encoder': type(model.encoder).__name__,
            'tdt': type(model.tdt).__name__,
            'lsp': type(model.lsp).__name__,
            'change_head': type(model.change_head).__name__,
            'caption_decoder': type(model.caption_decoder).__name__,
        }
    }
