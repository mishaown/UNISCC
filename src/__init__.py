"""
UniSCC Model Package - Shared Semantic Space Architecture

A unified framework for Semantic Change Captioning that jointly performs:
1. Change Detection (semantic or multi-class)
2. Change Captioning

v3.0: Dual Semantic Head with Transition Prompts
- Predicts both before (sem_A) and after (sem_B) semantic maps
- Uses transition embeddings for caption generation
- Enables "what changed into what" descriptions

Key Innovation: Shared Semantic Space
- Both change detection and captioning use the same semantic prompts
- Creates alignment between visual features, semantic classes, and language
- CLIP-initialized prompts provide strong semantic priors

For SECOND-CC:
    - Predicts semantic map for image A (7 classes)
    - Predicts semantic map for image B (7 classes)
    - Change is where classes differ

For LEVIR-MCI:
    - Predicts change type directly (3 classes)

Usage:
    from src import UniSCC, UniSCCConfig, build_uniscc

    # Create model (v3.0 dual head by default)
    config = UniSCCConfig(dataset='second_cc', vocab_size=10000, dual_head=True)
    model = UniSCC(config)

    # Forward pass
    outputs = model(img_t0, img_t1, captions, caption_lengths)
    sem_A = outputs['sem_A_logits']  # Before-change semantic map
    sem_B = outputs['sem_B_logits']  # After-change semantic map
    caption_logits = outputs['caption_logits']
"""

from .encoder import UniSCCEncoder
from .tdt import TemporalDifferenceTransformer
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

__version__ = '3.0.0'
__author__ = 'UniSCC Team'

__all__ = [
    # Main model
    'UniSCC',
    'UniSCCConfig',
    'build_uniscc',

    # v3.0 Components
    'TransitionLSP',
    'DualSemanticHead',
    'DualSemanticLoss',
    'TransitionCaptionDecoder',

    # Shared Components
    'UniSCCEncoder',
    'TemporalDifferenceTransformer',

    # Legacy Components (v2.0)
    'LearnableSemanticPrompts',
    'SemanticChangeHead',
    'SemanticCaptionDecoder',
    'SemanticLoss',

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
    # v3.0 config
    'dual_head': True,
    'share_decoder': True,
    'transition_hidden_dim': 256,
    'use_transition_attention': True,
    'use_focal_loss': True,
    'focal_gamma': 2.0,
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
    # v3.0 config
    'dual_head': True,
    'share_decoder': True,
    'transition_hidden_dim': 256,
    'use_transition_attention': True,
    'use_focal_loss': True,
    'focal_gamma': 2.0,
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
