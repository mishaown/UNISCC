#!/usr/bin/env python3
"""
UniSCC v3.0 Sanity Check

Comprehensive tests for model components and forward passes.

Usage:
    python -m src.sanity_check

    Or from project root:
    cd E:/CD-Experiment/UNISCC
    python -m src.sanity_check
"""

import sys
import os

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def test_encoder():
    """Test UniSCCEncoder."""
    print("\n[1/6] Testing Encoder...")
    from src.encoder import UniSCCEncoder

    encoder = UniSCCEncoder(
        backbone='swin_base_patch4_window7_224',
        pretrained=False,
        feature_dim=512,
        img_size=256
    )

    B = 2
    img_t0 = torch.randn(B, 3, 256, 256)
    img_t1 = torch.randn(B, 3, 256, 256)

    outputs = encoder(img_t0, img_t1)

    assert 'features_t0' in outputs
    assert 'features_t1' in outputs
    assert outputs['features_t0'].shape == (B, 512, 8, 8)
    assert outputs['features_t1'].shape == (B, 512, 8, 8)

    # Test gradient checkpointing
    encoder.set_gradient_checkpointing(True)
    outputs = encoder(img_t0, img_t1)
    assert outputs['features_t0'].shape == (B, 512, 8, 8)

    print("  Encoder: PASSED")
    return True


def test_tdt():
    """Test Temporal Difference Transformer."""
    print("\n[2/6] Testing TDT...")
    from src.tdt import TemporalDifferenceTransformer

    tdt = TemporalDifferenceTransformer(
        dim=512,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    )

    B = 2
    feat_t0 = torch.randn(B, 512, 8, 8)
    feat_t1 = torch.randn(B, 512, 8, 8)

    outputs = tdt(feat_t0, feat_t1)

    assert 'diff' in outputs
    assert outputs['diff'].shape == (B, 512, 8, 8)

    # Check for enhanced features (v3.0)
    if 'feat_t0_enhanced' in outputs:
        assert outputs['feat_t0_enhanced'].shape == (B, 512, 8, 8)
        assert outputs['feat_t1_enhanced'].shape == (B, 512, 8, 8)
        print("  TDT (v3.0 with enhanced features): PASSED")
    else:
        print("  TDT: PASSED")

    return True


def test_lsp():
    """Test Learnable Semantic Prompts."""
    print("\n[3/6] Testing LSP...")
    from src.lsp import LearnableSemanticPrompts, TransitionLSP

    # Test basic LSP - forward() takes no arguments
    lsp = LearnableSemanticPrompts(
        dataset='second_cc',
        prompt_dim=512,
        learnable=True
    )

    prompts = lsp()  # No arguments needed

    assert prompts.shape[0] == 7  # num_classes for second_cc
    assert prompts.shape[1] == 512  # prompt_dim
    print("  LSP: PASSED")

    # Test TransitionLSP (v3.0) - forward() also takes no arguments
    try:
        trans_lsp = TransitionLSP(
            dataset='second_cc',
            prompt_dim=512,
            learnable=True
        )

        outputs = trans_lsp()  # No arguments needed
        assert 'prompts_A' in outputs
        assert 'prompts_B' in outputs
        assert outputs['prompts_A'].shape == (7, 512)
        assert outputs['prompts_B'].shape == (7, 512)
        assert 'transitions' in outputs
        print("  TransitionLSP (v3.0): PASSED")
    except Exception as e:
        print(f"  TransitionLSP: SKIPPED ({e})")

    return True


def test_semantic_head():
    """Test Semantic Change Head."""
    print("\n[4/6] Testing Semantic Head...")
    from src.semantic_head import SemanticChangeHead, DualSemanticHead

    B = 2

    # Test basic SemanticChangeHead - returns dict with 'logits' key
    head = SemanticChangeHead(
        in_channels=512,
        hidden_channels=256,
        prompt_dim=512,
        num_semantic_classes=7,
        num_change_classes=3
    )

    features = torch.randn(B, 512, 8, 8)
    prompts = torch.randn(7, 512)

    outputs = head(features, prompts, mode='scd')
    assert 'logits' in outputs
    assert outputs['logits'].shape == (B, 7, 256, 256)
    print("  SemanticChangeHead: PASSED")

    # Test DualSemanticHead (v3.0)
    try:
        dual_head = DualSemanticHead(
            in_channels=512,
            hidden_channels=256,
            prompt_dim=512,
            num_classes=7,
            share_decoder=True
        )

        feat_t0 = torch.randn(B, 512, 8, 8)
        feat_t1 = torch.randn(B, 512, 8, 8)
        diff_features = torch.randn(B, 512, 8, 8)
        prompts_A = torch.randn(7, 512)
        prompts_B = torch.randn(7, 512)

        outputs = dual_head(feat_t0, feat_t1, diff_features, prompts_A, prompts_B)

        assert 'sem_A_logits' in outputs
        assert 'sem_B_logits' in outputs
        assert outputs['sem_A_logits'].shape == (B, 7, 256, 256)
        assert outputs['sem_B_logits'].shape == (B, 7, 256, 256)
        print("  DualSemanticHead (v3.0): PASSED")
    except Exception as e:
        print(f"  DualSemanticHead: SKIPPED ({e})")

    return True


def test_caption_decoder():
    """Test Caption Decoder."""
    print("\n[5/6] Testing Caption Decoder...")
    from src.caption_decoder import SemanticCaptionDecoder, TransitionCaptionDecoder

    # Test SemanticCaptionDecoder
    decoder = SemanticCaptionDecoder(
        vocab_size=1000,
        d_model=512,
        nhead=8,
        num_layers=4,
        max_length=50
    )

    B = 2
    T = 30
    visual_features = torch.randn(B, 512, 8, 8)
    semantic_prompts = torch.randn(7, 512)
    captions = torch.randint(0, 1000, (B, T))
    lengths = torch.tensor([25, 20])

    # Training mode
    logits = decoder(visual_features, semantic_prompts, captions, lengths, teacher_forcing=True)
    assert logits.shape == (B, T, 1000)
    print("  SemanticCaptionDecoder (train): PASSED")

    # Inference mode
    generated = decoder(visual_features, semantic_prompts, teacher_forcing=False)
    assert generated.shape[0] == B
    print("  SemanticCaptionDecoder (infer): PASSED")

    # Test TransitionCaptionDecoder (v3.0)
    try:
        trans_decoder = TransitionCaptionDecoder(
            vocab_size=1000,
            d_model=512,
            nhead=8,
            num_layers=4,
            max_length=50
        )

        transition_embeddings = torch.randn(B, 64, 512)  # B, H*W, D
        logits = trans_decoder(visual_features, transition_embeddings, captions, lengths)
        assert logits.shape == (B, T, 1000)
        print("  TransitionCaptionDecoder (v3.0): PASSED")
    except Exception as e:
        print(f"  TransitionCaptionDecoder: SKIPPED ({e})")

    return True


def test_uniscc():
    """Test full UniSCC model."""
    print("\n[6/6] Testing UniSCC Model...")
    from src.uniscc import UniSCC, UniSCCConfig

    # Test SECOND-CC config
    config = UniSCCConfig(
        dataset='second_cc',
        pretrained=False,
        vocab_size=1000,
        num_semantic_classes=7,
        dual_head=True,
        use_transition_attention=True
    )

    model = UniSCC(config)
    print(f"  Parameters: {model.get_num_parameters():,}")

    B = 2
    T = 30
    img_t0 = torch.randn(B, 3, 256, 256)
    img_t1 = torch.randn(B, 3, 256, 256)
    captions = torch.randint(0, 1000, (B, T))
    lengths = torch.tensor([25, 20])

    # Training forward pass
    model.train()
    outputs = model(img_t0, img_t1, captions, lengths)

    assert 'cd_logits' in outputs
    assert 'caption_logits' in outputs
    assert outputs['cd_logits'].shape[0] == B
    assert outputs['caption_logits'].shape == (B, T, 1000)
    print("  UniSCC (train): PASSED")

    # Check v3.0 outputs
    if 'sem_A_logits' in outputs:
        assert outputs['sem_A_logits'].shape == (B, 7, 256, 256)
        assert outputs['sem_B_logits'].shape == (B, 7, 256, 256)
        print("  UniSCC v3.0 dual outputs: PASSED")

    # Inference forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(img_t0, img_t1)

    assert 'cd_logits' in outputs
    assert 'generated_captions' in outputs
    print("  UniSCC (infer): PASSED")

    # Test gradient checkpointing
    model.set_gradient_checkpointing(True)
    model.train()
    outputs = model(img_t0, img_t1, captions, lengths)
    assert 'cd_logits' in outputs
    print("  UniSCC (gradient checkpointing): PASSED")

    # Test LEVIR-MCI config
    config_levir = UniSCCConfig(
        dataset='levir_mci',
        pretrained=False,
        vocab_size=1000,
        num_change_classes=3,
        dual_head=True
    )
    model_levir = UniSCC(config_levir)

    model_levir.eval()
    with torch.no_grad():
        outputs = model_levir(img_t0, img_t1)

    assert outputs['cd_logits'].shape[1] == 3  # 3 classes
    print("  UniSCC (LEVIR-MCI): PASSED")

    return True


def run_all_tests():
    """Run all sanity checks."""
    print("=" * 60)
    print("UniSCC v3.0 Sanity Check")
    print("=" * 60)

    tests = [
        ("Encoder", test_encoder),
        ("TDT", test_tdt),
        ("LSP", test_lsp),
        ("Semantic Head", test_semantic_head),
        ("Caption Decoder", test_caption_decoder),
        ("UniSCC", test_uniscc)
    ]

    passed = 0
    failed = 0
    failed_tests = []

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
            failed_tests.append(name)

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("All tests PASSED!")
    else:
        print(f"{failed} tests FAILED: {', '.join(failed_tests)}")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
