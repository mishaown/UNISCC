"""
Unit tests for UniSCC v5.0 components.

Tests all new modules:
1. FPN
2. Hierarchical Alignment
3. Multi-Scale TDT
4. Change-Aware Attention
5. Hierarchical LSP
6. Multi-Task CD Head
7. Multi-Level Caption Decoder
8. Full v5.0 Model
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn

# Test settings
BATCH_SIZE = 2
IMG_SIZE = 256
VOCAB_SIZE = 1000
MAX_CAPTION_LEN = 50


def test_fpn():
    """Test Feature Pyramid Network."""
    print("Testing FPN...")

    from fpn import FeaturePyramidNetwork

    fpn = FeaturePyramidNetwork(
        in_channels_list=[128, 256, 512, 1024],
        out_channels=256
    )

    # Simulate Swin-B stage outputs
    C1 = torch.randn(BATCH_SIZE, 128, 64, 64)
    C2 = torch.randn(BATCH_SIZE, 256, 32, 32)
    C3 = torch.randn(BATCH_SIZE, 512, 16, 16)
    C4 = torch.randn(BATCH_SIZE, 1024, 8, 8)

    pyramid = fpn([C1, C2, C3, C4])

    assert pyramid['P2'].shape == (BATCH_SIZE, 256, 64, 64), f"P2 shape mismatch: {pyramid['P2'].shape}"
    assert pyramid['P3'].shape == (BATCH_SIZE, 256, 32, 32), f"P3 shape mismatch: {pyramid['P3'].shape}"
    assert pyramid['P4'].shape == (BATCH_SIZE, 256, 16, 16), f"P4 shape mismatch: {pyramid['P4'].shape}"
    assert pyramid['P5'].shape == (BATCH_SIZE, 256, 8, 8), f"P5 shape mismatch: {pyramid['P5'].shape}"

    print("  FPN: PASSED")
    return True


def test_hierarchical_alignment():
    """Test Hierarchical Alignment module."""
    print("Testing Hierarchical Alignment...")

    from hierarchical_alignment import HierarchicalAlignmentV5

    alignment = HierarchicalAlignmentV5(dim=256, num_heads=8, num_scales=4)

    pyramid_t0 = {
        'P2': torch.randn(BATCH_SIZE, 256, 64, 64),
        'P3': torch.randn(BATCH_SIZE, 256, 32, 32),
        'P4': torch.randn(BATCH_SIZE, 256, 16, 16),
        'P5': torch.randn(BATCH_SIZE, 256, 8, 8),
    }

    pyramid_t1 = {
        'P2': torch.randn(BATCH_SIZE, 256, 64, 64),
        'P3': torch.randn(BATCH_SIZE, 256, 32, 32),
        'P4': torch.randn(BATCH_SIZE, 256, 16, 16),
        'P5': torch.randn(BATCH_SIZE, 256, 8, 8),
    }

    aligned, confidence = alignment(pyramid_t0, pyramid_t1)

    assert aligned['P2'].shape == (BATCH_SIZE, 256, 64, 64), f"Aligned P2 shape: {aligned['P2'].shape}"
    assert aligned['P3'].shape == (BATCH_SIZE, 256, 32, 32), f"Aligned P3 shape: {aligned['P3'].shape}"
    assert confidence['P2'].shape == (BATCH_SIZE, 1, 64, 64), f"Confidence P2 shape: {confidence['P2'].shape}"
    assert confidence['P5'].shape == (BATCH_SIZE, 1, 8, 8), f"Confidence P5 shape: {confidence['P5'].shape}"

    print("  Hierarchical Alignment: PASSED")
    return True


def test_multi_scale_tdt():
    """Test Multi-Scale TDT."""
    print("Testing Multi-Scale TDT...")

    from multi_scale_tdt import MultiScaleTDT

    tdt = MultiScaleTDT(dim=256, num_heads=8, num_layers=2, output_dim=512, target_size=256)

    aligned_pyramid = {
        'P2': torch.randn(BATCH_SIZE, 256, 64, 64),
        'P3': torch.randn(BATCH_SIZE, 256, 32, 32),
        'P4': torch.randn(BATCH_SIZE, 256, 16, 16),
        'P5': torch.randn(BATCH_SIZE, 256, 8, 8),
    }

    pyramid_t1 = {
        'P2': torch.randn(BATCH_SIZE, 256, 64, 64),
        'P3': torch.randn(BATCH_SIZE, 256, 32, 32),
        'P4': torch.randn(BATCH_SIZE, 256, 16, 16),
        'P5': torch.randn(BATCH_SIZE, 256, 8, 8),
    }

    out = tdt(aligned_pyramid, pyramid_t1)

    assert out['fused_diff'].shape == (BATCH_SIZE, 512, 256, 256), f"Fused diff shape: {out['fused_diff'].shape}"
    assert out['diff_pyramid']['P2'].shape == (BATCH_SIZE, 256, 64, 64), f"Diff P2 shape: {out['diff_pyramid']['P2'].shape}"

    print("  Multi-Scale TDT: PASSED")
    return True


def test_change_aware_attention():
    """Test Change-Aware Attention."""
    print("Testing Change-Aware Attention...")

    from change_aware_attention import ChangeAwareAttention

    attn = ChangeAwareAttention(dim=512)
    fused_diff = torch.randn(BATCH_SIZE, 512, 256, 256)

    out = attn(fused_diff)

    assert out['change_features'].shape == (BATCH_SIZE, 512, 256, 256), f"Change features shape: {out['change_features'].shape}"
    assert out['change_attention'].shape == (BATCH_SIZE, 1, 256, 256), f"Change attention shape: {out['change_attention'].shape}"

    # Check attention values are in [0, 1]
    assert out['change_attention'].min() >= 0, "Attention min should be >= 0"
    assert out['change_attention'].max() <= 1, "Attention max should be <= 1"

    print("  Change-Aware Attention: PASSED")
    return True


def test_hierarchical_lsp():
    """Test Hierarchical Semantic Prompts."""
    print("Testing Hierarchical LSP...")

    from hierarchical_lsp import HierarchicalSemanticPrompts

    # Test SECOND-CC (7 classes)
    lsp = HierarchicalSemanticPrompts(
        dataset='second_cc',
        prompt_dim=256,
        num_scales=4
    )

    prompts = lsp()

    assert prompts['prompts_P2'].shape == (7, 256), f"Prompts P2 shape: {prompts['prompts_P2'].shape}"
    assert prompts['prompts_P3'].shape == (7, 256), f"Prompts P3 shape: {prompts['prompts_P3'].shape}"
    assert prompts['prompts_P4'].shape == (7, 256), f"Prompts P4 shape: {prompts['prompts_P4'].shape}"
    assert prompts['prompts_P5'].shape == (7, 256), f"Prompts P5 shape: {prompts['prompts_P5'].shape}"

    # Check prompts are normalized
    norms = prompts['prompts_P2'].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Prompts should be normalized"

    print("  Hierarchical LSP: PASSED")
    return True


def test_multi_task_cd_head():
    """Test Multi-Task CD Head."""
    print("Testing Multi-Task CD Head...")

    from multi_task_cd_head import MultiTaskCDHead

    head = MultiTaskCDHead(
        in_channels=512,
        hidden_channels=256,
        num_classes=7,
        num_scales=4
    )

    change_features = torch.randn(BATCH_SIZE, 512, 256, 256)

    hierarchical_prompts = {
        'prompts_P2': torch.randn(7, 256),
        'prompts_P3': torch.randn(7, 256),
        'prompts_P4': torch.randn(7, 256),
        'prompts_P5': torch.randn(7, 256),
    }

    out = head(change_features, hierarchical_prompts)

    assert out['cd_logits'].shape == (BATCH_SIZE, 7, 256, 256), f"CD logits shape: {out['cd_logits'].shape}"
    assert out['magnitude'].shape == (BATCH_SIZE, 1, 256, 256), f"Magnitude shape: {out['magnitude'].shape}"

    # Check magnitude is in [0, 1]
    assert out['magnitude'].min() >= 0, "Magnitude min should be >= 0"
    assert out['magnitude'].max() <= 1, "Magnitude max should be <= 1"

    print("  Multi-Task CD Head: PASSED")
    return True


def test_multi_level_caption_decoder():
    """Test Multi-Level Caption Decoder."""
    print("Testing Multi-Level Caption Decoder...")

    from multi_level_caption_decoder import MultiLevelCaptionDecoder

    decoder = MultiLevelCaptionDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        max_length=MAX_CAPTION_LEN,
        num_scales=4,
        prompt_dim=256
    )

    diff_pyramid = {
        'P2': torch.randn(BATCH_SIZE, 256, 64, 64),
        'P3': torch.randn(BATCH_SIZE, 256, 32, 32),
        'P4': torch.randn(BATCH_SIZE, 256, 16, 16),
        'P5': torch.randn(BATCH_SIZE, 256, 8, 8),
    }

    hierarchical_prompts = {
        'prompts_P2': torch.randn(7, 256),
        'prompts_P3': torch.randn(7, 256),
        'prompts_P4': torch.randn(7, 256),
        'prompts_P5': torch.randn(7, 256),
    }

    change_attention = torch.rand(BATCH_SIZE, 1, 256, 256)
    captions = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_CAPTION_LEN))
    lengths = torch.tensor([30, 25])

    # Test training forward
    decoder.train()
    logits = decoder(
        diff_pyramid, hierarchical_prompts, change_attention,
        captions, lengths, teacher_forcing=True
    )

    assert logits.shape == (BATCH_SIZE, MAX_CAPTION_LEN, VOCAB_SIZE), f"Logits shape: {logits.shape}"

    print("  Multi-Level Caption Decoder: PASSED")
    return True


def test_full_model():
    """Test full UniSCC v5.0 model."""
    print("Testing Full UniSCC v5.0 Model...")

    from src.uniscc_v5 import UniSCCV5, UniSCCConfigV5

    config = UniSCCConfigV5(
        dataset='second_cc',
        pretrained=False,  # Don't load pretrained for test
        vocab_size=VOCAB_SIZE,
        num_semantic_classes=7,
        decoder_layers=4,  # Reduced for faster test
        decoder_ffn_dim=1024
    )

    model = UniSCCV5(config)

    # Count parameters
    num_params = model.get_num_parameters()
    print(f"  Model parameters: {num_params:,}")

    # Test inputs
    img_t0 = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    img_t1 = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    captions = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_CAPTION_LEN))
    lengths = torch.tensor([30, 25])

    # Test training forward
    model.train()
    outputs = model(img_t0, img_t1, captions, lengths)

    assert 'cd_logits' in outputs, "Missing cd_logits"
    assert 'magnitude' in outputs, "Missing magnitude"
    assert 'caption_logits' in outputs, "Missing caption_logits"
    assert 'change_attention' in outputs, "Missing change_attention"

    assert outputs['cd_logits'].shape == (BATCH_SIZE, 7, IMG_SIZE, IMG_SIZE), f"CD logits: {outputs['cd_logits'].shape}"
    assert outputs['magnitude'].shape == (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE), f"Magnitude: {outputs['magnitude'].shape}"
    assert outputs['caption_logits'].shape == (BATCH_SIZE, MAX_CAPTION_LEN, VOCAB_SIZE), f"Caption logits: {outputs['caption_logits'].shape}"

    # Test inference forward
    model.eval()
    with torch.no_grad():
        outputs = model(img_t0, img_t1)

    assert 'generated_captions' in outputs, "Missing generated_captions"
    assert outputs['generated_captions'].shape[0] == BATCH_SIZE, "Generated captions batch size mismatch"

    print("  Full Model: PASSED")
    return True


def test_gradient_flow():
    """Test gradient flow through the model."""
    print("Testing Gradient Flow...")

    from src.uniscc_v5 import UniSCCV5, UniSCCConfigV5

    config = UniSCCConfigV5(
        dataset='second_cc',
        pretrained=False,
        vocab_size=VOCAB_SIZE,
        decoder_layers=2,
        decoder_ffn_dim=512
    )

    model = UniSCCV5(config)
    model.train()

    img_t0 = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, requires_grad=True)
    img_t1 = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, requires_grad=True)
    captions = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_CAPTION_LEN))
    lengths = torch.tensor([30, 25])
    targets = torch.randint(0, 7, (BATCH_SIZE, IMG_SIZE, IMG_SIZE))

    outputs = model(img_t0, img_t1, captions, lengths)

    # Compute losses
    cd_loss = nn.CrossEntropyLoss()(outputs['cd_logits'], targets)
    cap_loss = nn.CrossEntropyLoss(ignore_index=0)(
        outputs['caption_logits'].reshape(-1, VOCAB_SIZE),
        captions.reshape(-1)
    )
    total_loss = cd_loss + cap_loss

    # Backward
    total_loss.backward()

    # Check gradients exist
    grad_exists = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break

    assert grad_exists, "No gradients found in model parameters"

    print("  Gradient Flow: PASSED")
    return True


def test_losses():
    """Test v5.0 loss functions."""
    print("Testing v5.0 Losses...")

    sys.path.insert(0, 'losses')
    from scd_loss import MultiTaskCDLoss, MagnitudeLoss

    # Test magnitude loss
    mag_loss = MagnitudeLoss()
    pred_mag = torch.rand(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
    targets = torch.randint(0, 7, (BATCH_SIZE, IMG_SIZE, IMG_SIZE))

    loss = mag_loss(pred_mag, targets)
    assert loss.shape == (), f"Magnitude loss should be scalar, got {loss.shape}"
    assert loss >= 0, "Loss should be non-negative"

    # Test multi-task loss
    multi_loss = MultiTaskCDLoss(num_classes=7)
    cd_logits = torch.randn(BATCH_SIZE, 7, IMG_SIZE, IMG_SIZE)
    magnitude = torch.rand(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

    total_loss, loss_dict = multi_loss(cd_logits, magnitude, targets)

    assert total_loss.shape == (), "Total loss should be scalar"
    assert 'cls' in loss_dict, "Missing cls loss"
    assert 'mag' in loss_dict, "Missing mag loss"
    assert 'total' in loss_dict, "Missing total loss"

    print("  Losses: PASSED")
    return True


def run_all_tests():
    """Run all v5.0 tests."""
    print("=" * 60)
    print("UniSCC v5.0 Component Tests")
    print("=" * 60)

    tests = [
        ("FPN", test_fpn),
        ("Hierarchical Alignment", test_hierarchical_alignment),
        ("Multi-Scale TDT", test_multi_scale_tdt),
        ("Change-Aware Attention", test_change_aware_attention),
        ("Hierarchical LSP", test_hierarchical_lsp),
        ("Multi-Task CD Head", test_multi_task_cd_head),
        ("Multi-Level Caption Decoder", test_multi_level_caption_decoder),
        ("Losses", test_losses),
        ("Full Model", test_full_model),
        ("Gradient Flow", test_gradient_flow),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\nAll tests PASSED!")
    else:
        print(f"\n{failed} test(s) FAILED!")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
