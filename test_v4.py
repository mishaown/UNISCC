"""Test UniSCC v4.0 with alignment module."""
import sys
sys.path.insert(0, 'src')

import torch
from src.uniscc import UniSCC, UniSCCConfig

def main():
    print("=" * 60)
    print("Testing UniSCC v4.0 with Feature Alignment")
    print("=" * 60)

    # Test SECOND-CC config
    print("\n1. Testing SECOND-CC model...")
    config = UniSCCConfig(
        dataset='second_cc',
        pretrained=False,
        vocab_size=1000,
        num_semantic_classes=7,
        use_alignment=True,
        alignment_type='cross_attention'
    )
    model = UniSCC(config)
    print(f"   Parameters: {model.get_num_parameters():,}")
    print(f"   Alignment: {config.alignment_type}")

    # Quick forward pass
    B = 2
    img_t0 = torch.randn(B, 3, 256, 256)
    img_t1 = torch.randn(B, 3, 256, 256)
    captions = torch.randint(0, 1000, (B, 50))
    lengths = torch.tensor([30, 25])

    model.train()
    outputs = model(img_t0, img_t1, captions, lengths)

    print(f"   cd_logits shape: {outputs['cd_logits'].shape}")
    print(f"   caption_logits shape: {outputs['caption_logits'].shape}")
    print(f"   alignment_confidence: {outputs['alignment_confidence'].shape if outputs['alignment_confidence'] is not None else 'None'}")

    # Verify outputs
    assert outputs['cd_logits'].shape == (B, 7, 256, 256), "CD logits shape mismatch!"
    print("   SECOND-CC: PASSED")

    # Test LEVIR-MCI config
    print("\n2. Testing LEVIR-MCI model...")
    config2 = UniSCCConfig(
        dataset='levir_mci',
        pretrained=False,
        vocab_size=1000,
        num_change_classes=3,
        use_alignment=True,
        alignment_type='cross_attention'
    )
    model2 = UniSCC(config2)

    model2.train()
    outputs2 = model2(img_t0, img_t1, captions, lengths)

    print(f"   cd_logits shape: {outputs2['cd_logits'].shape}")
    assert outputs2['cd_logits'].shape == (B, 3, 256, 256), "CD logits shape mismatch!"
    print("   LEVIR-MCI: PASSED")

    # Test without alignment
    print("\n3. Testing without alignment...")
    config3 = UniSCCConfig(
        dataset='second_cc',
        pretrained=False,
        vocab_size=1000,
        use_alignment=False
    )
    model3 = UniSCC(config3)
    model3.train()
    outputs3 = model3(img_t0, img_t1, captions, lengths)

    print(f"   cd_logits shape: {outputs3['cd_logits'].shape}")
    print(f"   alignment_confidence: {outputs3['alignment_confidence']}")
    assert outputs3['alignment_confidence'] is None, "Should be None when alignment disabled!"
    print("   No-alignment mode: PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

if __name__ == '__main__':
    main()
