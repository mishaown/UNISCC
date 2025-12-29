"""
Data and Model Verification Script

Tests:
1. Data loading and batch shapes
2. Model input/output dimensions
3. Parameter counts
4. Memory estimation
"""

import sys
sys.path.insert(0, 'src')

import torch
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_data_loading(config_path: str = None):
    """Test data loading and verify batch shapes."""
    print("=" * 60)
    print("Testing Data Loading")
    print("=" * 60)

    from data import create_dataloaders

    if config_path and Path(config_path).exists():
        config = load_config(config_path)
        print(f"Using config: {config_path}")
    else:
        # Default test config
        config = {
            'dataset': {
                'name': 'SECOND-CC',
                'root': 'E:/CD-Experiment/Datasets/SECOND-CC-AUG',
                'train_split': 'train',
                'val_split': 'val',
                'test_split': 'test',
                'image_size': 256,
                'num_classes': 7,
                'max_caption_length': 50,
                'min_word_freq': 5,
                'vocab_size': 10000,
                'num_captions_per_image': 5,
            },
            'training': {
                'batch_size': 2,
                'num_workers': 0,
            },
            'augmentation': {
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            }
        }
        print("Using default test config")

    try:
        train_loader, val_loader, test_loader, vocab = create_dataloaders(config)

        print(f"\nDataset: {config['dataset']['name']}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset) if test_loader else 'N/A'}")
        print(f"Vocabulary size: {len(vocab) if vocab else 'N/A'}")

        # Get a batch and verify shapes
        batch = next(iter(train_loader))

        print("\nBatch shapes:")
        print(f"  rgb_a:            {batch['rgb_a'].shape}")
        print(f"  rgb_b:            {batch['rgb_b'].shape}")

        if 'sem_a' in batch:
            print(f"  sem_a:            {batch['sem_a'].shape}")
        if 'sem_b' in batch:
            print(f"  sem_b:            {batch['sem_b'].shape}")
        if 'label' in batch:
            print(f"  label:            {batch['label'].shape}")

        print(f"  captions:         {batch['captions'].shape}")
        print(f"  caption_lengths:  {batch['caption_lengths'].shape}")

        # Verify expected shapes
        B = config['training']['batch_size']
        img_size = config['dataset']['image_size']

        assert batch['rgb_a'].shape == (B, 3, img_size, img_size), "rgb_a shape mismatch"
        assert batch['rgb_b'].shape == (B, 3, img_size, img_size), "rgb_b shape mismatch"

        print("\nData loading: PASSED")
        return True

    except Exception as e:
        print(f"\nData loading: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_io():
    """Test model input/output dimensions."""
    print("\n" + "=" * 60)
    print("Testing Model Input/Output")
    print("=" * 60)

    from src.uniscc_v5 import UniSCCV5, UniSCCConfigV5

    # Test settings
    BATCH_SIZE = 2
    IMG_SIZE = 256
    VOCAB_SIZE = 1000
    MAX_LEN = 50
    NUM_CLASSES = 7

    config = UniSCCConfigV5(
        dataset='second_cc',
        pretrained=False,
        vocab_size=VOCAB_SIZE,
        num_semantic_classes=NUM_CLASSES,
        decoder_layers=4,
        decoder_ffn_dim=1024
    )

    model = UniSCCV5(config)

    # Test inputs
    img_t0 = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    img_t1 = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    captions = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_LEN))
    lengths = torch.tensor([30, 25])

    print(f"\nInput shapes:")
    print(f"  img_t0:   {list(img_t0.shape)}")
    print(f"  img_t1:   {list(img_t1.shape)}")
    print(f"  captions: {list(captions.shape)}")
    print(f"  lengths:  {list(lengths.shape)}")

    # Training forward
    model.train()
    outputs = model(img_t0, img_t1, captions, lengths)

    print(f"\nTraining output shapes:")
    print(f"  cd_logits:        {list(outputs['cd_logits'].shape)}")
    print(f"  magnitude:        {list(outputs['magnitude'].shape)}")
    print(f"  caption_logits:   {list(outputs['caption_logits'].shape)}")
    print(f"  change_attention: {list(outputs['change_attention'].shape)}")

    # Verify shapes
    assert outputs['cd_logits'].shape == (BATCH_SIZE, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
    assert outputs['magnitude'].shape == (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
    assert outputs['caption_logits'].shape == (BATCH_SIZE, MAX_LEN, VOCAB_SIZE)
    assert outputs['change_attention'].shape == (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

    # Inference forward
    model.eval()
    with torch.no_grad():
        outputs = model(img_t0, img_t1)

    print(f"\nInference output shapes:")
    print(f"  cd_logits:           {list(outputs['cd_logits'].shape)}")
    print(f"  magnitude:           {list(outputs['magnitude'].shape)}")
    print(f"  generated_captions:  {list(outputs['generated_captions'].shape)}")
    print(f"  change_attention:    {list(outputs['change_attention'].shape)}")

    print("\nModel I/O: PASSED")
    return True


def test_model_parameters():
    """Test and display model parameter counts."""
    print("\n" + "=" * 60)
    print("Model Parameter Analysis")
    print("=" * 60)

    from src.uniscc_v5 import UniSCCV5, UniSCCConfigV5

    # SECOND-CC config (7 classes)
    config_scc = UniSCCConfigV5(
        dataset='second_cc',
        pretrained=False,
        vocab_size=10000,
        num_semantic_classes=7
    )
    model_scc = UniSCCV5(config_scc)

    # LEVIR-MCI config (3 classes)
    config_lmci = UniSCCConfigV5(
        dataset='levir_mci',
        pretrained=False,
        vocab_size=10000,
        num_change_classes=3
    )
    model_lmci = UniSCCV5(config_lmci)

    def count_parameters(model):
        """Count model parameters by component."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Component breakdown
        components = {
            'encoder': sum(p.numel() for p in model.encoder.parameters()),
            'fpn': sum(p.numel() for p in model.fpn.parameters()),
            'alignment': sum(p.numel() for p in model.alignment.parameters()),
            'tdt': sum(p.numel() for p in model.tdt.parameters()),
            'change_attention': sum(p.numel() for p in model.change_attention.parameters()),
            'lsp': sum(p.numel() for p in model.lsp.parameters()),
            'cd_head': sum(p.numel() for p in model.cd_head.parameters()),
            'caption_decoder': sum(p.numel() for p in model.caption_decoder.parameters()),
        }

        return total, trainable, components

    print("\nSECOND-CC Model (7 classes):")
    total, trainable, components = count_parameters(model_scc)
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"\n  Component breakdown:")
    for name, count in components.items():
        pct = 100.0 * count / total
        print(f"    {name:20s}: {count:12,} ({pct:5.1f}%)")

    print("\nLEVIR-MCI Model (3 classes):")
    total, trainable, components = count_parameters(model_lmci)
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    print("\nParameter analysis: PASSED")
    return True


def test_memory_estimation():
    """Estimate GPU memory requirements."""
    print("\n" + "=" * 60)
    print("Memory Estimation")
    print("=" * 60)

    from src.uniscc_v5 import UniSCCV5, UniSCCConfigV5

    config = UniSCCConfigV5(
        dataset='second_cc',
        pretrained=False,
        vocab_size=10000,
        num_semantic_classes=7
    )
    model = UniSCCV5(config)

    # Parameter memory (4 bytes per float32)
    param_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024**3)

    # Gradient memory (same as parameters)
    grad_memory = param_memory

    # Optimizer state (AdamW: 2x parameters for momentum and variance)
    optimizer_memory = 2 * param_memory

    # Activation memory estimation (rough)
    # Depends on batch size and image size
    batch_sizes = [1, 2, 4, 8]

    print(f"\nModel memory: {param_memory:.2f} GB")
    print(f"Gradient memory: {grad_memory:.2f} GB")
    print(f"Optimizer memory: {optimizer_memory:.2f} GB")
    print(f"Base memory: {param_memory + grad_memory + optimizer_memory:.2f} GB")

    print(f"\nEstimated total memory by batch size (256x256 images):")
    for bs in batch_sizes:
        # Rough activation estimate: ~0.5GB per sample for this model
        activation_estimate = bs * 0.5
        total = param_memory + grad_memory + optimizer_memory + activation_estimate
        print(f"  Batch size {bs}: ~{total:.1f} GB")

    print("\nMemory estimation: PASSED")
    return True


def run_all_tests(config_path: str = None):
    """Run all verification tests."""
    print("=" * 60)
    print("UniSCC v5.0 Data and Model Verification")
    print("=" * 60)

    tests = [
        ("Model I/O", test_model_io),
        ("Model Parameters", test_model_parameters),
        ("Memory Estimation", test_memory_estimation),
    ]

    # Add data loading test if config provided or datasets exist
    if config_path or Path("E:/CD-Experiment/Datasets/SECOND-CC-AUG").exists():
        tests.insert(0, ("Data Loading", lambda: test_data_loading(config_path)))

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n{name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data and Model Verification')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML for data loading test')
    args = parser.parse_args()

    success = run_all_tests(args.config)
    sys.exit(0 if success else 1)
