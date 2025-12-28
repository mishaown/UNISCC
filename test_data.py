"""Comprehensive test for SECOND-CC data loading and model."""
import sys
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
import numpy as np
from data.second_cc import SECONDCCDataset

def main():
    print("=" * 60)
    print("Testing SECOND-CC data loading with COLOR_MAP fix...")
    print("=" * 60)

    root = "E:/CD-Experiment/Datasets/SECOND-CC-AUG"

    # Create dataset
    train_dataset = SECONDCCDataset(
        root=root,
        split='train',
        max_caption_length=50,
        min_word_freq=5
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Vocab size: {len(train_dataset.vocab)}")

    # Get a single sample
    sample = train_dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"rgb_a shape: {sample['rgb_a'].shape}")
    print(f"sem_a shape: {sample['sem_a'].shape}")
    print(f"sem_b shape: {sample['sem_b'].shape}")

    sem_a_unique = sample['sem_a'].unique().tolist()
    sem_b_unique = sample['sem_b'].unique().tolist()
    print(f"sem_a unique values: {sem_a_unique}")
    print(f"sem_b unique values: {sem_b_unique}")

    # Verify class indices are in valid range (0-6)
    valid_classes = all(0 <= v <= 6 for v in sem_a_unique) and all(0 <= v <= 6 for v in sem_b_unique)
    print(f"Class indices valid (0-6): {valid_classes}")

    if not valid_classes:
        print("ERROR: Invalid class indices detected!")
        return

    print("\nData loading: PASSED!")

    # Test model
    print("\n" + "=" * 60)
    print("Testing SECOND-CC model forward pass...")
    print("=" * 60)

    from src.uniscc import UniSCC, UniSCCConfig

    config = UniSCCConfig(
        dataset='second_cc',
        pretrained=False,
        vocab_size=len(train_dataset.vocab),
        num_semantic_classes=7,
        num_change_classes=3
    )
    model = UniSCC(config)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Create a batch
    batch_rgb_a = torch.stack([sample['rgb_a'], sample['rgb_a']])
    batch_rgb_b = torch.stack([sample['rgb_b'], sample['rgb_b']])
    batch_captions = torch.stack([sample['captions'][0], sample['captions'][0]])
    batch_lengths = torch.tensor([sample['caption_lengths'][0].item(), sample['caption_lengths'][0].item()])
    batch_sem_a = torch.stack([sample['sem_a'], sample['sem_a']])
    batch_sem_b = torch.stack([sample['sem_b'], sample['sem_b']])

    print(f"\nInput shapes:")
    print(f"  rgb_a: {batch_rgb_a.shape}")
    print(f"  rgb_b: {batch_rgb_b.shape}")
    print(f"  sem_a: {batch_sem_a.shape}")
    print(f"  sem_b: {batch_sem_b.shape}")

    # Training forward
    model.train()
    outputs = model(batch_rgb_a, batch_rgb_b, batch_captions, batch_lengths)

    print(f"\nOutput shapes:")
    print(f"  sem_A_logits: {outputs['sem_A_logits'].shape}")
    print(f"  sem_B_logits: {outputs['sem_B_logits'].shape}")
    print(f"  caption_logits: {outputs['caption_logits'].shape}")

    # Test loss computation
    print("\n" + "=" * 60)
    print("Testing loss computation...")
    print("=" * 60)

    # sem_A loss
    loss_A = F.cross_entropy(outputs['sem_A_logits'], batch_sem_a, ignore_index=255)
    print(f"sem_A loss: {loss_A.item():.4f}")

    # sem_B loss
    loss_B = F.cross_entropy(outputs['sem_B_logits'], batch_sem_b, ignore_index=255)
    print(f"sem_B loss: {loss_B.item():.4f}")

    # Caption loss
    cap_logits = outputs['caption_logits'][:, :-1]
    cap_targets = batch_captions[:, 1:]
    loss_cap = F.cross_entropy(
        cap_logits.reshape(-1, cap_logits.size(-1)),
        cap_targets.reshape(-1),
        ignore_index=0
    )
    print(f"caption loss: {loss_cap.item():.4f}")

    # Total loss
    total_loss = loss_A + loss_B + loss_cap
    print(f"total loss: {total_loss.item():.4f}")

    # Verify losses are reasonable (not NaN or inf)
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("\nERROR: Loss is NaN or Inf!")
        return

    print("\nLoss computation: PASSED!")
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

if __name__ == '__main__':
    main()
