# UniSCC v3.0: Project Summary

## Overview

UniSCC is a unified framework for **joint semantic change detection and change captioning** in remote sensing imagery. Version 3.0 introduces a **Dual Semantic Head** architecture that predicts both before and after-change semantics with transition-aware captioning.

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            UniSCC v3.0                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Image T0 ─────┐                                                           │
│   [B,3,256,256] │    ┌──────────────────┐                                   │
│                 ├───►│  Siamese Encoder │───► feat_t0, feat_t1 [B,512,8,8]  │
│   Image T1 ─────┘    │  (Swin-B + Ckpt) │                                   │
│   [B,3,256,256]      └──────────────────┘                                   │
│                              │                                              │
│                              ▼                                              │
│                 ┌────────────────────────┐                                  │
│                 │  TDT (3 Bidir Blocks)  │                                  │
│                 └────────────────────────┘                                  │
│                              │                                              │
│          ┌───────────────────┼───────────────────┐                          │
│          ▼                   ▼                   ▼                          │
│    feat_t0_enh          diff_feat          feat_t1_enh                      │
│          │                   │                   │                          │
│          │                   ▼                   │                          │
│          │          ┌─────────────────┐          │                          │
│          │          │  TransitionLSP  │          │                          │
│          │          │ prompts_A/B     │          │                          │
│          │          │ transitions     │          │                          │
│          │          └─────────────────┘          │                          │
│          │                   │                   │                          │
│          ▼                   │                   ▼                          │
│   ┌───────────────┐          │          ┌───────────────┐                   │
│   │DualSemanticHd │◄─────────┴─────────►│TransitionCap  │                   │
│   │               │                     │Decoder        │                   │
│   └───────────────┘                     └───────────────┘                   │
│          │                                      │                           │
│          ▼                                      ▼                           │
│   sem_A [B,K,256,256]                   caption_logits [B,T,V]              │
│   sem_B [B,K,256,256]                                                       │
│   change_mask [B,256,256]                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

| Component | File | Description |
|-----------|------|-------------|
| **Encoder** | `src/encoder.py` | Siamese Swin-B with gradient checkpointing |
| **TDT** | `src/tdt.py` | Temporal Difference Transformer (3 layers) |
| **TransitionLSP** | `src/lsp.py` | Dual prompts + transition embeddings |
| **DualSemanticHead** | `src/semantic_head.py` | Predicts sem_A and sem_B |
| **TransitionCaptionDecoder** | `src/caption_decoder.py` | Transition-aware captioning |
| **UniSCC** | `src/uniscc.py` | Main model (~143M params) |

## Datasets

| Dataset | Images | CD Classes | Caption |
|---------|--------|------------|---------|
| SECOND-CC | 6,041 pairs | 7 semantic | 5/image |
| LEVIR-MCI | 10,077 pairs | 3 change type | 5/image |

### SECOND-CC Classes
background, low_vegetation, non_vegetated_ground, tree, water, building, playground

### LEVIR-MCI Classes
no_change (black), building (red), road (blue)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Gradient Accumulation | 3 steps |
| Effective Batch | 12 |
| Optimizer | AdamW (lr=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts |
| AMP | Enabled |
| Gradient Checkpointing | Enabled |
| Epochs | 10 |

## Loss Functions

```python
# Dual Semantic Loss (with Focal)
dual_loss = sem_A_weight * focal(sem_A, target_A) +
            sem_B_weight * focal(sem_B, target_B)

# Caption Loss
caption_loss = cross_entropy(logits, targets, smoothing=0.1)

# Total
total = scd_weight * dual_loss + caption_weight * caption_loss
```

## Evaluation Metrics

### Change Detection
- mIoU (mean Intersection over Union)
- F1 Score
- Overall Accuracy (OA)
- Per-class IoU

### Captioning
- BLEU-1/2/3/4
- METEOR
- ROUGE-L
- CIDEr

## File Structure

```
UNISCC/
├── src/                      # Core model
│   ├── uniscc.py             # Main model + config
│   ├── encoder.py            # Swin encoder
│   ├── tdt.py                # Temporal transformer
│   ├── lsp.py                # Semantic prompts
│   ├── semantic_head.py      # Dual semantic head
│   ├── caption_decoder.py    # Caption decoder
│   └── sanity_check.py       # Model tests
├── data/                     # Dataset loaders
├── losses/                   # Loss functions
├── utils/                    # Metrics
├── configs/                  # YAML configs
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
└── inference.py              # Inference script
```

## Usage

### Training

```bash
python train.py --config configs/second_cc.yaml
python train.py --config configs/levir_mci.yaml
```

### Evaluation

```bash
python evaluate.py --config configs/second_cc.yaml \
                   --checkpoint checkpoints/best.pth
```

### Inference

```bash
python inference.py --config configs/second_cc.yaml \
                    --checkpoint checkpoints/best.pth \
                    --image_a before.png \
                    --image_b after.png
```

### Sanity Check

```bash
python -m src.sanity_check
```

## Memory Optimization

| Technique | Effect |
|-----------|--------|
| Gradient Accumulation | Simulate larger effective batch size |
| Gradient Checkpointing | Reduce memory by ~40% |
| Mixed Precision (AMP) | Reduce memory by ~50% |

## v3.0 Changes

1. **Dual Semantic Head**: Predicts both sem_A and sem_B
2. **TransitionLSP**: Dual prompts with transition embeddings
3. **TransitionCaptionDecoder**: Transition-aware attention
4. **Focal Loss**: For class imbalance
5. **Memory Optimization**: Gradient checkpointing + accumulation

---

**Version**: 3.0.0
**Date**: 2025-12-27
