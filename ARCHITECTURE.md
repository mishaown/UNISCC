# UniSCC v3.0 Architecture

## Overview

UniSCC v3.0 implements a **Dual Semantic Head** architecture with **Transition-aware Captioning**. The model predicts both before-change (sem_A) and after-change (sem_B) semantic maps, then uses transition embeddings to generate descriptive captions.

## Key Innovations

1. **DualSemanticHead**: Predicts semantics at both timestamps using shared decoder
2. **TransitionLSP**: Generates transition embeddings for (class_A → class_B) pairs
3. **TransitionCaptionDecoder**: Attends to transition embeddings for better captions
4. **Derived Change Mask**: Automatically computed where sem_A != sem_B

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UniSCC v3.0 ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Image T0 ─────┐         ┌──────────────────┐                              │
│   [B,3,256,256] │         │  Siamese Encoder │    feat_t0 [B,512,8,8]       │
│                 ├────────►│  (Swin-B Shared) │───►                          │
│   Image T1 ─────┘         │  + Grad Ckpt     │    feat_t1 [B,512,8,8]       │
│   [B,3,256,256]           └──────────────────┘                              │
│                                    │                                        │
│                                    ▼                                        │
│                  ┌─────────────────────────────────┐                        │
│                  │  Temporal Difference Transformer │                       │
│                  │  (TDT - 3 Bidirectional Blocks)  │                       │
│                  └─────────────────────────────────┘                        │
│                                    │                                        │
│                  ┌─────────────────┼─────────────────┐                      │
│                  │                 │                 │                      │
│                  ▼                 ▼                 ▼                      │
│           feat_t0_enh         diff_feat         feat_t1_enh                 │
│           [B,512,8,8]        [B,512,8,8]        [B,512,8,8]                  │
│                  │                 │                 │                      │
│                  └────────────┬────┴────────────────┘                       │
│                               │                                             │
│                               ▼                                             │
│                  ┌─────────────────────────────────┐                        │
│                  │       TransitionLSP             │                        │
│                  │  ┌─────────┐  ┌─────────┐      │                        │
│                  │  │prompts_A│  │prompts_B│      │                        │
│                  │  │  [K,D]  │  │  [K,D]  │      │                        │
│                  │  └────┬────┘  └────┬────┘      │                        │
│                  │       └─────┬──────┘           │                        │
│                  │             ▼                  │                        │
│                  │    transition_embed [K*K,D]    │                        │
│                  └─────────────────────────────────┘                        │
│                               │                                             │
│            ┌──────────────────┼──────────────────┐                         │
│            │                  │                  │                         │
│            ▼                  ▼                  ▼                         │
│   ┌─────────────────┐  ┌───────────────┐  ┌─────────────────┐              │
│   │ DualSemanticHead│  │  Change Mask  │  │TransitionCaption│              │
│   │                 │  │  Derivation   │  │    Decoder      │              │
│   │ sem_A_logits    │  │               │  │                 │              │
│   │ sem_B_logits    │  │ A != B        │  │ Visual + Trans  │              │
│   │ [B,K,256,256]   │  │ [B,256,256]   │  │ Cross-Attention │              │
│   └─────────────────┘  └───────────────┘  └─────────────────┘              │
│            │                  │                  │                         │
│            ▼                  ▼                  ▼                         │
│       sem_A_logits      change_mask       caption_logits                   │
│       sem_B_logits                        [B,T,Vocab]                      │
│       cd_logits (=sem_B)                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Encoder (`src/encoder.py`)

**UniSCCEncoder**: Siamese Swin Transformer with gradient checkpointing.

```python
class UniSCCEncoder:
    - backbone: Swin-B (88M params)
    - temporal_embed: Learnable t0/t1 embeddings
    - gradient_checkpointing: Memory optimization

    Input:  img_t0, img_t1 [B, 3, 256, 256]
    Output: feat_t0, feat_t1 [B, 512, 8, 8]
```

### 2. TDT (`src/tdt.py`)

**TemporalDifferenceTransformer**: Bidirectional cross-temporal attention.

```python
class TemporalDifferenceTransformer:
    - num_layers: 3
    - bidirectional: t0→t1 and t1→t0 attention

    Input:  feat_t0, feat_t1
    Output: {
        'diff': difference features [B, 512, 8, 8],
        'feat_t0_enhanced': [B, 512, 8, 8],
        'feat_t1_enhanced': [B, 512, 8, 8]
    }
```

### 3. TransitionLSP (`src/lsp.py`)

**TransitionLSP**: Dual semantic prompts with transition embeddings.

```python
class TransitionLSP:
    - prompts_A: Before-change semantic prompts [K, D]
    - prompts_B: After-change semantic prompts [K, D]
    - transition_embed: [K*K, D] for (class_i → class_j) pairs

    Input:  feat_t0_enhanced, feat_t1_enhanced
    Output: {
        'prompts_A': [K, D],
        'prompts_B': [K, D],
        'transitions': [B, H*W, D] per-pixel transition embeddings
    }
```

### 4. DualSemanticHead (`src/semantic_head.py`)

**DualSemanticHead**: Predicts semantics at both timestamps.

```python
class DualSemanticHead:
    - shared_decoder: Upsampling 8×8 → 256×256
    - cosine_similarity: With semantic prompts
    - temperature: 0.1

    Input:  feat_t0_enhanced, feat_t1_enhanced, prompts_A, prompts_B
    Output: {
        'sem_A_logits': [B, K, 256, 256],
        'sem_B_logits': [B, K, 256, 256],
        'change_mask': [B, 256, 256]  # where A != B
    }
```

### 5. TransitionCaptionDecoder (`src/caption_decoder.py`)

**TransitionCaptionDecoder**: Caption generation with transition attention.

```python
class TransitionCaptionDecoder:
    - num_layers: 6
    - attention: Self + Visual + Transition cross-attention
    - gated_fusion: Balance visual and transition information

    Input:  visual_features, transition_embeddings, captions
    Output: caption_logits [B, T, Vocab]
```

## Data Flow

```
1. ENCODING
   img_t0, img_t1 → Encoder → feat_t0, feat_t1

2. TEMPORAL MODELING
   feat_t0, feat_t1 → TDT → diff, feat_t0_enh, feat_t1_enh

3. SEMANTIC PROMPTS
   feat_t0_enh, feat_t1_enh → TransitionLSP → prompts_A, prompts_B, transitions

4. DUAL PREDICTION
   feat_t0_enh + prompts_A → DualSemanticHead → sem_A_logits
   feat_t1_enh + prompts_B → DualSemanticHead → sem_B_logits
   sem_A != sem_B → change_mask

5. CAPTION GENERATION
   visual + transitions → TransitionCaptionDecoder → caption_logits
```

## Loss Functions

### DualSemanticLoss

```python
loss = sem_A_weight * focal_loss(sem_A_logits, target_A) +
       sem_B_weight * focal_loss(sem_B_logits, target_B)
```

- **Focal Loss**: γ=2.0 for class imbalance
- **Label Smoothing**: 0.1
- **Ignore Index**: 255

### CaptionLoss

```python
loss = cross_entropy(caption_logits, targets, label_smoothing=0.1)
```

### Total Loss

```python
total = scd_weight * dual_semantic_loss + caption_weight * caption_loss
```

## Memory Optimization

### Gradient Checkpointing

```python
model.set_gradient_checkpointing(True)
```

- Saves ~30-50% memory
- ~20% slower training
- Applied to: Encoder, TDT, CaptionDecoder

### Gradient Accumulation

```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 3  # Effective batch = 12
```

### Mixed Precision (AMP)

```yaml
training:
  amp:
    enabled: true
```

## Model Outputs

### Training Mode

```python
outputs = model(img_t0, img_t1, captions, lengths)

outputs = {
    'sem_A_logits': [B, K, 256, 256],    # Before-change
    'sem_B_logits': [B, K, 256, 256],    # After-change
    'cd_logits': [B, K, 256, 256],       # Alias for sem_B
    'change_mask': [B, 256, 256],        # Derived
    'caption_logits': [B, T, V],
    'prompts_A': [K, D],
    'prompts_B': [K, D],
}
```

### Inference Mode

```python
outputs = model(img_t0, img_t1)

outputs = {
    'sem_A_logits': [B, K, 256, 256],
    'sem_B_logits': [B, K, 256, 256],
    'cd_logits': [B, K, 256, 256],
    'change_mask': [B, 256, 256],
    'generated_captions': [B, T],
}
```

## Parameter Count

| Component | Parameters |
|-----------|------------|
| Encoder (Swin-B) | ~88M |
| TDT | ~2M |
| TransitionLSP | ~10K |
| DualSemanticHead | ~15M |
| TransitionCaptionDecoder | ~38M |
| **Total** | **~143M** |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2024-12 | Initial release with transition classes |
| v2.0 | 2024-12 | Shared semantic space architecture |
| v3.0 | 2025-12 | Dual semantic head + transition attention + memory optimization |

---

**Version**: 3.0.0
**Date**: 2025-12-27
