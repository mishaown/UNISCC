# UniSCC: Unified Semantic Change Detection and Change Captioning

A unified framework for joint Semantic Change Detection (SCD) and Change Captioning (CC) on remote sensing imagery.

## v3.0 Features

- **Dual Semantic Head**: Predicts both before-change (sem_A) and after-change (sem_B) semantic maps
- **Transition-aware Captioning**: Uses transition embeddings for "what changed into what" descriptions
- **Derived Change Mask**: Automatically computes binary change mask where sem_A != sem_B
- **Focal Loss**: Handles class imbalance with configurable focal loss
- **Memory Optimization**: Gradient checkpointing and accumulation for large models

## Supported Datasets

### SECOND-CC (Semantic Change Detection + Change Captioning)
| Attribute | Value |
|-----------|-------|
| Images | 6,041 bitemporal pairs (256x256) |
| Task | 7-class semantic change detection + captioning |
| Annotations | Semantic maps (sem_a, sem_b) + 5 captions/image |
| Classes | background, low_vegetation, ground, tree, water, building, playground |

### LEVIR-MCI (Multi-level Change Interpretation)
| Attribute | Value |
|-----------|-------|
| Images | 10,077 bitemporal pairs (256x256, 0.5m/px) |
| Task | 3-class semantic change detection + captioning |
| Annotations | RGB change masks + 5 captions/image |
| Classes | no_change (black), building (red), road (blue) |

## Installation

```bash
conda create -n uniscc python=3.9
conda activate uniscc
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src import UniSCC, UniSCCConfig
import torch

# Create model with v3.0 dual semantic head
config = UniSCCConfig(
    dataset='second_cc',
    vocab_size=10000,
    pretrained=True,
    dual_head=True,
    use_transition_attention=True
)
model = UniSCC(config)

# Enable gradient checkpointing for memory optimization
model.set_gradient_checkpointing(True)

# Prepare inputs
img_t0 = torch.randn(2, 3, 256, 256)  # Before
img_t1 = torch.randn(2, 3, 256, 256)  # After
captions = torch.randint(0, 10000, (2, 50))
lengths = torch.tensor([30, 25])

# Training forward pass
model.train()
outputs = model(img_t0, img_t1, captions, lengths)

# v3.0 outputs
sem_A_logits = outputs['sem_A_logits']    # [B, 7, 256, 256] before-change
sem_B_logits = outputs['sem_B_logits']    # [B, 7, 256, 256] after-change
change_mask = outputs['change_mask']       # [B, 256, 256] derived
caption_logits = outputs['caption_logits'] # [B, T, V]
cd_logits = outputs['cd_logits']           # Alias for sem_B_logits

# Inference
model.eval()
with torch.no_grad():
    outputs = model(img_t0, img_t1)
    generated = outputs['generated_captions']
```

### Run Sanity Checks

```bash
python -m src.sanity_check
```

## Project Structure

```
UniSCC/
├── configs/
│   ├── second_cc.yaml
│   └── levir_mci.yaml
├── src/
│   ├── encoder.py           # Siamese Swin Transformer + gradient checkpointing
│   ├── tdt.py               # Temporal Difference Transformer
│   ├── lsp.py               # TransitionLSP with dual prompts
│   ├── semantic_head.py     # DualSemanticHead + DualSemanticLoss
│   ├── caption_decoder.py   # TransitionCaptionDecoder
│   ├── uniscc.py            # Main model (v3.0)
│   └── sanity_check.py      # Model verification tests
├── data/
│   ├── second_cc.py
│   ├── levir_mci.py
│   ├── vocabulary.py
│   └── transforms.py
├── losses/
│   ├── caption_loss.py
│   └── scd_loss.py
├── utils/
│   └── metrics.py
├── train.py
├── evaluate.py
├── inference.py
└── README.md
```

## Training

### Memory-Optimized Training (Default)

```bash
# SECOND-CC (batch=4, accum=3, effective=12)
python train.py --config configs/second_cc.yaml

# LEVIR-MCI
python train.py --config configs/levir_mci.yaml

# Resume training
python train.py --config configs/second_cc.yaml --resume checkpoints/last.pth
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Batch size per step |
| `gradient_accumulation_steps` | 3 | Accumulation steps (effective batch=12) |
| `gradient_checkpointing` | true | Trade compute for memory |
| `num_workers` | 4 | DataLoader workers |
| `amp.enabled` | true | Mixed precision training |

## Evaluation

```bash
python evaluate.py --config configs/second_cc.yaml \
                   --checkpoint checkpoints/best.pth \
                   --split test
```

### v3.0 Evaluation Output
- **sem_A metrics**: mIoU, F1, OA for before-change prediction
- **sem_B metrics**: mIoU, F1, OA for after-change prediction (primary)
- **Caption metrics**: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr

## Inference

```bash
python inference.py --config configs/second_cc.yaml \
                    --checkpoint checkpoints/best.pth \
                    --image_a before.png \
                    --image_b after.png \
                    --output_dir outputs/
```

### v3.0 Inference Outputs
- `*_sem_A.npy`: Before-change semantic map
- `*_sem_B.npy`: After-change semantic map
- `*_change_mask.npy`: Derived binary change mask
- `*_caption.txt`: Generated caption
- `*_viz.png`: 2x3 visualization grid

## Model Architecture (v3.0)

```
img_t0, img_t1 [B,3,256,256]
       │
       ▼
┌──────────────────────┐
│ Encoder (Swin-B)     │──► feat_t0, feat_t1 [B,512,8,8]
│ + Gradient Checkpoint│
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ TDT (3 layers)       │──► {diff, feat_t0_enh, feat_t1_enh}
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ TransitionLSP        │──► prompts_A, prompts_B, transitions
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ DualSemanticHead     │──► sem_A_logits [B,K,256,256]
│                      │──► sem_B_logits [B,K,256,256]
│                      │──► change_mask [B,256,256]
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ TransitionCaption    │──► caption_logits [B,T,V]
│ Decoder              │
└──────────────────────┘
```

## Configuration

```python
UniSCCConfig(
    # Dataset
    dataset='second_cc',              # 'second_cc' or 'levir_mci'

    # Encoder
    backbone='swin_base_patch4_window7_224',
    pretrained=True,
    feature_dim=512,

    # Architecture
    tdt_layers=3,
    decoder_layers=6,
    num_semantic_classes=7,           # SECOND-CC
    num_change_classes=3,             # LEVIR-MCI

    # v3.0 Dual Head
    dual_head=True,
    share_decoder=True,
    use_transition_attention=True,

    # v3.0 Loss
    use_focal_loss=True,
    focal_gamma=2.0,
    sem_A_weight=1.0,
    sem_B_weight=1.0,
)
```

## Memory Optimization

The model supports several memory optimization techniques:

| Technique | Effect | Trade-off |
|-----------|--------|-----------|
| Gradient Accumulation | Simulate larger batch | None |
| Gradient Checkpointing | ~40% memory reduction | ~20% slower |
| Mixed Precision (AMP) | ~50% memory reduction | Minimal |

Adjust `batch_size` and `gradient_accumulation_steps` in config based on available memory.

## License

MIT License
