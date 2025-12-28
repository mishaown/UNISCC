# UniSCC: Unified Semantic Change Detection and Change Captioning

A unified framework for joint Semantic Change Detection (SCD) and Change Captioning (CC) on remote sensing imagery.

## v4.0 Features

- **Feature Alignment**: Cross-attention, deformable, or hierarchical alignment for bitemporal images
- **Single Semantic Head**: Predicts after-change semantic classes with shared CD-captioning space
- **Learnable Semantic Prompts**: CLIP-initialized prompts for semantic-aware feature enhancement
- **Optimized Training**: cuDNN benchmark, non-blocking transfers, persistent workers

## Supported Datasets

### SECOND-CC (Semantic Change Detection + Change Captioning)
| Attribute | Value |
|-----------|-------|
| Images | 6,041 bitemporal pairs (256x256) |
| Task | 7-class semantic change detection + captioning |
| Annotations | Semantic maps (RGB) + 5 captions/image |
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

# Create model with v4.0 feature alignment
config = UniSCCConfig(
    dataset='second_cc',
    vocab_size=10000,
    pretrained=True,
    use_alignment=True,
    alignment_type='cross_attention',  # or 'deformable', 'hierarchical'
    alignment_heads=8
)
model = UniSCC(config)

# Prepare inputs
img_t0 = torch.randn(2, 3, 256, 256)  # Before
img_t1 = torch.randn(2, 3, 256, 256)  # After
captions = torch.randint(0, 10000, (2, 50))
lengths = torch.tensor([30, 25])

# Training forward pass
model.train()
outputs = model(img_t0, img_t1, captions, lengths)

# v4.0 outputs
cd_logits = outputs['cd_logits']              # [B, K, 256, 256] semantic change map
caption_logits = outputs['caption_logits']    # [B, T, V]
alignment_conf = outputs['alignment_confidence']  # [B, H, W] confidence map

# Inference
model.eval()
with torch.no_grad():
    outputs = model(img_t0, img_t1)
    generated = outputs['generated_captions']
```

### Run Tests

```bash
python test_v4.py
```

## Project Structure

```
UniSCC/
├── configs/
│   ├── second_cc.yaml
│   └── levir_mci.yaml
├── src/
│   ├── encoder.py           # Siamese Swin Transformer
│   ├── alignment.py         # Feature alignment modules (v4.0)
│   ├── tdt.py               # Temporal Difference Transformer
│   ├── lsp.py               # Learnable Semantic Prompts
│   ├── semantic_head.py     # Semantic Change Detection Head
│   ├── caption_decoder.py   # Semantic Caption Decoder
│   ├── uniscc.py            # Main model (v4.0)
│   └── __init__.py
├── data/
│   ├── second_cc.py         # SECOND-CC dataset with RGB label handling
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

### Optimized Training (Default)

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
| `gradient_checkpointing` | false | Trade compute for memory (enable if OOM) |
| `num_workers` | 8 | DataLoader workers |
| `amp.enabled` | true | Mixed precision training |

## Evaluation

```bash
python evaluate.py --config configs/second_cc.yaml \
                   --checkpoint checkpoints/best.pth \
                   --split test
```

### v4.0 Evaluation Output
- **CD metrics**: mIoU, F1, OA for after-change semantic prediction
- **Caption metrics**: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr

## Inference

```bash
python inference.py --config configs/second_cc.yaml \
                    --checkpoint checkpoints/best.pth \
                    --image_a before.png \
                    --image_b after.png \
                    --output_dir outputs/
```

### v4.0 Inference Outputs
- `*_change_map.npy`: Semantic change map (after-change classes)
- `*_caption.txt`: Generated caption
- `*_viz.png`: Visualization

## Model Architecture (v4.0)

```
img_t0, img_t1 [B,3,256,256]
       │
       ▼
┌──────────────────────┐
│ Encoder (Swin-B)     │──► feat_t0, feat_t1 [B,1024,8,8]
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Feature Alignment    │──► aligned_t0, aligned_t1 + confidence
│ (Cross-Attention)    │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ TDT (3 layers)       │──► diff_features [B,512,8,8]
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Learnable Semantic   │──► semantic_features [B,512,8,8]
│ Prompts (CLIP-init)  │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Semantic Change Head │──► cd_logits [B,K,256,256]
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Caption Decoder      │──► caption_logits [B,T,V]
│ (Shared Semantic)    │
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

    # v4.0 Alignment
    use_alignment=True,
    alignment_type='cross_attention',  # 'cross_attention', 'deformable', 'hierarchical'
    alignment_heads=8,

    # Architecture
    tdt_layers=3,
    decoder_layers=6,
    num_semantic_classes=7,           # SECOND-CC after-change classes
    num_change_classes=3,             # LEVIR-MCI classes

    # Caption
    vocab_size=10000,
    max_caption_length=50,
)
```

## Alignment Types

| Type | Description | Best For |
|------|-------------|----------|
| `cross_attention` | Query from t1, Key/Value from t0 | General use, best quality |
| `deformable` | Learnable spatial offsets | Large misalignments |
| `hierarchical` | Multi-scale alignment | Multi-resolution changes |

## Performance Optimization

| Technique | Effect | Trade-off |
|-----------|--------|-----------|
| Gradient Accumulation | Simulate larger batch | None |
| Gradient Checkpointing | ~40% memory reduction | ~20% slower |
| Mixed Precision (AMP) | ~50% memory reduction | Minimal |
| cuDNN Benchmark | Faster convolutions | First batch slower |
| Persistent Workers | Faster data loading | Higher memory |

Adjust `batch_size` and `gradient_accumulation_steps` in config based on available memory.

## License

MIT License
