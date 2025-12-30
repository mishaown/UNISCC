# UniSCC: Unified Semantic Change Detection and Change Captioning

A unified framework for joint Semantic Change Detection (SCD) and Change Captioning (CC) on remote sensing imagery.

## v5.0 Features

- **Multi-Scale Architecture**: Hierarchical feature pyramid with 4 scales for multi-resolution change detection
- **Hierarchical Alignment**: Cross-scale feature alignment between bitemporal images
- **Multi-Task CD Head**: Semantic change prediction with magnitude estimation
- **Multi-Level Caption Decoder**: Scale-aware caption generation
- **Change-Aware Attention**: Spatial attention guided by change magnitude

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

# Create model with v5.0 multi-scale architecture
config = UniSCCConfig(
    dataset='second_cc',
    vocab_size=10000,
    pretrained=True,
    use_pyramid=True,
    num_scales=4,
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

# v5.0 outputs
cd_logits = outputs['cd_logits']              # [B, K, 256, 256] semantic change map
caption_logits = outputs['caption_logits']    # [B, T, V]
magnitude = outputs.get('magnitude')          # [B, 1, 256, 256] change magnitude

# Inference
model.eval()
with torch.no_grad():
    outputs = model(img_t0, img_t1)
    generated = outputs['generated_captions']
```

### Run Tests

```bash
python test_model.py
python test_data.py
```

## Project Structure

```
UniSCC/
├── configs/
│   ├── second_cc.yaml          # SECOND-CC v5.0 config
│   ├── levir_mci.yaml          # LEVIR-MCI v5.0 config
│   ├── second_cc_linux.yaml    # Linux paths
│   └── levir_mci_linux.yaml    # Linux paths
├── src/
│   ├── encoder.py              # Siamese Swin Transformer
│   ├── fpn.py                  # Feature Pyramid Network
│   ├── hierarchical_alignment.py   # Multi-scale alignment
│   ├── multi_scale_tdt.py      # Multi-scale Temporal Difference Transformer
│   ├── change_aware_attention.py   # Change-guided attention
│   ├── hierarchical_lsp.py     # Hierarchical Learnable Semantic Prompts
│   ├── multi_task_cd_head.py   # Multi-task CD head with magnitude
│   ├── multi_level_caption_decoder.py  # Scale-aware caption decoder
│   ├── uniscc_v5.py            # Main model (v5.0)
│   └── __init__.py
├── data/
│   ├── second_cc.py            # SECOND-CC dataset with RGB label handling
│   ├── levir_mci.py            # LEVIR-MCI dataset
│   ├── vocabulary.py           # Caption vocabulary
│   └── transforms.py           # Image transforms
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

### Training Commands

```bash
# SECOND-CC (batch=2, accum=6, effective=12)
python train.py --config configs/second_cc.yaml

# LEVIR-MCI
python train.py --config configs/levir_mci.yaml

# Resume training
python train.py --config configs/second_cc.yaml --resume checkpoints/last.pth
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 2 | Batch size per step (reduced for v5.0) |
| `gradient_accumulation_steps` | 6 | Accumulation steps (effective batch=12) |
| `gradient_checkpointing` | true | Required for v5.0 due to larger model |
| `num_workers` | 8 | DataLoader workers |
| `amp.enabled` | true | Mixed precision training |

## Evaluation

```bash
python evaluate.py --config configs/second_cc.yaml \
                   --checkpoint checkpoints/best.pth \
                   --split test
```

### Evaluation Metrics
- **CD metrics**: mIoU, F1, OA, boundary_IoU
- **Caption metrics**: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr
- **Magnitude metrics**: MSE, MAE

## Inference

### Single Image Pair

```bash
python inference.py --config configs/second_cc.yaml \
                    --checkpoint checkpoints/best.pth \
                    --image_a before.png \
                    --image_b after.png \
                    --output_dir outputs/
```

### Directory Batch Processing

Both datasets use folder-based organization with separate A/ and B/ directories:

```bash
# SECOND-CC dataset structure: root/split/rgb/A/ and root/split/rgb/B/
python inference.py --config configs/second_cc.yaml \
                    --checkpoint checkpoints/best.pth \
                    --dir_a E:/CD-Experiment/Datasets/SECOND-CC-AUG/test/rgb/A \
                    --dir_b E:/CD-Experiment/Datasets/SECOND-CC-AUG/test/rgb/B \
                    --output_dir outputs/

# LEVIR-MCI dataset structure: root/images/split/A/ and root/images/split/B/
python inference.py --config configs/levir_mci.yaml \
                    --checkpoint checkpoints/best.pth \
                    --dir_a E:/CD-Experiment/Datasets/LEVIR-MCI-dataset/images/test/A \
                    --dir_b E:/CD-Experiment/Datasets/LEVIR-MCI-dataset/images/test/B \
                    --output_dir outputs/
```

### Inference Outputs
- `*_change_map.npy`: Semantic change map
- `*_caption.txt`: Generated caption
- `*_viz.png`: Visualization

## Model Architecture (v5.0)

```
img_t0, img_t1 [B,3,256,256]
       │
       ▼
┌──────────────────────┐
│ Encoder (Swin-B)     │──► Multi-scale features
│ + FPN                │    P2[64], P3[32], P4[16], P5[8]
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Hierarchical         │──► Aligned features per scale
│ Alignment            │    + skip connections
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Multi-Scale TDT      │──► Difference features per scale
│ (2 layers/scale)     │    with change-aware attention
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Hierarchical LSP     │──► Scale-aware semantic prompts
│ (CLIP-initialized)   │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Multi-Task CD Head   │──► cd_logits [B,K,256,256]
│ (with magnitude)     │    magnitude [B,1,256,256]
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Multi-Level Caption  │──► caption_logits [B,T,V]
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

    # v5.0 Multi-scale
    use_pyramid=True,
    pyramid_channels=256,
    num_scales=4,

    # Architecture
    tdt_layers=2,                     # Per-scale layers
    decoder_layers=6,
    num_semantic_classes=7,           # SECOND-CC classes
    num_change_classes=3,             # LEVIR-MCI classes

    # Caption
    vocab_size=10000,
    max_caption_length=50,
)
```

## Dataset Directory Structure

### SECOND-CC
```
SECOND-CC-AUG/
├── SECOND-CC-AUG.json    # Annotations with splits
├── train/
│   ├── rgb/
│   │   ├── A/            # Before images
│   │   └── B/            # After images
│   └── sem/
│       ├── A/            # Before semantic maps
│       └── B/            # After semantic maps
├── val/
│   └── ...
└── test/
    └── ...
```

### LEVIR-MCI
```
LEVIR-MCI-dataset/
├── cls_LEVIR_MCI.json    # Annotations
├── images/
│   ├── train/
│   │   ├── A/            # Before images
│   │   ├── B/            # After images
│   │   ├── label/        # Binary change masks
│   │   └── label_rgb/    # RGB change masks
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── ...
```

## Performance Optimization

| Technique | Effect | Trade-off |
|-----------|--------|-----------|
| Gradient Accumulation | Simulate larger batch | None |
| Gradient Checkpointing | ~40% memory reduction | ~20% slower |
| Mixed Precision (AMP) | ~50% memory reduction | Minimal |
| cuDNN Benchmark | Faster convolutions | First batch slower |
| Persistent Workers | Faster data loading | Higher memory |

## License

MIT License
