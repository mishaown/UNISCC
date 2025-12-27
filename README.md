# UniSCC: Unified Semantic Change Detection and Change Captioning

A unified framework for joint Semantic Change Detection (SCD) and Change Captioning (CC) on remote sensing imagery.

## v3.0 Features

- **Dual Semantic Head**: Predicts both before-change (sem_A) and after-change (sem_B) semantic maps
- **Transition-aware Captioning**: Uses transition embeddings from semantic changes for better caption generation
- **Derived Change Mask**: Automatically computes binary change mask where sem_A != sem_B
- **Focal Loss**: Handles class imbalance with configurable focal loss

## Supported Datasets

### SECOND-CC (Semantic Change Detection + Change Captioning)
- **Images**: 6,041 bitemporal pairs (256x256 pixels)
- **Task**: Semantic change detection (7 after-change classes) + captioning
- **Annotations**: Semantic segmentation maps (sem_a, sem_b) + 5 captions per image
- **Target**: Uses `sem_b` (after-change semantics) - what the area became
- **Classes**: background, low_vegetation, non_vegetated_ground, tree, water, building, playground

### LEVIR-MCI (Multi-level Change Interpretation)
- **Images**: 10,077 bitemporal pairs (256x256 pixels, 0.5m/pixel)
- **Task**: Semantic change detection (3 classes) + captioning
- **Annotations**: Semantic labels from `label_rgb` folder + 5 captions per image
- **Target**: Uses `label_rgb` (after-change semantics from RGB color masks)
- **Classes**: no_change (black), building (red), road (blue)
- **Source**: IEEE TGRS 2024 - Change-Agent paper

## Installation

```bash
# Create environment
conda create -n uniscc python=3.9
conda activate uniscc

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage (v3.0)

```python
from src.uniscc import UniSCC, UniSCCConfig
import torch

# Create model for SECOND-CC dataset with v3.0 dual semantic head
config = UniSCCConfig(
    dataset='second_cc',
    vocab_size=10000,
    pretrained=True,
    dual_head=True,                # v3.0: Enable dual semantic head
    use_transition_attention=True  # v3.0: Transition-aware captioning
)
model = UniSCC(config)

# Prepare inputs
img_t0 = torch.randn(2, 3, 256, 256)  # Pre-change images
img_t1 = torch.randn(2, 3, 256, 256)  # Post-change images
captions = torch.randint(0, 10000, (2, 50))  # Caption tokens
lengths = torch.tensor([30, 25])

# Training forward pass
model.train()
outputs = model(img_t0, img_t1, captions, lengths)

# v3.0 outputs
sem_A_logits = outputs['sem_A_logits']    # [B, 7, 256, 256] before-change
sem_B_logits = outputs['sem_B_logits']    # [B, 7, 256, 256] after-change
change_mask = outputs['change_mask']       # [B, 256, 256] derived mask
caption_logits = outputs['caption_logits'] # [B, 50, 10000]

# Backward compatible
cd_logits = outputs['cd_logits']  # Same as sem_B_logits

# Inference (caption generation)
model.eval()
with torch.no_grad():
    outputs = model(img_t0, img_t1)
    generated = outputs['generated_captions']  # [B, T]

# For LEVIR-MCI (3 semantic classes)
config_levir = UniSCCConfig(dataset='levir_mci', vocab_size=10000, dual_head=True)
model_levir = UniSCC(config_levir)
# sem_A/sem_B shape: [B, 3, 256, 256] - no_change, building, road
```

### Run Sanity Checks

```bash
python src/sanity_check.py
```

## Project Structure

```
UniSCC/
├── configs/
│   ├── second_cc.yaml          # SECOND-CC configuration
│   └── levir_mci.yaml          # LEVIR-MCI configuration
├── src/
│   ├── encoder.py              # Vision encoder with temporal embeddings
│   ├── tdt.py                  # Temporal Difference Transformer (v3: returns enhanced features)
│   ├── lsp.py                  # Learnable Semantic Prompts + TransitionLSP (v3)
│   ├── semantic_head.py        # DualSemanticHead (v3) + SemanticChangeHead
│   ├── caption_decoder.py      # TransitionCaptionDecoder (v3) + SemanticCaptionDecoder
│   ├── uniscc.py               # Main model integration (v3.0)
│   └── sanity_check.py         # Comprehensive test suite
├── losses/
│   ├── scd_loss.py             # Change detection losses
│   ├── caption_loss.py         # Caption generation losses
│   └── consistency_loss.py     # Multi-task consistency loss
├── data/
│   ├── second_cc.py            # SECOND-CC dataset loader
│   ├── levir_mci.py            # LEVIR-MCI dataset loader
│   ├── vocabulary.py           # Vocabulary utilities
│   └── transforms.py           # Data augmentations
├── utils/
│   └── metrics.py              # Evaluation metrics
├── train.py                    # Training script (v3.0)
├── evaluate.py                 # Evaluation script (v3.0)
├── inference.py                # Inference script (v3.0)
├── requirements.txt
└── README.md
```

## Dataset Preparation

### SECOND-CC
```
/path/to/SECOND-CC-AUG/
├── SECOND-CC-AUG.json
├── train/
│   ├── rgb/A/*.png, rgb/B/*.png
│   └── sem/A/*.png, sem/B/*.png
├── val/
└── test/
```

### LEVIR-MCI
```
/path/to/LEVIR-MCI-dataset/
├── LevirCCcaptions.json
└── images/
    ├── train/
    │   ├── A/, B/              # RGB images
    │   ├── label/              # Binary masks (not used)
    │   └── label_rgb/          # Semantic masks (used for training/evaluation)
    │                           # Colors: black=no_change, red=building, blue=road
    ├── val/
    └── test/
```

## Usage

### Training
```bash
# Train on SECOND-CC
python train.py --config configs/second_cc.yaml

# Train on LEVIR-MCI
python train.py --config configs/levir_mci.yaml

# Resume training
python train.py --config configs/levir_mci.yaml --resume checkpoints/last.pth
```

### Evaluation
```bash
# Evaluate on SECOND-CC
python evaluate.py --config configs/second_cc.yaml --checkpoint checkpoints/best.pth

# Evaluate on LEVIR-MCI
python evaluate.py --config configs/levir_mci.yaml --checkpoint checkpoints/best.pth
```

### Inference
```bash
python inference.py --config configs/levir_mci.yaml \
                    --checkpoint checkpoints/best.pth \
                    --image_a path/to/before.png \
                    --image_b path/to/after.png \
                    --output_dir outputs/
```

## Model Architecture

### Components (v3.0)

1. **Encoder**: Siamese Swin Transformer with learnable temporal embeddings
2. **TDT (Temporal Difference Transformer)**: Returns diff + enhanced features for both timestamps
3. **TransitionLSP**: Dual semantic prompts (A/B) + transition embeddings for "what changed into what"
4. **DualSemanticHead**: Predicts both sem_A (before) and sem_B (after) semantic maps
5. **TransitionCaptionDecoder**: Transformer decoder with transition-aware attention

### Data Flow (v3.0)
```
img_t0, img_t1 [B,3,H,W]
    │
    ▼
Encoder [shared weights]
    │
    ▼
feat_t0, feat_t1 [B,C,H',W']
    │
    ▼
TDT [bidirectional attention]
    │
    ▼
{diff, feat_t0_enhanced, feat_t1_enhanced}
    │
    ▼
TransitionLSP → prompts_A, prompts_B, transitions
    │
    ▼
DualSemanticHead
    ├──► sem_A_logits [B,K,H,W]  (before-change semantics)
    ├──► sem_B_logits [B,K,H,W]  (after-change semantics)
    └──► change_mask [B,H,W]     (derived: sem_A != sem_B)
    │
    ▼
TransitionCaptionDecoder
    └──► caption_logits [B,T,V]
```

## Configuration

```python
from src.uniscc import UniSCCConfig

config = UniSCCConfig(
    # Dataset
    dataset='second_cc',          # or 'levir_mci'

    # Encoder
    backbone='swin_base_patch4_window7_224',
    pretrained=True,
    feature_dim=512,

    # TDT
    tdt_layers=3,
    tdt_heads=8,

    # Change Detection
    num_semantic_classes=7,       # SECOND-CC: 7 semantic classes
    num_change_classes=3,         # LEVIR-MCI: 3 semantic classes

    # Caption Decoder
    vocab_size=10000,
    decoder_layers=6,
    max_caption_length=50,

    # v3.0: Dual Head Configuration
    dual_head=True,               # Enable dual semantic head
    share_decoder=True,           # Share decoder weights between A and B
    use_transition_attention=True,# Transition-aware captioning

    # v3.0: Loss Configuration
    use_focal_loss=True,          # Use focal loss for class imbalance
    focal_gamma=2.0,              # Focal loss gamma
    sem_A_weight=1.0,             # Weight for before-change loss
    sem_B_weight=1.0,             # Weight for after-change loss
)
```

## Metrics

### Change Detection (v3.0)
- **Before-change (sem_A)**: mIoU, F1, OA for before-change semantic prediction
- **After-change (sem_B)**: mIoU, F1, OA for after-change semantic prediction (primary metric)
- **SECOND-CC**: 7-class semantic evaluation
- **LEVIR-MCI**: 3-class semantic evaluation

### Change Captioning
- BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr

<!-- ## Citation

```bibtex
@article{liu2024change_agent,
  title={Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis},
  author={Liu, Chenyang and others},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}
``` -->

## License

MIT License
