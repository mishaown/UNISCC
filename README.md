# UniSCC: Unified Semantic Change Detection and Change Captioning

A unified framework for joint Semantic Change Detection (SCD) and Change Captioning (CC) on remote sensing imagery.

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

### Basic Usage

```python
from src.uniscc import UniSCC, UniSCCConfig
import torch

# Create model for SECOND-CC dataset (7 after-change semantic classes)
config = UniSCCConfig(
    dataset='second_cc',
    vocab_size=10000,
    pretrained=True
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
cd_logits = outputs['cd_logits']        # [B, 7, 256, 256] - 7 after-change classes
caption_logits = outputs['caption_logits']  # [B, 50, 10000]

# Inference (caption generation)
model.eval()
with torch.no_grad():
    outputs = model(img_t0, img_t1)
    generated = outputs['generated_captions']  # [B, T]

# For LEVIR-MCI (3 semantic classes)
config_levir = UniSCCConfig(dataset='levir_mci', vocab_size=10000)
model_levir = UniSCC(config_levir)
# cd_logits shape: [B, 3, 256, 256] - no_change, building, road
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
│   ├── tdt.py                  # Temporal Difference Transformer
│   ├── lsp.py                  # Learnable Semantic Prompts
│   ├── change_head.py          # Unified Change Detection Head
│   ├── caption_decoder.py      # Change-Guided Caption Decoder
│   ├── uniscc.py               # Main model integration
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
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── inference.py                # Inference script
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

### Components

1. **Encoder**: Siamese Swin Transformer with learnable temporal embeddings
2. **TDT (Temporal Difference Transformer)**: Bidirectional cross-temporal attention for capturing "what appeared" and "what disappeared"
3. **LSP (Learnable Semantic Prompts)**: CLIP-initialized prompts with learnable offsets
4. **Change Head**: Progressive upsampling decoder for change detection
5. **Caption Decoder**: Transformer decoder with change-guided attention

### Data Flow
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
diff_features [B,C,H',W']
    │
    ├──► Change Head → cd_logits [B,K,H,W]  (K=7 for SECOND-CC, K=3 for LEVIR-MCI)
    └──► Caption Decoder → caption_logits [B,T,V]
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

    # Change Detection (after-change semantics)
    num_semantic_classes=7,       # SECOND-CC: 7 after-change classes
    num_change_classes=3,         # LEVIR-MCI: 3 semantic classes

    # Caption Decoder
    vocab_size=10000,
    decoder_layers=6,
    max_caption_length=50,
)
```

## Metrics

### Change Detection
- **SECOND-CC (7-class semantic)**: mIoU, F1, OA
- **LEVIR-MCI (3-class semantic)**: mIoU, F1, OA

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
