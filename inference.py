#!/usr/bin/env python3
"""
UniSCC Inference Script (v5.0)

Run inference on single image pairs or directories with multi-scale architecture.

v5.0 Features:
- Multi-scale hierarchical architecture with FPN
- Hierarchical feature alignment across scales
- Multi-task change detection head with magnitude estimation
- Multi-level caption decoder

Outputs:
- change_map.npy: Semantic change map
- caption.txt: Generated change caption
- viz.png: Visualization with all outputs

Usage:
    # Single image pair
    python inference.py --config configs/second_cc.yaml \
                        --checkpoint checkpoints/best.pth \
                        --image_a path/to/before.png \
                        --image_b path/to/after.png

    # Directory with separate A/B folders (SECOND-CC structure)
    python inference.py --config configs/second_cc.yaml \
                        --checkpoint checkpoints/best.pth \
                        --dir_a E:/CD-Experiment/Datasets/SECOND-CC-AUG/test/rgb/A \
                        --dir_b E:/CD-Experiment/Datasets/SECOND-CC-AUG/test/rgb/B \
                        --output_dir outputs/

    # Directory with separate A/B folders (LEVIR-MCI structure)
    python inference.py --config configs/levir_mci.yaml \
                        --checkpoint checkpoints/best.pth \
                        --dir_a E:/CD-Experiment/Datasets/LEVIR-MCI-dataset/images/test/A \
                        --dir_b E:/CD-Experiment/Datasets/LEVIR-MCI-dataset/images/test/B \
                        --output_dir outputs/
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data.transforms import PairedTransform, NORMALIZE_STATS
from data.vocabulary import Vocabulary
from src import UniSCC, UniSCCConfig, build_uniscc


# Color maps
SECOND_CC_COLORS = {
    0: (0, 0, 0), 1: (144, 238, 144), 2: (139, 119, 101),
    3: (34, 139, 34), 4: (0, 0, 255), 5: (255, 0, 0), 6: (255, 0, 255)
}
SECOND_CC_NAMES = ['No Change', 'Low Vegetation', 'Ground', 'Tree', 'Water', 'Building', 'Playground']

LEVIR_MCI_COLORS = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
LEVIR_MCI_NAMES = ['No Change', 'Building', 'Road']


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Inferencer:
    """UniSCC Inference Engine v4.0."""

    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.device = torch.device(
            config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
        )

        # Dataset info
        self.dataset_name = config['dataset']['name']
        self.is_levir = self.dataset_name == 'LEVIR-MCI'
        self.num_classes = config['dataset']['num_classes']
        self.image_size = config['dataset']['image_size']

        # Setup transform
        normalize_type = 'levir_mci' if self.dataset_name == 'LEVIR-MCI' else 'second_cc'
        self.transform = PairedTransform(
            image_size=self.image_size,
            is_train=False,
            normalize_type=normalize_type
        )

        # Load vocabulary
        self._load_vocab()

        # Load model
        self._load_model(checkpoint_path)

        # Visualization settings
        if self.is_levir:
            self.colors = LEVIR_MCI_COLORS
            self.class_names = self.config['dataset'].get('class_names', LEVIR_MCI_NAMES)
        else:
            self.colors = SECOND_CC_COLORS
            self.class_names = SECOND_CC_NAMES

    def _load_vocab(self):
        """Load or create vocabulary."""
        vocab_path = self.config.get('vocab_path')
        if vocab_path and os.path.exists(vocab_path):
            self.vocab = Vocabulary.load(vocab_path)
        else:
            # Create minimal vocab
            self.vocab = Vocabulary(min_word_freq=1)
            common_words = [
                'the', 'a', 'an', 'in', 'on', 'is', 'are', 'was', 'no', 'change',
                'building', 'buildings', 'new', 'built', 'constructed', 'demolished',
                'vegetation', 'tree', 'trees', 'grass', 'green', 'removed', 'added',
                'water', 'road', 'roads', 'ground', 'scene', 'area', 'region'
            ]
            self.vocab.build_vocab([[w] for w in common_words])

    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint with v5.0 configuration."""
        print(f"\n{'='*60}")
        print(f"UniSCC v5.0 Inference - {self.dataset_name}")
        print(f"{'='*60}\n")

        # Determine dataset type
        dataset_type = 'levir_mci' if self.is_levir else 'second_cc'

        # Load checkpoint first to extract vocab_size
        vocab_size = len(self.vocab)  # Default fallback
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Extract vocab_size from checkpoint weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Check caption decoder embedding layer shape
                embed_key = 'caption_decoder.token_embed.weight'
                if embed_key in state_dict:
                    vocab_size = state_dict[embed_key].shape[0]
                    print(f"Detected vocab_size from checkpoint: {vocab_size}")
        else:
            checkpoint = None
            print(f"Warning: Checkpoint not found, using random weights")

        # Build model using v5.0 config with multi-scale architecture
        model_cfg = self.config.get('model', {})
        config = UniSCCConfig(
            dataset=dataset_type,
            backbone=model_cfg.get('encoder', {}).get('backbone', 'swin_base_patch4_window7_224'),
            feature_dim=model_cfg.get('tdt', {}).get('hidden_dim', 512),
            vocab_size=vocab_size,
            num_semantic_classes=self.num_classes if not self.is_levir else 7,
            num_change_classes=self.num_classes if self.is_levir else 3,
            max_caption_length=self.config['dataset'].get('max_caption_length', 50),
            # v4.0: Alignment configuration
            use_alignment=model_cfg.get('use_alignment', True),
            alignment_type=model_cfg.get('alignment_type', 'cross_attention'),
            alignment_heads=model_cfg.get('alignment_heads', 8),
        )
        self.model = UniSCC(config).to(self.device)

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint: {checkpoint_path}")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

        self.model.eval()
        print(f"  Alignment: {config.alignment_type if config.use_alignment else 'Disabled'}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Vocab size: {vocab_size}")

    @torch.no_grad()
    def predict(self, image_a: Image.Image, image_b: Image.Image) -> Dict[str, Any]:
        """Run inference on image pair with v5.0 multi-scale architecture."""
        # Transform images
        rgb_a, rgb_b, _ = self.transform(image_a, image_b, None)
        rgb_a = rgb_a.unsqueeze(0).to(self.device)
        rgb_b = rgb_b.unsqueeze(0).to(self.device)

        # Forward pass
        outputs = self.model(rgb_a, rgb_b)

        # Get semantic change predictions
        cd_logits = outputs['cd_logits']
        cd_probs = F.softmax(cd_logits, dim=1)
        change_map = cd_logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Get alignment confidence if available
        alignment_confidence = outputs.get('alignment_confidence')
        if alignment_confidence is not None:
            alignment_confidence = alignment_confidence.squeeze(0).cpu().numpy()

        # Caption from generated tokens
        generated_captions = outputs.get('generated_captions')
        if generated_captions is not None:
            caption = self.vocab.decode(generated_captions[0])
        else:
            # Fallback to caption_logits if available
            caption_logits = outputs.get('caption_logits')
            if caption_logits is not None:
                caption_tokens = caption_logits.argmax(dim=-1).squeeze(0)
                caption = self.vocab.decode(caption_tokens)
            else:
                caption = "No caption generated"

        return {
            'change_map': change_map,
            'change_probs': cd_probs.squeeze(0).cpu().numpy(),
            'caption': caption,
            'alignment_confidence': alignment_confidence,
        }

    def colorize_map(self, change_map: np.ndarray) -> np.ndarray:
        """Convert change map to RGB.

        Both datasets output semantic class predictions directly:
        - SECOND-CC: 7 after-change semantic classes
        - LEVIR-MCI: 3 semantic classes (no_change, building, road)
        """
        H, W = change_map.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        for class_id, color in self.colors.items():
            rgb[change_map == class_id] = color

        return rgb

    def visualize(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create visualization for change detection results."""
        has_alignment = results.get('alignment_confidence') is not None

        if has_alignment:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Before image
        axes[0].imshow(image_a.resize((self.image_size, self.image_size)))
        axes[0].set_title('Before Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # After image
        axes[1].imshow(image_b.resize((self.image_size, self.image_size)))
        axes[1].set_title('After Image', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Change map
        change_rgb = self.colorize_map(results['change_map'])
        axes[2].imshow(change_rgb)
        axes[2].set_title('Semantic Change Map', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # Add legend
        patches = [mpatches.Patch(color=np.array(c)/255, label=n)
                   for n, c in zip(self.class_names, self.colors.values())]
        axes[2].legend(handles=patches, loc='upper right', fontsize=8)

        # Caption (or alignment confidence)
        if has_alignment:
            # Show alignment confidence as heatmap
            axes[3].text(0.5, 0.7, f'"{results["caption"]}"',
                        ha='center', va='center', fontsize=11, wrap=True,
                        transform=axes[3].transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[3].set_title('Generated Caption', fontsize=12, fontweight='bold')
            axes[3].axis('off')
        else:
            axes[3].text(0.5, 0.5, f'"{results["caption"]}"',
                        ha='center', va='center', fontsize=12, wrap=True,
                        transform=axes[3].transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[3].set_title('Generated Caption', fontsize=12, fontweight='bold')
            axes[3].axis('off')

        plt.suptitle('UniSCC v5.0 - Multi-Scale Architecture', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved: {save_path}")

        return fig

    def process_pair(
        self,
        image_a_path: str,
        image_b_path: str,
        output_dir: str,
        output_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a single image pair and save outputs."""
        # Load images
        image_a = Image.open(image_a_path).convert('RGB')
        image_b = Image.open(image_b_path).convert('RGB')

        # Predict
        results = self.predict(image_a, image_b)

        # Output name
        if output_name is None:
            output_name = Path(image_a_path).stem

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save change map
        np.save(output_dir / f'{output_name}_change_map.npy', results['change_map'])

        # Save caption
        with open(output_dir / f'{output_name}_caption.txt', 'w') as f:
            f.write(results['caption'])

        # Save visualization
        self.visualize(image_a, image_b, results,
                      save_path=str(output_dir / f'{output_name}_viz.png'))

        return results

    def process_directory_pair(
        self,
        dir_a: str,
        dir_b: str,
        output_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Process all image pairs from separate A and B directories.

        This handles the standard bitemporal dataset structure:
        - SECOND-CC: root/split/rgb/A/*.png and root/split/rgb/B/*.png
        - LEVIR-MCI: root/images/split/A/*.png and root/images/split/B/*.png

        Args:
            dir_a: Directory containing before (t0) images
            dir_b: Directory containing after (t1) images
            output_dir: Output directory for results
        """
        dir_a = Path(dir_a)
        dir_b = Path(dir_b)
        results_list = []

        # Find all images in dir_a
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        images_a = []
        for ext in image_extensions:
            images_a.extend(dir_a.glob(ext))
        images_a = sorted(images_a)

        if not images_a:
            print(f"Warning: No images found in {dir_a}")
            return results_list

        print(f"Found {len(images_a)} images in {dir_a}")

        for img_a_path in images_a:
            # Look for matching image in dir_b with same filename
            img_b_path = dir_b / img_a_path.name

            if not img_b_path.exists():
                print(f"Warning: No matching image in B for {img_a_path.name}")
                continue

            name = img_a_path.stem
            print(f"Processing: {name}")

            results = self.process_pair(str(img_a_path), str(img_b_path), output_dir, name)
            results_list.append(results)

        print(f"\nProcessed {len(results_list)} image pairs")
        return results_list

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern_a: str = '*_A.png',
        pattern_b: str = '*_B.png'
    ) -> List[Dict[str, Any]]:
        """[Deprecated] Process image pairs using filename patterns."""
        input_dir = Path(input_dir)
        results_list = []

        images_a = sorted(input_dir.glob(pattern_a))

        for img_a_path in images_a:
            img_b_path = input_dir / img_a_path.name.replace('_A', '_B')

            if not img_b_path.exists():
                print(f"Warning: No matching B image for {img_a_path}")
                continue

            name = img_a_path.stem.replace('_A', '')
            print(f"Processing: {name}")

            results = self.process_pair(str(img_a_path), str(img_b_path), output_dir, name)
            results_list.append(results)

        print(f"\nProcessed {len(results_list)} image pairs")
        return results_list


def main():
    parser = argparse.ArgumentParser(description='UniSCC v5.0 Inference')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)

    # Single image pair
    parser.add_argument('--image_a', type=str, default=None,
                       help='Path to before image (t0)')
    parser.add_argument('--image_b', type=str, default=None,
                       help='Path to after image (t1)')

    # Directory-based input (folder structure with A/ and B/ subdirs)
    parser.add_argument('--dir_a', type=str, default=None,
                       help='Directory containing before images (e.g., test/rgb/A for SECOND-CC)')
    parser.add_argument('--dir_b', type=str, default=None,
                       help='Directory containing after images (e.g., test/rgb/B for SECOND-CC)')

    # Legacy: pattern-based matching (deprecated)
    parser.add_argument('--input_dir', type=str, default=None,
                       help='[Deprecated] Use --dir_a and --dir_b instead')

    parser.add_argument('--output_dir', type=str, default='./outputs')
    args = parser.parse_args()

    # Validate input
    has_single_pair = args.image_a is not None
    has_dir_pair = args.dir_a is not None and args.dir_b is not None
    has_legacy = args.input_dir is not None

    if not (has_single_pair or has_dir_pair or has_legacy):
        parser.error("Provide either --image_a/--image_b OR --dir_a/--dir_b")

    if has_single_pair and not args.image_b:
        parser.error("--image_b required with --image_a")

    if (args.dir_a is None) != (args.dir_b is None):
        parser.error("Both --dir_a and --dir_b must be provided together")

    # Load config
    config = load_config(args.config)

    # Create inferencer
    inferencer = Inferencer(config, args.checkpoint)

    # Run inference
    if has_dir_pair:
        inferencer.process_directory_pair(args.dir_a, args.dir_b, args.output_dir)
    elif has_legacy:
        print("Warning: --input_dir is deprecated. Use --dir_a and --dir_b instead.")
        inferencer.process_directory(args.input_dir, args.output_dir)
    else:
        results = inferencer.process_pair(args.image_a, args.image_b, args.output_dir)
        print(f"\n=== v5.0 Results ===")
        print(f"Caption: {results['caption']}")
        print(f"Change map shape: {results['change_map'].shape}")
        if results.get('alignment_confidence') is not None:
            print(f"Alignment confidence: mean={results['alignment_confidence'].mean():.3f}")


if __name__ == '__main__':
    main()
