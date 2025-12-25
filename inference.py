#!/usr/bin/env python3
"""
UniSCC Inference Script

Run inference on single image pairs or directories.

Usage:
    python inference.py --config configs/levir_mci.yaml \\
                        --checkpoint checkpoints/best.pth \\
                        --image_a path/to/before.png \\
                        --image_b path/to/after.png

    python inference.py --config configs/second_cc.yaml \\
                        --checkpoint checkpoints/best.pth \\
                        --input_dir path/to/images/ \\
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

LEVIR_MCI_COLORS = {0: (0, 0, 0), 1: (255, 0, 0), 2: (255, 255, 0)}
LEVIR_MCI_NAMES = ['No Change', 'Building', 'Road']


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Inferencer:
    """UniSCC Inference Engine."""
    
    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.device = torch.device(
            config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
        )
        
        # Dataset info
        self.dataset_name = config['dataset']['name']
        self.is_levir = self.dataset_name == 'LEVIR-MCI'
        self.is_binary = config['dataset']['num_classes'] == 2
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
        """Load model from checkpoint."""
        # Determine dataset type
        dataset_type = 'levir_mci' if self.is_levir else 'second_cc'
        
        # Build model using config
        model_cfg = self.config.get('model', {})
        dataset_num_classes = self.config['dataset']['num_classes']
        config = UniSCCConfig(
            dataset=dataset_type,
            backbone=model_cfg.get('encoder', {}).get('backbone', 'swin_base_patch4_window7_224'),
            feature_dim=model_cfg.get('tdt', {}).get('hidden_dim', 512),
            vocab_size=len(self.vocab),
            scd_classes=dataset_num_classes if not self.is_levir else 7,
            bcd_classes=dataset_num_classes if self.is_levir else 2,
            max_caption_length=self.config['dataset'].get('max_caption_length', 50)
        )
        self.model = UniSCC(config).to(self.device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found, using random weights")
        
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, image_a: Image.Image, image_b: Image.Image) -> Dict[str, Any]:
        """Run inference on image pair."""
        # Transform images
        rgb_a, rgb_b, _ = self.transform(image_a, image_b, None)
        rgb_a = rgb_a.unsqueeze(0).to(self.device)
        rgb_b = rgb_b.unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(rgb_a, rgb_b)
        
        # CD predictions
        cd_logits = outputs['cd_logits']
        cd_probs = F.softmax(cd_logits, dim=1)
        cd_pred = cd_logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Caption
        caption_logits = outputs.get('caption_logits')
        if caption_logits is not None:
            caption_tokens = caption_logits.argmax(dim=-1).squeeze(0)
            caption = self.vocab.decode(caption_tokens)
        else:
            caption = "No caption generated"
        
        return {
            'change_map': cd_pred,
            'change_probs': cd_probs.squeeze(0).cpu().numpy(),
            'caption': caption
        }
    
    def colorize_map(self, change_map: np.ndarray) -> np.ndarray:
        """Convert change map to RGB."""
        H, W = change_map.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        
        if self.is_levir:
            for class_id, color in self.colors.items():
                rgb[change_map == class_id] = color
        else:
            # For semantic, decode transition
            to_class = change_map % self.num_classes
            for class_id, color in self.colors.items():
                rgb[to_class == class_id] = color
        
        return rgb
    
    def visualize(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create visualization."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Before
        axes[0].imshow(image_a.resize((self.image_size, self.image_size)))
        axes[0].set_title('Before', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # After
        axes[1].imshow(image_b.resize((self.image_size, self.image_size)))
        axes[1].set_title('After', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Change map
        change_rgb = self.colorize_map(results['change_map'])
        axes[2].imshow(change_rgb)
        axes[2].set_title('Change Map', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Add legend
        patches = [mpatches.Patch(color=np.array(c)/255, label=n) 
                   for n, c in zip(self.class_names, self.colors.values())]
        axes[2].legend(handles=patches, loc='upper right', fontsize=8)
        
        # Caption
        axes[3].text(0.5, 0.5, f'"{results["caption"]}"',
                    ha='center', va='center', fontsize=12, wrap=True,
                    transform=axes[3].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[3].set_title('Generated Caption', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
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
        """Process a single image pair."""
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
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern_a: str = '*_A.png',
        pattern_b: str = '*_B.png'
    ) -> List[Dict[str, Any]]:
        """Process all image pairs in directory."""
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
    parser = argparse.ArgumentParser(description='UniSCC Inference')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image_a', type=str, default=None)
    parser.add_argument('--image_b', type=str, default=None)
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    args = parser.parse_args()
    
    # Validate input
    if args.image_a is None and args.input_dir is None:
        parser.error("Either --image_a/--image_b or --input_dir required")
    
    if args.image_a and not args.image_b:
        parser.error("--image_b required with --image_a")
    
    # Load config
    config = load_config(args.config)
    
    # Create inferencer
    inferencer = Inferencer(config, args.checkpoint)
    
    # Run inference
    if args.input_dir:
        inferencer.process_directory(args.input_dir, args.output_dir)
    else:
        results = inferencer.process_pair(args.image_a, args.image_b, args.output_dir)
        print(f"\nCaption: {results['caption']}")
        print(f"Change map shape: {results['change_map'].shape}")


if __name__ == '__main__':
    main()
