#!/usr/bin/env python3
"""
UniSCC Evaluation Script (v3.0)

Evaluates trained models with dual semantic head support.

v3.0 Features:
- Evaluates both before-change (sem_A) and after-change (sem_B) predictions
- Uses focal loss compatible metrics
- Supports transition-aware caption evaluation

Metrics:
- SECOND-CC: SeK, mIoU, F1, OA + BLEU, METEOR, ROUGE-L, CIDEr
- LEVIR-MCI: Precision, Recall, F1, IoU, OA, Kappa + BLEU, METEOR, ROUGE-L, CIDEr

Usage:
  python evaluate.py --config configs/second_cc.yaml --checkpoint checkpoints/second_cc/best.pth
  python evaluate.py --config configs/levir_mci.yaml --checkpoint checkpoints/levir_mci/best.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import yaml
import torch
from tqdm import tqdm

from data import create_dataloaders
from src import UniSCC, UniSCCConfig, build_uniscc
from utils import MultiClassChangeMetrics, CaptionMetrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Evaluator:
    """
    UniSCC Evaluator.
    
    Supports both SECOND-CC (semantic) and LEVIR-MCI (binary) datasets.
    """
    
    def __init__(self, config: dict, checkpoint_path: str, split: str = 'test'):
        self.config = config
        self.device = torch.device(
            config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
        )
        self.split = split
        
        # Dataset type
        self.dataset_name = config['dataset']['name']
        self.is_levir = self.dataset_name == 'LEVIR-MCI'
        
        print(f"\n{'='*60}")
        print(f"UniSCC Evaluation - {self.dataset_name}")
        print(f"Split: {split}")
        print(f"{'='*60}\n")
        
        # Load data
        self._setup_data()
        
        # Load model
        self._load_model(checkpoint_path)
        
        # Initialize metrics
        # v3.0: Evaluate both sem_A (before) and sem_B (after) predictions
        # SECOND-CC: 7 semantic classes
        # LEVIR-MCI: 3 semantic classes (no_change, building, road)
        self.sem_A_metrics = MultiClassChangeMetrics(config['dataset']['num_classes'])
        self.sem_B_metrics = MultiClassChangeMetrics(config['dataset']['num_classes'])
        # Backward compat alias
        self.cd_metrics = self.sem_B_metrics

        self.caption_metrics = CaptionMetrics()
    
    def _setup_data(self):
        """Setup data loader."""
        print("Loading data...")
        train_loader, val_loader, test_loader, self.vocab = create_dataloaders(self.config)
        
        if self.split == 'test':
            self.dataloader = test_loader
        elif self.split == 'val':
            self.dataloader = val_loader
        else:
            self.dataloader = train_loader
        
        # Update vocab size
        if self.vocab:
            self.config['dataset']['vocab_size'] = len(self.vocab)
        
        print(f"  {self.split}: {len(self.dataloader.dataset)} samples")
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        print(f"\nLoading model from: {checkpoint_path}")
        
        # Determine dataset type
        dataset_type = 'levir_mci' if self.is_levir else 'second_cc'

        # Build model using config
        model_cfg = self.config.get('model', {})

        # v3.0 config with dual semantic head
        config = UniSCCConfig(
            dataset=dataset_type,
            backbone=model_cfg.get('encoder', {}).get('backbone', 'swin_base_patch4_window7_224'),
            feature_dim=model_cfg.get('tdt', {}).get('hidden_dim', 512),
            vocab_size=self.config['dataset'].get('vocab_size', 10000),
            # v3.0: Use num_semantic_classes and num_change_classes
            num_semantic_classes=7,  # SECOND-CC semantic classes
            num_change_classes=3,    # LEVIR-MCI change classes
            max_caption_length=self.config['dataset'].get('max_caption_length', 50),
            # v3.0: Enable dual head with transition prompts
            dual_head=True,
            use_transition_attention=True
        )
        self.model = UniSCC(config).to(self.device)
        
        # Load weights
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        else:
            print(f"  Warning: Checkpoint not found, using random weights")
        
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation with v3.0 dual semantic head support."""
        print(f"\nEvaluating on {self.split} set...")

        # Reset metrics
        self.sem_A_metrics.reset()
        self.sem_B_metrics.reset()
        self.caption_metrics.reset()

        all_predictions = []
        all_references = []

        for batch in tqdm(self.dataloader, desc="Evaluating"):
            rgb_a = batch['rgb_a'].to(self.device)
            rgb_b = batch['rgb_b'].to(self.device)
            raw_captions = batch['raw_captions']

            # v3.0: Get targets for both before and after
            # SECOND-CC: sem_a (before), sem_b (after)
            # LEVIR-MCI: label for both (semantic change type)
            if self.is_levir:
                targets_A = batch['label']
                targets_B = batch['label']
            else:
                targets_A = batch['sem_a']
                targets_B = batch['sem_b']

            # Forward pass
            outputs = self.model(rgb_a, rgb_b)

            # v3.0: Get both sem_A and sem_B predictions
            sem_A_logits = outputs.get('sem_A_logits')
            sem_B_logits = outputs.get('sem_B_logits', outputs['cd_logits'])

            # Update metrics for sem_B (after-change, primary evaluation)
            sem_B_preds = sem_B_logits.argmax(dim=1)
            self.sem_B_metrics.update(sem_B_preds.cpu(), targets_B)

            # Update metrics for sem_A (before-change) if available
            if sem_A_logits is not None:
                sem_A_preds = sem_A_logits.argmax(dim=1)
                self.sem_A_metrics.update(sem_A_preds.cpu(), targets_A)

            # Generate captions
            generated_captions = outputs.get('generated_captions')
            if generated_captions is not None:
                for i in range(generated_captions.shape[0]):
                    pred_caption = self.vocab.decode(generated_captions[i]) if self.vocab else ""
                    all_predictions.append(pred_caption)
                    all_references.append(raw_captions[i])

        # Update caption metrics
        if all_predictions:
            self.caption_metrics.update(all_predictions, all_references)

        # Compute all metrics
        sem_A_results = self.sem_A_metrics.compute()
        sem_B_results = self.sem_B_metrics.compute()
        caption_results = self.caption_metrics.compute()

        results = {
            'sem_A': sem_A_results,  # Before-change semantic metrics
            'sem_B': sem_B_results,  # After-change semantic metrics (primary)
            'cd': sem_B_results,     # Backward compatibility alias
            'caption': caption_results
        }

        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted results with v3.0 dual semantic metrics."""
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS - {self.dataset_name} (v3.0)")
        print("=" * 60)

        # v3.0: Print both sem_A and sem_B metrics
        if self.is_levir:
            class_info = "3-class: no_change, building, road"
        else:
            class_info = "7-class semantic"

        # Before-change (sem_A) metrics
        if 'sem_A' in results:
            print(f"\nBefore-Change Semantics ({class_info}):")
            print("-" * 40)
            for metric, value in results['sem_A'].items():
                print(f"  {metric:15s}: {value:.4f}")

        # After-change (sem_B) metrics - primary evaluation
        print(f"\nAfter-Change Semantics ({class_info}) [Primary]:")
        print("-" * 40)
        for metric, value in results['sem_B'].items():
            print(f"  {metric:15s}: {value:.4f}")

        # Caption metrics
        print("\nChange Captioning Metrics:")
        print("-" * 40)
        for metric, value in results['caption'].items():
            print(f"  {metric:15s}: {value:.4f}")

        print("\n" + "=" * 60)
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file with v3.0 metrics."""
        # Convert to serializable format
        results_json = {
            'dataset': self.dataset_name,
            'split': self.split,
            'version': 'v3.0',
            # v3.0: Include both sem_A and sem_B metrics
            'sem_A_metrics': {k: float(v) for k, v in results.get('sem_A', {}).items()},
            'sem_B_metrics': {k: float(v) for k, v in results['sem_B'].items()},
            'cd_metrics': {k: float(v) for k, v in results['cd'].items()},  # Backward compat
            'caption_metrics': {k: float(v) for k, v in results['caption'].items()}
        }

        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='UniSCC Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint, args.split)
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    if args.output:
        evaluator.save_results(results, args.output)
    else:
        output_path = f"eval_results_{config['dataset']['name'].lower()}_{args.split}.json"
        evaluator.save_results(results, output_path)


if __name__ == '__main__':
    main()
