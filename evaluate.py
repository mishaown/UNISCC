#!/usr/bin/env python3
"""
UniSCC Evaluation Script

Evaluates trained models on:
- SECOND-CC: SeK, mIoU, F1, OA + BLEU, METEOR, ROUGE-L, CIDEr
- LEVIR-MCI: Precision, Recall, F1, IoU, OA, Kappa + BLEU, METEOR, ROUGE-L, CIDEr

Usage:
    python evaluate.py --config configs/second_cc.yaml --checkpoint checkpoints/best.pth
    python evaluate.py --config configs/levir_mci.yaml --checkpoint checkpoints/best.pth
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
from utils import BinaryChangeMetrics, SemanticChangeMetrics, MultiClassChangeMetrics, CaptionMetrics


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
        self.is_binary = config['dataset']['num_classes'] == 2
        
        print(f"\n{'='*60}")
        print(f"UniSCC Evaluation - {self.dataset_name}")
        print(f"Split: {split}")
        print(f"{'='*60}\n")
        
        # Load data
        self._setup_data()
        
        # Load model
        self._load_model(checkpoint_path)
        
        # Initialize metrics
        # SECOND-CC uses SemanticChangeMetrics with semantic_classes (internally computes transitions)
        # LEVIR-MCI uses MultiClassChangeMetrics with direct num_classes
        if self.is_levir:
            if self.is_binary:
                self.cd_metrics = BinaryChangeMetrics()
            else:
                self.cd_metrics = MultiClassChangeMetrics(config['dataset']['num_classes'])
        else:
            # SemanticChangeMetrics takes semantic classes and computes transitions internally
            self.cd_metrics = SemanticChangeMetrics(config['dataset']['num_classes'])
        
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
        dataset_num_classes = self.config['dataset']['num_classes']
        config = UniSCCConfig(
            dataset=dataset_type,
            backbone=model_cfg.get('encoder', {}).get('backbone', 'swin_base_patch4_window7_224'),
            feature_dim=model_cfg.get('tdt', {}).get('hidden_dim', 512),
            vocab_size=self.config['dataset'].get('vocab_size', 10000),
            scd_classes=dataset_num_classes if not self.is_levir else 7,
            bcd_classes=dataset_num_classes if self.is_levir else 2,
            max_caption_length=self.config['dataset'].get('max_caption_length', 50)
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
        """Run evaluation."""
        print(f"\nEvaluating on {self.split} set...")
        
        # Reset metrics
        self.cd_metrics.reset()
        self.caption_metrics.reset()
        
        all_predictions = []
        all_references = []
        
        for batch in tqdm(self.dataloader, desc="Evaluating"):
            rgb_a = batch['rgb_a'].to(self.device)
            rgb_b = batch['rgb_b'].to(self.device)
            raw_captions = batch['raw_captions']
            
            # Get targets
            if self.is_levir:
                targets = batch['label']
            else:
                targets = batch['change_map']
            
            # Forward pass
            outputs = self.model(rgb_a, rgb_b)
            
            # CD predictions
            cd_logits = outputs['cd_logits']
            cd_preds = cd_logits.argmax(dim=1)
            
            # Update CD metrics
            self.cd_metrics.update(cd_preds.cpu(), targets)
            
            # Generate captions
            caption_logits = outputs.get('caption_logits')
            if caption_logits is not None:
                caption_preds = caption_logits.argmax(dim=-1)
                
                for i in range(caption_preds.shape[0]):
                    pred_caption = self.vocab.decode(caption_preds[i]) if self.vocab else ""
                    all_predictions.append(pred_caption)
                    all_references.append(raw_captions[i])
        
        # Update caption metrics
        if all_predictions:
            self.caption_metrics.update(all_predictions, all_references)
        
        # Compute all metrics
        cd_results = self.cd_metrics.compute()
        caption_results = self.caption_metrics.compute()
        
        results = {
            'cd': cd_results,
            'caption': caption_results
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted results."""
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS - {self.dataset_name}")
        print("=" * 60)
        
        # CD metrics
        if self.is_levir:
            metric_name = "Binary Change Detection" if self.is_binary else "Multi-class Change Detection"
        else:
            metric_name = "Semantic Change Detection"
        print(f"\n📊 {metric_name} Metrics:")
        print("-" * 40)
        for metric, value in results['cd'].items():
            print(f"  {metric:15s}: {value:.4f}")
        
        # Caption metrics
        print("\n📝 Change Captioning Metrics:")
        print("-" * 40)
        for metric, value in results['caption'].items():
            print(f"  {metric:15s}: {value:.4f}")
        
        print("\n" + "=" * 60)
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file."""
        # Convert to serializable format
        results_json = {
            'dataset': self.dataset_name,
            'split': self.split,
            'cd_metrics': {k: float(v) for k, v in results['cd'].items()},
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
