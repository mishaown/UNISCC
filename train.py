#!/usr/bin/env python3
"""
UniSCC Training Script - Shared Semantic Space Architecture

Unified training for:
- SECOND-CC: Semantic Change Detection (7 after-change classes) + Captioning
- LEVIR-MCI: Semantic Change Detection (3 classes) + Captioning

Usage:
    python train.py --config configs/second_cc.yaml
    python train.py --config configs/levir_mci.yaml
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import create_dataloaders
from src import UniSCC, UniSCCConfig, build_uniscc
from losses import CaptionLoss
from utils import MultiClassChangeMetrics, CaptionMetrics


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class SemanticCDLoss(nn.Module):
    """
    Loss for Semantic Change Detection with Shared Semantic Space.

    Unified loss for both datasets - predicts what changed:
    - SECOND-CC: Uses sem_b (what the area became after change) as target
    - LEVIR-MCI: Uses label (change type) as target
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.

        Args:
            logits: [B, K, H, W] prediction logits
            targets: [B, H, W] ground truth labels

        Returns:
            total_loss, loss_dict
        """
        loss = self.ce_loss(logits, targets)
        loss_dict = {'cd': loss.item()}
        return loss, loss_dict


class Trainer:
    """
    UniSCC Trainer with Shared Semantic Space.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(self, config: dict, resume_path: Optional[str] = None):
        self.config = config
        self.device = torch.device(
            config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
        )

        # Determine dataset type
        self.dataset_name = config['dataset']['name']
        self.is_levir = self.dataset_name == 'LEVIR-MCI'
        self.num_classes = config['dataset']['num_classes']

        print(f"\n{'='*60}")
        print(f"UniSCC Training - {self.dataset_name}")
        print(f"Device: {self.device}")
        print(f"Architecture: Shared Semantic Space")
        if self.is_levir:
            print(f"Task: {self.num_classes}-class Semantic Change Detection + Captioning")
        else:
            print(f"Task: {self.num_classes}-class After-Change Semantic Detection + Captioning")
        print(f"{'='*60}\n")

        # Setup components
        self._setup_data()
        self._setup_model()
        self._setup_losses()
        self._setup_optimizer()
        self._setup_logging()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # Resume if specified
        if resume_path:
            self._load_checkpoint(resume_path)

    def _setup_data(self):
        """Setup dataloaders."""
        print("Loading data...")
        self.train_loader, self.val_loader, self.test_loader, self.vocab = create_dataloaders(self.config)

        # Update vocab size in config
        if self.vocab:
            self.config['dataset']['vocab_size'] = len(self.vocab)

        print(f"  Train: {len(self.train_loader.dataset)} samples")
        print(f"  Val: {len(self.val_loader.dataset)} samples")
        print(f"  Vocab: {len(self.vocab) if self.vocab else 'N/A'} words")

    def _setup_model(self):
        """Setup model."""
        print("\nBuilding model...")

        # Determine dataset type for config
        dataset_type = 'levir_mci' if self.is_levir else 'second_cc'

        model_config = self.config.get('model', {})
        config = UniSCCConfig(
            dataset=dataset_type,
            backbone=model_config.get('encoder', {}).get('backbone', 'swin_base_patch4_window7_224'),
            feature_dim=model_config.get('tdt', {}).get('hidden_dim', 512),
            vocab_size=self.config['dataset'].get('vocab_size', 10000),
            num_semantic_classes=self.num_classes if not self.is_levir else 7,
            num_change_classes=self.num_classes if self.is_levir else 3,
            max_caption_length=self.config['dataset'].get('max_caption_length', 50)
        )
        self.model = UniSCC(config).to(self.device)

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Parameters: {num_params:,}")
        print(f"  Trainable: {num_trainable:,}")

    def _setup_losses(self):
        """Setup loss functions."""
        loss_config = self.config['loss']

        print(f"  CD classes: {self.num_classes}")

        # Change detection loss
        self.cd_loss = SemanticCDLoss(
            num_classes=self.num_classes,
            ignore_index=loss_config.get('scd_loss', {}).get('ignore_index', 255),
            label_smoothing=0.1
        ).to(self.device)

        # Caption loss
        cap_config = loss_config.get('caption_loss', {})
        self.caption_loss = CaptionLoss(
            vocab_size=self.config['dataset'].get('vocab_size', 10000),
            label_smoothing=cap_config.get('label_smoothing', 0.1),
            ignore_index=cap_config.get('ignore_index', 0)
        ).to(self.device)

        # Loss weights
        self.loss_weights = {
            'cd': loss_config.get('scd_weight', 1.0),
            'caption': loss_config.get('caption_weight', 1.0),
        }

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        opt_config = self.config['training']['optimizer']

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_config.get('lr', 1e-4),
            weight_decay=opt_config.get('weight_decay', 0.01)
        )

        sched_config = self.config['training'].get('scheduler', {})
        if sched_config.get('type') == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config.get('T_0', 5),
                T_mult=sched_config.get('T_mult', 2),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = None

        # AMP
        self.use_amp = self.config['training'].get('amp', {}).get('enabled', True)
        self.scaler = GradScaler() if self.use_amp else None

    def _setup_logging(self):
        """Setup logging."""
        log_config = self.config.get('logging', {})
        log_dir = log_config.get('log_dir', './logs')

        Path(log_dir).mkdir(parents=True, exist_ok=True)

        if log_config.get('use_tensorboard', True):
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        # Checkpoint directory
        ckpt_dir = self.config.get('checkpoint', {}).get('save_dir', './checkpoints')
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(ckpt_dir)

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        captions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss."""
        loss_dict = {}

        # Change detection loss - unified for both datasets
        cd_logits = outputs['cd_logits']
        cd_loss, cd_loss_dict = self.cd_loss(cd_logits, targets)
        loss_dict.update(cd_loss_dict)

        # Caption loss
        cap_logits = outputs.get('caption_logits')
        if cap_logits is not None:
            cap_targets = captions[:, 1:]
            cap_preds = cap_logits[:, :-1]
            cap = self.caption_loss(cap_preds, cap_targets)
            loss_dict['caption'] = cap.item()
        else:
            cap = torch.tensor(0.0, device=self.device)
            loss_dict['caption'] = 0.0

        # Total loss
        total = (
            self.loss_weights['cd'] * cd_loss +
            self.loss_weights['caption'] * cap
        )
        loss_dict['total'] = total.item()

        return total, loss_dict

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {'total': 0, 'cd': 0, 'caption': 0}
        if not self.is_levir:
            epoch_losses['sem_a'] = 0
            epoch_losses['sem_b'] = 0

        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rgb_a = batch['rgb_a'].to(self.device)
            rgb_b = batch['rgb_b'].to(self.device)

            # Get targets - unified approach
            # SECOND-CC: use sem_b (what the area became after change)
            # LEVIR-MCI: use label (change type)
            if self.is_levir:
                targets = batch['label'].to(self.device)
            else:
                targets = batch['sem_b'].to(self.device)

            captions = batch['captions'][:, 0].to(self.device)
            lengths = batch['caption_lengths'][:, 0].to(self.device)

            # Forward
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(rgb_a, rgb_b, captions, lengths)
                    loss, loss_dict = self._compute_loss(outputs, targets, captions)

                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config['training'].get('gradient_clip'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(rgb_a, rgb_b, captions, lengths)
                loss, loss_dict = self._compute_loss(outputs, targets, captions)

                loss.backward()

                if self.config['training'].get('gradient_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.optimizer.step()

            # Update metrics
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'cd': f"{loss_dict.get('cd', 0):.4f}",
                'cap': f"{loss_dict.get('caption', 0):.4f}"
            })

            # Log to tensorboard
            if self.writer and batch_idx % 50 == 0:
                self.writer.add_scalar('train/loss', loss_dict['total'], self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        val_losses = {'total': 0, 'cd': 0, 'caption': 0}

        # Metrics - unified for both datasets
        cd_metrics = MultiClassChangeMetrics(self.num_classes)

        for batch in tqdm(self.val_loader, desc="Validating"):
            rgb_a = batch['rgb_a'].to(self.device)
            rgb_b = batch['rgb_b'].to(self.device)

            # Get targets - unified approach
            if self.is_levir:
                targets = batch['label'].to(self.device)
            else:
                targets = batch['sem_b'].to(self.device)

            captions = batch['captions'][:, 0].to(self.device)
            lengths = batch['caption_lengths'][:, 0].to(self.device)

            # Use force_teacher_forcing=True to compute caption loss during validation
            outputs = self.model(rgb_a, rgb_b, captions, lengths, force_teacher_forcing=True)
            _, loss_dict = self._compute_loss(outputs, targets, captions)

            for k, v in loss_dict.items():
                val_losses[k] = val_losses.get(k, 0) + v

            # Update CD metrics - unified for both datasets
            cd_preds = outputs['cd_logits'].argmax(dim=1)
            cd_metrics.update(cd_preds.cpu(), targets.cpu())

        # Average losses
        for k in val_losses:
            val_losses[k] /= len(self.val_loader)

        # Compute metrics - unified for both datasets
        metrics = cd_metrics.compute()
        val_losses.update(metrics)

        return val_losses

    def _save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        ckpt = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        if self.scheduler:
            ckpt['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save last
        torch.save(ckpt, self.ckpt_dir / 'last.pth')

        # Save best
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')

        # Save periodic
        save_every = self.config.get('checkpoint', {}).get('save_every', 5)
        if (self.epoch + 1) % save_every == 0:
            torch.save(ckpt, self.ckpt_dir / f'epoch_{self.epoch+1}.pth')

    def _load_checkpoint(self, path: str):
        """Load checkpoint."""
        print(f"\nLoading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.best_metric = ckpt.get('best_metric', 0.0)

        if self.scheduler and 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        print(f"  Resumed from epoch {self.epoch}")

    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']

        # Monitor metric
        monitor = 'mIoU'

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Monitor metric: {monitor}\n")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Print results
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_metrics['total']:.4f}")
            print(f"  Val mIoU: {val_metrics.get('mIoU', 0):.4f}")
            print(f"  Val F1: {val_metrics.get('F1', 0):.4f}")
            print(f"  Val OA: {val_metrics.get('OA', 0):.4f}")

            current_metric = val_metrics.get('mIoU', 0)

            # Check if best
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                print(f"  New best: {current_metric:.4f}")

            # Save checkpoint
            self._save_checkpoint(is_best)

            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('val/loss', val_metrics['total'], epoch)
                for k, v in val_metrics.items():
                    if k != 'total':
                        self.writer.add_scalar(f'val/{k}', v, epoch)

        print(f"\nTraining complete. Best mIoU: {self.best_metric:.4f}")

        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='UniSCC Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    seed = args.seed or config.get('seed', 42)
    set_seed(seed)

    # Create trainer
    trainer = Trainer(config, resume_path=args.resume)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
