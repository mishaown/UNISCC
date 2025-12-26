#!/usr/bin/env python3
"""
Training Results Visualization and Analysis Script

Reads TensorBoard logs and creates comprehensive visualizations for:
- Loss curves (train/val)
- Change Detection metrics (mIoU, F1, OA)
- Learning rate schedule
- Comparison between datasets

Usage:
    python visualize_training.py --log_dir logs/levir_mci --output_dir results/levir_mci
    python visualize_training.py --log_dir logs/second_cc --output_dir results/second_cc
    python visualize_training.py --compare logs/levir_mci logs/second_cc --output_dir results/comparison
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not installed. Install with: pip install tensorboard")


def load_tensorboard_logs(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load metrics from TensorBoard event files.

    Args:
        log_dir: Path to directory containing TensorBoard event files

    Returns:
        Dictionary mapping metric names to list of (step, value) tuples
    """
    if not TENSORBOARD_AVAILABLE:
        raise ImportError("tensorboard is required. Install with: pip install tensorboard")

    metrics = defaultdict(list)

    # Find all event files
    log_path = Path(log_dir)
    event_files = list(log_path.glob("events.out.tfevents.*"))

    if not event_files:
        print(f"No event files found in {log_dir}")
        return metrics

    for event_file in event_files:
        try:
            ea = EventAccumulator(str(event_file))
            ea.Reload()

            # Get all scalar tags
            tags = ea.Tags().get('scalars', [])

            for tag in tags:
                events = ea.Scalars(tag)
                for event in events:
                    metrics[tag].append((event.step, event.value))

        except Exception as e:
            print(f"Error reading {event_file}: {e}")

    # Sort by step
    for tag in metrics:
        metrics[tag] = sorted(metrics[tag], key=lambda x: x[0])

    return dict(metrics)


def extract_epochs(metrics: Dict[str, List[Tuple[int, float]]]) -> Dict[str, np.ndarray]:
    """Convert step-based metrics to epoch arrays."""
    result = {}
    for tag, values in metrics.items():
        if values:
            steps = np.array([v[0] for v in values])
            vals = np.array([v[1] for v in values])
            result[tag] = {'steps': steps, 'values': vals}
    return result


def plot_loss_curves(
    metrics: Dict[str, dict],
    title: str = "Training Progress",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    ax = axes[0]
    if 'train/loss' in metrics:
        data = metrics['train/loss']
        ax.plot(data['steps'], data['values'], 'b-', linewidth=1.5, label='Train Loss')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation loss
    ax = axes[1]
    if 'val/loss' in metrics:
        data = metrics['val/loss']
        ax.plot(data['steps'], data['values'], 'r-', linewidth=2, marker='o', markersize=4, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def plot_cd_metrics(
    metrics: Dict[str, dict],
    title: str = "Change Detection Metrics",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot change detection metrics (mIoU, F1, OA)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metric_names = ['mIoU', 'F1', 'OA']
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    for ax, metric_name, color in zip(axes, metric_names, colors):
        tag = f'val/{metric_name}'
        if tag in metrics:
            data = metrics[tag]
            ax.plot(data['steps'], data['values'], color=color, linewidth=2,
                   marker='o', markersize=6, label=metric_name)

            # Add best value annotation
            best_idx = np.argmax(data['values'])
            best_val = data['values'][best_idx]
            best_epoch = data['steps'][best_idx]
            ax.axhline(y=best_val, color=color, linestyle='--', alpha=0.5)
            ax.annotate(f'Best: {best_val:.4f}\n(Epoch {best_epoch})',
                       xy=(best_epoch, best_val),
                       xytext=(10, -20), textcoords='offset points',
                       fontsize=9, color=color,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, 1)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def plot_learning_rate(
    metrics: Dict[str, dict],
    title: str = "Learning Rate Schedule",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 4))

    if 'train/lr' in metrics:
        data = metrics['train/lr']
        ax.plot(data['steps'], data['values'], 'g-', linewidth=1.5)
        ax.set_yscale('log')

    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def plot_comprehensive_summary(
    metrics: Dict[str, dict],
    dataset_name: str = "Dataset",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create a comprehensive summary plot with all metrics."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Training Loss (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if 'train/loss' in metrics:
        data = metrics['train/loss']
        ax1.plot(data['steps'], data['values'], 'b-', linewidth=1, alpha=0.7, label='Train')
    if 'val/loss' in metrics:
        data = metrics['val/loss']
        # Convert epochs to approximate steps for overlay
        ax1_twin = ax1.twinx()
        ax1_twin.plot(data['steps'], data['values'], 'r-', linewidth=2, marker='o', markersize=4, label='Val')
        ax1_twin.set_ylabel('Val Loss', color='r', fontsize=10)
        ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel('Step', fontsize=10)
    ax1.set_ylabel('Train Loss', color='b', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Loss Curves', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Learning Rate (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    if 'train/lr' in metrics:
        data = metrics['train/lr']
        ax2.plot(data['steps'], data['values'], 'g-', linewidth=1.5)
        ax2.set_yscale('log')
    ax2.set_xlabel('Step', fontsize=10)
    ax2.set_ylabel('LR', fontsize=10)
    ax2.set_title('Learning Rate', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # mIoU (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'val/mIoU' in metrics:
        data = metrics['val/mIoU']
        ax3.plot(data['steps'], data['values'], '#2ecc71', linewidth=2, marker='o', markersize=5)
        best_val = max(data['values'])
        ax3.axhline(y=best_val, color='#2ecc71', linestyle='--', alpha=0.5)
        ax3.set_title(f'mIoU (Best: {best_val:.4f})', fontsize=11, fontweight='bold')
    else:
        ax3.set_title('mIoU', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('mIoU', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    # F1 (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'val/F1' in metrics:
        data = metrics['val/F1']
        ax4.plot(data['steps'], data['values'], '#3498db', linewidth=2, marker='o', markersize=5)
        best_val = max(data['values'])
        ax4.axhline(y=best_val, color='#3498db', linestyle='--', alpha=0.5)
        ax4.set_title(f'F1 Score (Best: {best_val:.4f})', fontsize=11, fontweight='bold')
    else:
        ax4.set_title('F1 Score', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('F1', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

    # OA (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    if 'val/OA' in metrics:
        data = metrics['val/OA']
        ax5.plot(data['steps'], data['values'], '#9b59b6', linewidth=2, marker='o', markersize=5)
        best_val = max(data['values'])
        ax5.axhline(y=best_val, color='#9b59b6', linestyle='--', alpha=0.5)
        ax5.set_title(f'Overall Accuracy (Best: {best_val:.4f})', fontsize=11, fontweight='bold')
    else:
        ax5.set_title('Overall Accuracy', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Epoch', fontsize=10)
    ax5.set_ylabel('OA', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)
    ax5.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Summary statistics (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    # Generate summary text
    summary_lines = [f"Training Summary - {dataset_name}", "=" * 50]

    if 'val/mIoU' in metrics:
        best_miou = max(metrics['val/mIoU']['values'])
        summary_lines.append(f"Best mIoU: {best_miou:.4f}")
    if 'val/F1' in metrics:
        best_f1 = max(metrics['val/F1']['values'])
        summary_lines.append(f"Best F1: {best_f1:.4f}")
    if 'val/OA' in metrics:
        best_oa = max(metrics['val/OA']['values'])
        summary_lines.append(f"Best OA: {best_oa:.4f}")
    if 'val/loss' in metrics:
        min_loss = min(metrics['val/loss']['values'])
        summary_lines.append(f"Min Val Loss: {min_loss:.4f}")
    if 'train/loss' in metrics:
        final_loss = metrics['train/loss']['values'][-1]
        summary_lines.append(f"Final Train Loss: {final_loss:.4f}")

    summary_text = "\n".join(summary_lines)
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.suptitle(f'UniSCC Training Results - {dataset_name}',
                fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def compare_datasets(
    metrics_list: List[Dict[str, dict]],
    dataset_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Compare metrics across multiple datasets."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    metric_tags = ['val/mIoU', 'val/F1', 'val/OA', 'val/loss']
    metric_titles = ['mIoU', 'F1 Score', 'Overall Accuracy', 'Validation Loss']

    for ax, tag, title in zip(axes.flat, metric_tags, metric_titles):
        for i, (metrics, name) in enumerate(zip(metrics_list, dataset_names)):
            if tag in metrics:
                data = metrics[tag]
                ax.plot(data['steps'], data['values'],
                       color=colors[i % len(colors)],
                       linewidth=2, marker='o', markersize=4,
                       label=name)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if 'loss' not in tag.lower():
            ax.set_ylim(0, 1)

    plt.suptitle('Dataset Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def generate_report(
    metrics: Dict[str, dict],
    dataset_name: str,
    output_path: str
):
    """Generate a text report of training results."""
    lines = [
        "=" * 60,
        f"UniSCC Training Report - {dataset_name}",
        "=" * 60,
        ""
    ]

    # Best metrics
    lines.append("Best Validation Metrics:")
    lines.append("-" * 40)

    if 'val/mIoU' in metrics:
        best_idx = np.argmax(metrics['val/mIoU']['values'])
        best_val = metrics['val/mIoU']['values'][best_idx]
        best_epoch = metrics['val/mIoU']['steps'][best_idx]
        lines.append(f"  mIoU:  {best_val:.4f} (Epoch {best_epoch})")

    if 'val/F1' in metrics:
        best_idx = np.argmax(metrics['val/F1']['values'])
        best_val = metrics['val/F1']['values'][best_idx]
        best_epoch = metrics['val/F1']['steps'][best_idx]
        lines.append(f"  F1:    {best_val:.4f} (Epoch {best_epoch})")

    if 'val/OA' in metrics:
        best_idx = np.argmax(metrics['val/OA']['values'])
        best_val = metrics['val/OA']['values'][best_idx]
        best_epoch = metrics['val/OA']['steps'][best_idx]
        lines.append(f"  OA:    {best_val:.4f} (Epoch {best_epoch})")

    lines.append("")
    lines.append("Loss Summary:")
    lines.append("-" * 40)

    if 'val/loss' in metrics:
        min_idx = np.argmin(metrics['val/loss']['values'])
        min_val = metrics['val/loss']['values'][min_idx]
        min_epoch = metrics['val/loss']['steps'][min_idx]
        final_val = metrics['val/loss']['values'][-1]
        lines.append(f"  Min Val Loss:   {min_val:.4f} (Epoch {min_epoch})")
        lines.append(f"  Final Val Loss: {final_val:.4f}")

    if 'train/loss' in metrics:
        final_train = metrics['train/loss']['values'][-1]
        lines.append(f"  Final Train Loss: {final_train:.4f}")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize UniSCC Training Results')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Path to TensorBoard log directory')
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                       help='Compare multiple log directories')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for visualizations')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name for labels')
    args = parser.parse_args()

    if not TENSORBOARD_AVAILABLE:
        print("Error: tensorboard is required. Install with: pip install tensorboard")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # Compare multiple datasets
        metrics_list = []
        names = []
        for log_dir in args.compare:
            raw_metrics = load_tensorboard_logs(log_dir)
            metrics = extract_epochs(raw_metrics)
            metrics_list.append(metrics)
            names.append(Path(log_dir).name)

        compare_datasets(metrics_list, names,
                        save_path=str(output_dir / 'comparison.png'))

    elif args.log_dir:
        # Single dataset analysis
        dataset_name = args.dataset_name or Path(args.log_dir).name

        print(f"\nLoading logs from: {args.log_dir}")
        raw_metrics = load_tensorboard_logs(args.log_dir)

        if not raw_metrics:
            print("No metrics found in log directory")
            return

        print(f"Found metrics: {list(raw_metrics.keys())}")

        metrics = extract_epochs(raw_metrics)

        # Generate all plots
        plot_loss_curves(metrics, f"Loss Curves - {dataset_name}",
                        save_path=str(output_dir / 'loss_curves.png'))

        plot_cd_metrics(metrics, f"CD Metrics - {dataset_name}",
                       save_path=str(output_dir / 'cd_metrics.png'))

        plot_learning_rate(metrics, f"Learning Rate - {dataset_name}",
                          save_path=str(output_dir / 'learning_rate.png'))

        plot_comprehensive_summary(metrics, dataset_name,
                                  save_path=str(output_dir / 'summary.png'))

        # Generate text report
        generate_report(metrics, dataset_name,
                       str(output_dir / 'report.txt'))

        print(f"\nAll visualizations saved to: {output_dir}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
