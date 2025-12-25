#!/usr/bin/env python3
"""
LEVIR-MCI Dataset Visualization Script

Visualizes paired RGB images, change detection masks, and captions.
"""

import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter

# Configuration - Update this path to your dataset location
DATASET_ROOT = r"E:\CD-Experiment\Datasets\LEVIR-MCI-dataset"

# Change detection color map
CHANGE_CLASSES = {
    0: ("No Change", "#000000"),
    1: ("Building", "#FF0000"),
    2: ("Road", "#FFFF00"),
}


def load_dataset_json(json_path):
    """Load the LEVIR-MCI JSON annotation file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_samples_by_split(data, split='train', num_samples=5, random_seed=42):
    """Get random samples from a specific split."""
    random.seed(random_seed)
    
    split_images = [img for img in data['images'] if img.get('split') == split]
    
    if len(split_images) < num_samples:
        samples = split_images
    else:
        samples = random.sample(split_images, num_samples)
    
    return samples


def load_image(image_path):
    """Load an image and return as numpy array."""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except FileNotFoundError:
        print(f"Warning: Image not found: {image_path}")
        return None


def create_change_legend():
    """Create legend patches for change classes."""
    patches = []
    for class_id, (name, color) in CHANGE_CLASSES.items():
        patches.append(mpatches.Patch(color=color, label=name))
    return patches


def visualize_sample(sample, dataset_root, ax_row, sample_idx):
    """Visualize a single sample."""
    filepath = sample.get('split', sample.get('filepath', 'train'))
    filename = sample['filename']
    
    rgb_a_path = Path(dataset_root) / "images" / filepath / "A" / filename
    rgb_b_path = Path(dataset_root) / "images" / filepath / "B" / filename
    label_path = Path(dataset_root) / "images" / filepath / "label" / filename
    label_rgb_path = Path(dataset_root) / "images" / filepath / "label_rgb" / filename
    
    rgb_a = load_image(rgb_a_path)
    rgb_b = load_image(rgb_b_path)
    label = load_image(label_path)
    label_rgb = load_image(label_rgb_path)
    
    captions = [sent['raw'] for sent in sample.get('sentences', [])]
    change_flag = sample.get('changeflag', 'N/A')
    
    if rgb_a is not None:
        ax_row[0].imshow(rgb_a)
    ax_row[0].set_title(f"RGB A (Before)", fontsize=10)
    ax_row[0].axis('off')
    
    if rgb_b is not None:
        ax_row[1].imshow(rgb_b)
    ax_row[1].set_title(f"RGB B (After)", fontsize=10)
    ax_row[1].axis('off')
    
    if label is not None:
        ax_row[2].imshow(label, cmap='gray')
    ax_row[2].set_title(f"Change Mask (Binary)", fontsize=10)
    ax_row[2].axis('off')
    
    if label_rgb is not None:
        ax_row[3].imshow(label_rgb)
    ax_row[3].set_title(f"Change Mask (RGB)", fontsize=10)
    ax_row[3].axis('off')
    
    caption_text = f"Sample {sample_idx + 1}: {filename}\n"
    caption_text += f"Split: {filepath} | Change: {'Yes' if change_flag == 1 else 'No'}\n"
    caption_text += "-" * 40 + "\n"
    caption_text += "Captions:\n"
    for i, cap in enumerate(captions[:3], 1):
        caption_text += f"  {i}. {cap}\n"
    if len(captions) > 3:
        caption_text += f"  ... and {len(captions) - 3} more"
    
    ax_row[4].text(0.05, 0.95, caption_text, transform=ax_row[4].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_row[4].axis('off')
    ax_row[4].set_title("Captions", fontsize=10)


def visualize_dataset(dataset_root, num_samples=5, split='train', random_seed=42):
    """Main visualization function."""
    json_path = Path(dataset_root) / "LevirCCcaptions.json"
    
    print(f"Loading dataset from: {json_path}")
    data = load_dataset_json(json_path)
    
    samples = get_samples_by_split(data, split=split, num_samples=num_samples, random_seed=random_seed)
    print(f"Found {len(samples)} samples from '{split}' split")
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for idx, sample in enumerate(samples):
        visualize_sample(sample, dataset_root, axes[idx], idx)
    
    fig.suptitle(f"LEVIR-MCI Dataset Visualization ({split.upper()} split)\n"
                 f"Remote Sensing Change Detection & Captioning Dataset", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    legend_patches = create_change_legend()
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, 
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    return fig


def visualize_all_splits(dataset_root, samples_per_split=2, random_seed=42):
    """Visualize samples from all splits."""
    json_path = Path(dataset_root) / "LevirCCcaptions.json"
    data = load_dataset_json(json_path)
    
    all_samples = []
    for split in ['train', 'val', 'test']:
        samples = get_samples_by_split(data, split=split, num_samples=samples_per_split, random_seed=random_seed)
        for s in samples:
            s['_display_split'] = split
        all_samples.extend(samples)
    
    num_samples = len(all_samples)
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for idx, sample in enumerate(all_samples):
        visualize_sample(sample, dataset_root, axes[idx], idx)
    
    fig.suptitle("LEVIR-MCI Dataset - Samples from All Splits\n"
                 "Remote Sensing Image Change Detection & Captioning", 
                 fontsize=14, fontweight='bold', y=1.01)
    
    legend_patches = create_change_legend()
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, 
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    
    plt.tight_layout()
    return fig


def collect_dataset_statistics(dataset_root):
    """Collect basic statistics about the dataset."""
    json_path = Path(dataset_root) / "LevirCCcaptions.json"
    data = load_dataset_json(json_path)
    
    images = data['images']
    
    splits = {}
    for img in images:
        split = img.get('split', img.get('filepath', 'unknown'))
        splits[split] = splits.get(split, 0) + 1
    
    total_captions = sum(len(img.get('sentences', [])) for img in images)
    change_count = sum(1 for img in images if img.get('changeflag') == 1)
    no_change_count = len(images) - change_count
    
    # Vocabulary statistics
    all_tokens = []
    caption_lengths = []
    for img in images:
        for sent in img.get('sentences', []):
            tokens = sent.get('tokens', [])
            all_tokens.extend(tokens)
            caption_lengths.append(len(tokens))
    
    unique_words = set(all_tokens)
    word_counts = Counter(all_tokens)
    
    stats = {
        "total_images": len(images),
        "images_per_split": dict(sorted(splits.items())),
        "total_captions": total_captions,
        "avg_captions_per_image": round(total_captions / len(images), 2) if images else 0,
        "classes": {
            "count": len(CHANGE_CLASSES),
            "labels": [
                {"id": class_id, "name": name}
                for class_id, (name, _color) in CHANGE_CLASSES.items()
            ],
        },
        "change_distribution": {
            "with_change": change_count,
            "no_change": no_change_count,
        },
        "vocabulary": {
            "size": len(unique_words),
            "avg_caption_length": round(float(np.mean(caption_lengths)), 2) if caption_lengths else 0,
            "max_caption_length": max(caption_lengths) if caption_lengths else 0,
            "min_caption_length": min(caption_lengths) if caption_lengths else 0,
            "top_10_words": [
                {"word": word, "count": count}
                for word, count in word_counts.most_common(10)
            ],
        },
        "dataset_info": {
            "image_size": "256x256",
            "resolution": "0.5 m/pixel",
            "change_classes": ["Building (red)", "Road (yellow)"],
        },
    }
    return stats


def print_dataset_statistics(stats):
    """Print basic statistics about the dataset."""
    print()
    print("=" * 60)
    print("LEVIR-MCI Dataset Statistics")
    print("=" * 60)
    print()
    print(f"Total images: {stats['total_images']}")
    print()
    print("Images per split:")
    for split, count in stats["images_per_split"].items():
        print(f"  - {split}: {count}")
    
    print()
    print(f"Total captions: {stats['total_captions']}")
    print(f"Average captions per image: {stats['avg_captions_per_image']:.2f}")
    
    change_dist = stats["change_distribution"]
    print()
    print("Change distribution:")
    print(f"  - With change: {change_dist['with_change']}")
    print(f"  - No change: {change_dist['no_change']}")
    
    classes = stats["classes"]
    print()
    print(f"Classes: {classes['count']}")
    for entry in classes["labels"]:
        print(f"  - {entry['id']}: {entry['name']}")
    
    vocab = stats["vocabulary"]
    print()
    print(f"Vocabulary size: {vocab['size']}")
    print(f"Average caption length: {vocab['avg_caption_length']:.2f} words")
    print(f"Max caption length: {vocab['max_caption_length']} words")
    print(f"Min caption length: {vocab['min_caption_length']} words")
    
    print()
    print("Top 10 most common words:")
    for entry in vocab["top_10_words"]:
        print(f"  - '{entry['word']}': {entry['count']}")
    
    print()
    print("=" * 60)
    print("Dataset Info:")
    print("  - Image size: 256x256 pixels")
    print("  - Resolution: 0.5 m/pixel")
    print("  - Change classes: Building (red), Road (yellow)")
    print("=" * 60)
    print()


if __name__ == "__main__":
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    stats = collect_dataset_statistics(DATASET_ROOT)
    print_dataset_statistics(stats)
    
    stats_path = results_dir / "levir_dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset statistics saved to: {stats_path}")
    
    print("Generating visualization with samples from all splits...")
    fig = visualize_all_splits(DATASET_ROOT, samples_per_split=2, random_seed=42)
    
    output_path = results_dir / "levir_mci_visualization.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.close()
    
    print("\nGenerating train split visualization (5 samples)...")
    fig2 = visualize_dataset(DATASET_ROOT, num_samples=5, split='train', random_seed=123)
    
    output_path2 = results_dir / "levir_mci_train_samples.png"
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Train visualization saved to: {output_path2}")
    
    plt.close()
