"""
LEVIR-MCI Dataset Loader

Dataset for Multi-class Change Detection + Change Captioning.
- 10,077 bitemporal image pairs (256×256, 0.5m/pixel)
- Binary change masks (buildings + roads)
- 5 captions per image pair
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from .vocabulary import Vocabulary
from .transforms import PairedTransform


class LEVIRMCIDataset(Dataset):
    """
    LEVIR-MCI Dataset for Multi-class Change Detection and Change Captioning.
    
    Args:
        root: Path to LEVIR-MCI-dataset directory
        split: 'train', 'val', or 'test'
        vocab: Vocabulary object
        transform: Transform for images
        max_caption_length: Maximum caption length
        num_captions: Number of captions per sample
    """
    
    NUM_CLASSES = 3  # No Change, Building, Road
    CLASS_NAMES = ['no_change', 'building', 'road']

    # RGB color mapping for label_rgb semantic masks
    # Format: (R, G, B) -> class_id
    COLOR_MAP = {
        (0, 0, 0): 0,       # background/no_change
        (255, 0, 0): 1,     # building (red)
        (0, 0, 255): 2,     # road (blue)
    }
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        vocab: Optional[Vocabulary] = None,
        transform: Optional[PairedTransform] = None,
        max_caption_length: int = 50,
        num_captions: int = 5,
        min_word_freq: int = 5
    ):
        self.root = Path(root)
        self.split = split
        self.max_caption_length = max_caption_length
        self.num_captions = num_captions
        
        # Load annotations
        json_path = self.root / "LevirCCcaptions.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Filter by split
        self.samples = [
            img for img in data['images']
            if img.get('split', img.get('filepath')) == split
        ]
        
        # Build vocabulary from training data
        if vocab is None and split == 'train':
            self.vocab = Vocabulary(min_word_freq=min_word_freq)
            all_sentences = []
            for sample in self.samples:
                for sent in sample.get('sentences', []):
                    all_sentences.append(sent.get('tokens', []))
            self.vocab.build_vocab(all_sentences)
        else:
            self.vocab = vocab
        
        # Set up transforms
        self.transform = transform or PairedTransform(
            image_size=256,
            is_train=(split == 'train'),
            normalize_type='levir_mci'
        )
        
        print(f"LEVIR-MCI [{split}]: {len(self.samples)} samples, vocab: {len(self.vocab) if self.vocab else 0}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        split_dir = sample.get('split', sample.get('filepath', self.split))
        filename = sample['filename']
        
        # Load images
        rgb_a = Image.open(self.root / "images" / split_dir / "A" / filename).convert('RGB')
        rgb_b = Image.open(self.root / "images" / split_dir / "B" / filename).convert('RGB')
        
        # Load semantic label from label_rgb (after-change semantics)
        label_rgb_path = self.root / "images" / split_dir / "label_rgb" / filename
        if label_rgb_path.exists():
            label_rgb = Image.open(label_rgb_path).convert('RGB')
        else:
            # Fallback to regular label if label_rgb doesn't exist
            label_path = self.root / "images" / split_dir / "label" / filename
            if label_path.exists():
                label = Image.open(label_path)
                label_rgb = None
            else:
                label = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))
                label_rgb = None

        # Apply transforms to RGB images only
        rgb_a_t, rgb_b_t, _ = self.transform(rgb_a, rgb_b, None)

        # Convert label_rgb to class indices
        if label_rgb is not None:
            # Resize label_rgb to match image size
            label_rgb = label_rgb.resize((256, 256), Image.NEAREST)
            label_array = np.array(label_rgb)

            # Map RGB colors to class IDs
            label_t = np.zeros((256, 256), dtype=np.int64)
            for color, class_id in self.COLOR_MAP.items():
                mask = (label_array[:, :, 0] == color[0]) & \
                       (label_array[:, :, 1] == color[1]) & \
                       (label_array[:, :, 2] == color[2])
                label_t[mask] = class_id
            label_t = torch.from_numpy(label_t).long()
        else:
            # Fallback: use binary label
            label = label.resize((256, 256), Image.NEAREST)
            label_t = torch.from_numpy(np.array(label)).long()
            # Ensure binary values
            label_t = (label_t > 0).long()
        
        # Process captions
        sentences = sample.get('sentences', [])
        raw_captions = [s.get('raw', '').strip() for s in sentences[:self.num_captions]]
        
        # Encode captions
        encoded_captions = []
        caption_lengths = []
        
        for sent in sentences[:self.num_captions]:
            tokens = sent.get('tokens', [])
            if self.vocab:
                encoded = self.vocab.encode(tokens, self.max_caption_length)
                encoded_captions.append(encoded)
                caption_lengths.append(min(len(tokens) + 2, self.max_caption_length))
        
        # Pad to num_captions
        while len(encoded_captions) < self.num_captions:
            if encoded_captions:
                encoded_captions.append(encoded_captions[-1].clone())
                caption_lengths.append(caption_lengths[-1])
            else:
                encoded_captions.append(torch.zeros(self.max_caption_length, dtype=torch.long))
                caption_lengths.append(1)
        
        while len(raw_captions) < self.num_captions:
            raw_captions.append(raw_captions[-1] if raw_captions else "")
        
        return {
            'rgb_a': rgb_a_t,
            'rgb_b': rgb_b_t,
            'label': label_t,
            'captions': torch.stack(encoded_captions),
            'caption_lengths': torch.tensor(caption_lengths),
            'raw_captions': raw_captions,
            'change_flag': sample.get('changeflag', 1),
            'filename': filename,
            'img_id': sample.get('imgid', idx)
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function."""
        return {
            'rgb_a': torch.stack([b['rgb_a'] for b in batch]),
            'rgb_b': torch.stack([b['rgb_b'] for b in batch]),
            'label': torch.stack([b['label'] for b in batch]),
            'captions': torch.stack([b['captions'] for b in batch]),
            'caption_lengths': torch.stack([b['caption_lengths'] for b in batch]),
            'raw_captions': [b['raw_captions'] for b in batch],
            'change_flag': torch.tensor([b['change_flag'] for b in batch]),
            'filename': [b['filename'] for b in batch],
            'img_id': torch.tensor([b['img_id'] for b in batch])
        }


def create_levirmci_dataloaders(
    root: str,
    batch_size: int = 16,
    num_workers: int = 8,
    image_size: int = 256,
    max_caption_length: int = 50,
    min_word_freq: int = 5,
    augmentation: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """
    Create train, val, test dataloaders for LEVIR-MCI.
    
    Returns:
        train_loader, val_loader, test_loader, vocabulary
    """
    train_transform = PairedTransform(
        image_size=image_size,
        is_train=True,
        augmentation=augmentation,
        normalize_type='levir_mci'
    )
    
    eval_transform = PairedTransform(
        image_size=image_size,
        is_train=False,
        normalize_type='levir_mci'
    )
    
    # Create training dataset (builds vocabulary)
    train_dataset = LEVIRMCIDataset(
        root=root,
        split='train',
        transform=train_transform,
        max_caption_length=max_caption_length,
        min_word_freq=min_word_freq
    )
    
    vocab = train_dataset.vocab
    
    val_dataset = LEVIRMCIDataset(
        root=root,
        split='val',
        vocab=vocab,
        transform=eval_transform,
        max_caption_length=max_caption_length
    )
    
    test_dataset = LEVIRMCIDataset(
        root=root,
        split='test',
        vocab=vocab,
        transform=eval_transform,
        max_caption_length=max_caption_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=LEVIRMCIDataset.collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=LEVIRMCIDataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=LEVIRMCIDataset.collate_fn
    )
    
    return train_loader, val_loader, test_loader, vocab
