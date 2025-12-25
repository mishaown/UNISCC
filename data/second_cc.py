"""
SECOND-CC Dataset Loader

Dataset for Semantic Change Detection + Change Captioning.
- 6,041 bitemporal image pairs (256×256)
- 7 semantic classes
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


class SECONDCCDataset(Dataset):
    """
    SECOND-CC Dataset for Semantic Change Detection and Change Captioning.
    
    Args:
        root: Path to SECOND-CC-AUG directory
        split: 'train', 'val', or 'test'
        vocab: Vocabulary object
        transform: Transform for images
        max_caption_length: Maximum caption length
        num_captions: Number of captions per sample
    """
    
    NUM_CLASSES = 7
    CLASS_NAMES = [
        'no_change', 'low_vegetation', 'non_vegetated_ground',
        'tree', 'water', 'building', 'playground'
    ]
    
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
        json_path = self.root / "SECOND-CC-AUG.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Filter by split
        self.samples = [
            img for img in data['images']
            if img.get('split') == split
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
            normalize_type='second_cc'
        )
        
        print(f"SECOND-CC [{split}]: {len(self.samples)} samples, vocab: {len(self.vocab) if self.vocab else 0}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        filepath = sample['filepath']
        filename = sample['filename']
        
        # Load images
        rgb_a = Image.open(self.root / filepath / "rgb" / "A" / filename).convert('RGB')
        rgb_b = Image.open(self.root / filepath / "rgb" / "B" / filename).convert('RGB')
        sem_a = Image.open(self.root / filepath / "sem" / "A" / filename).convert('L')
        sem_b = Image.open(self.root / filepath / "sem" / "B" / filename).convert('L')
        
        # Apply transforms to RGB images only, handle semantic maps separately
        rgb_a_t, rgb_b_t, _ = self.transform(rgb_a, rgb_b, None)
        
        # Convert semantic maps
        sem_a = torch.from_numpy(np.array(sem_a.resize((256, 256), Image.NEAREST))).long()
        sem_b = torch.from_numpy(np.array(sem_b.resize((256, 256), Image.NEAREST))).long()
        
        # Compute semantic change map (transition encoding)
        # change_id = sem_a * K + sem_b where K = num_classes
        invalid = (sem_a == 255) | (sem_b == 255)
        sem_a = sem_a.clamp(0, self.NUM_CLASSES - 1)
        sem_b = sem_b.clamp(0, self.NUM_CLASSES - 1)
        change_map = sem_a * self.NUM_CLASSES + sem_b
        change_map[invalid] = 255
        
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
            'sem_a': sem_a,
            'sem_b': sem_b,
            'change_map': change_map,
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
            'sem_a': torch.stack([b['sem_a'] for b in batch]),
            'sem_b': torch.stack([b['sem_b'] for b in batch]),
            'change_map': torch.stack([b['change_map'] for b in batch]),
            'captions': torch.stack([b['captions'] for b in batch]),
            'caption_lengths': torch.stack([b['caption_lengths'] for b in batch]),
            'raw_captions': [b['raw_captions'] for b in batch],
            'change_flag': torch.tensor([b['change_flag'] for b in batch]),
            'filename': [b['filename'] for b in batch],
            'img_id': torch.tensor([b['img_id'] for b in batch])
        }


def create_secondcc_dataloaders(
    root: str,
    batch_size: int = 16,
    num_workers: int = 8,
    image_size: int = 256,
    max_caption_length: int = 50,
    min_word_freq: int = 5,
    augmentation: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """
    Create train, val, test dataloaders for SECOND-CC.
    
    Returns:
        train_loader, val_loader, test_loader, vocabulary
    """
    train_transform = PairedTransform(
        image_size=image_size,
        is_train=True,
        augmentation=augmentation,
        normalize_type='second_cc'
    )
    
    eval_transform = PairedTransform(
        image_size=image_size,
        is_train=False,
        normalize_type='second_cc'
    )
    
    # Create training dataset (builds vocabulary)
    train_dataset = SECONDCCDataset(
        root=root,
        split='train',
        transform=train_transform,
        max_caption_length=max_caption_length,
        min_word_freq=min_word_freq
    )
    
    vocab = train_dataset.vocab
    
    val_dataset = SECONDCCDataset(
        root=root,
        split='val',
        vocab=vocab,
        transform=eval_transform,
        max_caption_length=max_caption_length
    )
    
    test_dataset = SECONDCCDataset(
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
        collate_fn=SECONDCCDataset.collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=SECONDCCDataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=SECONDCCDataset.collate_fn
    )
    
    return train_loader, val_loader, test_loader, vocab
