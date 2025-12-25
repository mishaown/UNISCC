"""
UniSCC Data Loading Modules

Supports:
- SECOND-CC: Semantic Change Detection + Change Captioning
- LEVIR-MCI: Binary Change Detection + Change Captioning
"""

from .second_cc import (
    SECONDCCDataset,
    create_secondcc_dataloaders,
)

from .levir_mci import (
    LEVIRMCIDataset,
    create_levirmci_dataloaders,
)

from .vocabulary import Vocabulary
from .transforms import PairedTransform

__all__ = [
    'SECONDCCDataset',
    'create_secondcc_dataloaders',
    'LEVIRMCIDataset',
    'create_levirmci_dataloaders',
    'Vocabulary',
    'PairedTransform',
]


def create_dataloaders(config):
    """
    Factory function to create dataloaders based on config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    dataset_name = config['dataset']['name']
    
    if dataset_name == 'SECOND-CC':
        return create_secondcc_dataloaders(
            root=config['dataset']['root'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            image_size=config['dataset']['image_size'],
            max_caption_length=config['dataset']['max_caption_length'],
            min_word_freq=config['dataset']['min_word_freq'],
            augmentation=config.get('augmentation', {})
        )
    elif dataset_name == 'LEVIR-MCI':
        return create_levirmci_dataloaders(
            root=config['dataset']['root'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            image_size=config['dataset']['image_size'],
            max_caption_length=config['dataset']['max_caption_length'],
            min_word_freq=config['dataset']['min_word_freq'],
            augmentation=config.get('augmentation', {})
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
