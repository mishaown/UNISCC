"""
Data transforms for bitemporal image pairs.
Ensures identical spatial transforms are applied to both time points.
"""

import random
from typing import Dict, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# Dataset-specific normalization statistics
NORMALIZE_STATS = {
    'imagenet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'second_cc': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'levir_mci': {
        'mean': [0.39073, 0.38623, 0.32989],
        'std': [0.15329, 0.14628, 0.13648]
    },
    'levir_mci_A': {
        'mean': [0.44152, 0.43863, 0.37418],
        'std': [0.17623, 0.16578, 0.15337]
    },
    'levir_mci_B': {
        'mean': [0.33992, 0.33383, 0.28561],
        'std': [0.13035, 0.12678, 0.11959]
    }
}


class PairedTransform:
    """
    Applies identical spatial transforms to paired images and labels.
    
    Args:
        image_size: Target image size
        is_train: Enable data augmentation
        augmentation: Augmentation configuration dict
        normalize_type: Normalization statistics to use
    """
    
    def __init__(
        self,
        image_size: int = 256,
        is_train: bool = True,
        augmentation: Optional[Dict] = None,
        normalize_type: str = 'imagenet'
    ):
        self.image_size = image_size
        self.is_train = is_train
        self.aug = augmentation or {}
        
        # Get normalization stats
        stats = NORMALIZE_STATS.get(normalize_type, NORMALIZE_STATS['imagenet'])
        self.normalize = T.Normalize(mean=stats['mean'], std=stats['std'])
    
    def __call__(
        self,
        rgb_a: Image.Image,
        rgb_b: Image.Image,
        label: Optional[Union[Image.Image, Tuple[Image.Image, ...], list]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...], list]]]:
        """
        Apply transforms to image pair and optional label.
        
        Args:
            rgb_a: Pre-change RGB image
            rgb_b: Post-change RGB image
            label: Segmentation label (semantic or binary)
        
        Returns:
            Transformed tensors (rgb_a, rgb_b, label)
        """
        # Resize
        rgb_a = TF.resize(rgb_a, [self.image_size, self.image_size])
        rgb_b = TF.resize(rgb_b, [self.image_size, self.image_size])
        if label is not None:
            label = self._apply_to_label(
                label,
                lambda img: TF.resize(
                    img,
                    [self.image_size, self.image_size],
                    interpolation=TF.InterpolationMode.NEAREST
                )
            )
        
        # Apply augmentation during training
        if self.is_train:
            rgb_a, rgb_b, label = self._augment(rgb_a, rgb_b, label)
        
        # Convert to tensors (skip if already tensor from augmentation)
        if not torch.is_tensor(rgb_a):
            rgb_a = TF.to_tensor(rgb_a)
        if not torch.is_tensor(rgb_b):
            rgb_b = TF.to_tensor(rgb_b)
        
        if label is not None:
            label = self._labels_to_tensor(label)
        
        # Normalize RGB
        rgb_a = self.normalize(rgb_a)
        rgb_b = self.normalize(rgb_b)
        
        return rgb_a, rgb_b, label

    def _apply_to_label(self, label, fn):
        if label is None:
            return None
        if isinstance(label, (list, tuple)):
            return type(label)(fn(item) for item in label)
        return fn(label)

    def _labels_to_tensor(self, label):
        if label is None:
            return None
        if isinstance(label, (list, tuple)):
            return type(label)(
                item.long() if torch.is_tensor(item) else torch.from_numpy(np.array(item)).long()
                for item in label
            )
        if torch.is_tensor(label):
            return label.long()
        return torch.from_numpy(np.array(label)).long()
    
    def _augment(
        self,
        rgb_a: Image.Image,
        rgb_b: Image.Image,
        label: Optional[Union[Image.Image, Tuple[Image.Image, ...], list]]
    ) -> Tuple[Image.Image, Image.Image, Optional[Union[Image.Image, Tuple[Image.Image, ...], list]]]:
        """Apply spatial and color augmentations."""
        
        train_aug = self.aug.get('train', {})
        
        # Horizontal flip
        if random.random() < train_aug.get('horizontal_flip', 0.5):
            rgb_a = TF.hflip(rgb_a)
            rgb_b = TF.hflip(rgb_b)
            if label is not None:
                label = self._apply_to_label(label, TF.hflip)
        
        # Vertical flip
        if random.random() < train_aug.get('vertical_flip', 0.5):
            rgb_a = TF.vflip(rgb_a)
            rgb_b = TF.vflip(rgb_b)
            if label is not None:
                label = self._apply_to_label(label, TF.vflip)
        
        # Random rotation (90 degree increments)
        if random.random() < train_aug.get('rotation', 0.3):
            angle = random.choice([90, 180, 270])
            rgb_a = TF.rotate(rgb_a, angle)
            rgb_b = TF.rotate(rgb_b, angle)
            if label is not None:
                label = self._apply_to_label(
                    label,
                    lambda img: TF.rotate(img, angle, interpolation=TF.InterpolationMode.NEAREST)
                )
        
        # Color jitter (same transform for both images)
        cj_config = train_aug.get('color_jitter', {})
        if random.random() < cj_config.get('p', 0.3):
            # Use tensor-based jitter to avoid PIL hue overflow in recent numpy
            brightness = cj_config.get('brightness', 0.2)
            contrast = cj_config.get('contrast', 0.2)
            saturation = cj_config.get('saturation', 0.1)
            hue = cj_config.get('hue', 0.05)

            rgb_a = TF.to_tensor(rgb_a) if not torch.is_tensor(rgb_a) else rgb_a
            rgb_b = TF.to_tensor(rgb_b) if not torch.is_tensor(rgb_b) else rgb_b

            transforms = []
            if brightness > 0:
                b = random.uniform(max(0, 1 - brightness), 1 + brightness)
                transforms.append(lambda img: TF.adjust_brightness(img, b))
            if contrast > 0:
                c = random.uniform(max(0, 1 - contrast), 1 + contrast)
                transforms.append(lambda img: TF.adjust_contrast(img, c))
            if saturation > 0:
                s = random.uniform(max(0, 1 - saturation), 1 + saturation)
                transforms.append(lambda img: TF.adjust_saturation(img, s))
            if hue > 0:
                h = random.uniform(-hue, hue)
                transforms.append(lambda img: TF.adjust_hue(img, h))

            random.shuffle(transforms)
            for fn in transforms:
                rgb_a = fn(rgb_a)
                rgb_b = fn(rgb_b)
        
        # Gaussian blur
        if random.random() < train_aug.get('gaussian_blur', 0.2):
            kernel_size = 3
            rgb_a = TF.gaussian_blur(rgb_a, kernel_size)
            rgb_b = TF.gaussian_blur(rgb_b, kernel_size)
        
        return rgb_a, rgb_b, label


def denormalize(tensor: torch.Tensor, normalize_type: str = 'imagenet') -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        normalize_type: Type of normalization used
    
    Returns:
        Denormalized tensor
    """
    stats = NORMALIZE_STATS.get(normalize_type, NORMALIZE_STATS['imagenet'])
    mean = torch.tensor(stats['mean'])
    std = torch.tensor(stats['std'])
    
    if tensor.dim() == 4:  # [B, C, H, W]
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:  # [C, H, W]
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    
    return tensor * std.to(tensor.device) + mean.to(tensor.device)
