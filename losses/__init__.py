"""
UniSCC Loss Functions

- scd_loss: Change Detection losses (Focal, Dice, BCE, Combined)
- caption_loss: Caption generation losses (CE, Label Smoothing)
- consistency_loss: Multi-task consistency loss
"""

from .scd_loss import (
    FocalLoss,
    DiceLoss,
    SCDLoss,
    BinaryChangeLoss,
)

from .caption_loss import (
    CaptionLoss,
    LabelSmoothingLoss,
)

from .consistency_loss import (
    ConsistencyLoss,
)

__all__ = [
    'FocalLoss',
    'DiceLoss',
    'SCDLoss',
    'BinaryChangeLoss',
    'CaptionLoss',
    'LabelSmoothingLoss',
    'ConsistencyLoss',
]
