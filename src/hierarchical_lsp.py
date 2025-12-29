"""
Hierarchical Learnable Semantic Prompts (LSP) for UniSCC v5.0

Scale-specific semantic prompts that capture different levels of detail:
    - P2 (fine): "detailed view of {class}"
    - P3 (local): "local area with {class}"
    - P4 (regional): "regional {class} area"
    - P5 (context): "large scale {class} region"

Each scale has its own set of prompts initialized from CLIP with
scale-appropriate text templates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

try:
    from transformers import CLIPModel, CLIPTokenizer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: transformers not available. Using random initialization.")


# Dataset-specific class names
SECOND_CC_CLASSES = [
    "background with no change",
    "low vegetation area",
    "non-vegetated ground surface",
    "tree and forest area",
    "water body",
    "building structure",
    "playground and sports field"
]

LEVIR_MCI_CLASSES = [
    "unchanged area with no change",
    "building change area",
    "road change area"
]

# Scale-specific text templates
SCALE_TEMPLATES = {
    'P2': "detailed view of {} in satellite imagery",
    'P3': "local area showing {} in remote sensing image",
    'P4': "regional {} area in aerial photograph",
    'P5': "large scale {} region in satellite view"
}


class HierarchicalSemanticPrompts(nn.Module):
    """
    Hierarchical Semantic Prompts for UniSCC v5.0.

    Creates scale-specific prompts for each pyramid level, enabling
    the model to learn different representations at different scales.

    Each scale has:
    - Base prompts: Initialized from CLIP with scale-specific templates
    - Learnable offsets: Task-specific adaptations
    - Projection: Maps to unified prompt dimension

    Args:
        dataset: 'second_cc' or 'levir_mci'
        prompt_dim: Output prompt dimension (256 to match pyramid features)
        num_scales: Number of pyramid levels (4)
        clip_model_name: CLIP model for initialization
        learnable: Whether offsets are learnable
    """

    def __init__(
        self,
        dataset: str = 'second_cc',
        prompt_dim: int = 256,
        num_scales: int = 4,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        learnable: bool = True
    ):
        super().__init__()

        self.dataset = dataset
        self.prompt_dim = prompt_dim
        self.num_scales = num_scales
        self.scale_names = ['P2', 'P3', 'P4', 'P5'][:num_scales]

        # Set class names based on dataset
        if dataset == 'second_cc':
            self.class_names = SECOND_CC_CLASSES
        else:
            self.class_names = LEVIR_MCI_CLASSES

        self.num_classes = len(self.class_names)

        # CLIP dimension
        clip_dim = 512

        # Initialize base prompts from CLIP for each scale
        base_prompts = self._init_from_clip(clip_model_name)

        # Register base prompts as buffers (frozen)
        for scale in self.scale_names:
            self.register_buffer(f'base_prompts_{scale}', base_prompts[scale])

        # Learnable offsets for each scale
        if learnable:
            self.offsets = nn.ParameterDict({
                scale: nn.Parameter(torch.zeros(self.num_classes, clip_dim))
                for scale in self.scale_names
            })
        else:
            for scale in self.scale_names:
                self.register_buffer(f'offsets_{scale}', torch.zeros(self.num_classes, clip_dim))
            self.offsets = None

        # Project from CLIP dim to prompt dim
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, prompt_dim),
            nn.LayerNorm(prompt_dim)
        )

        # Learnable scale for similarity computation
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def _init_from_clip(self, model_name: str) -> Dict[str, torch.Tensor]:
        """
        Initialize prompts from CLIP text encoder with scale-specific templates.

        Returns:
            dict of {'P2': [K, 512], 'P3': [K, 512], ...}
        """
        if not CLIP_AVAILABLE:
            print(f"CLIP not available, using random initialization")
            return {
                scale: torch.randn(self.num_classes, 512) * 0.02
                for scale in self.scale_names
            }

        try:
            clip_model = CLIPModel.from_pretrained(model_name)
            tokenizer = CLIPTokenizer.from_pretrained(model_name)

            base_prompts = {}

            for scale in self.scale_names:
                template = SCALE_TEMPLATES[scale]

                # Create scale-specific text prompts
                text_prompts = [
                    template.format(name) for name in self.class_names
                ]

                # Encode with CLIP
                with torch.no_grad():
                    inputs = tokenizer(text_prompts, return_tensors="pt", padding=True)
                    text_features = clip_model.get_text_features(**inputs)
                    text_features = F.normalize(text_features, dim=-1)

                base_prompts[scale] = text_features
                print(f"Initialized {self.num_classes} prompts for {scale} from CLIP")

            return base_prompts

        except Exception as e:
            print(f"Failed to load CLIP: {e}, using random initialization")
            return {
                scale: torch.randn(self.num_classes, 512) * 0.02
                for scale in self.scale_names
            }

    def get_prompts_for_scale(self, scale: str) -> torch.Tensor:
        """
        Get prompts for a specific scale.

        Args:
            scale: 'P2', 'P3', 'P4', or 'P5'

        Returns:
            [K, prompt_dim] normalized prompts
        """
        base = getattr(self, f'base_prompts_{scale}')

        if self.offsets is not None:
            offset = self.offsets[scale]
        else:
            offset = getattr(self, f'offsets_{scale}')

        # Combine base and offset
        prompts = base + offset

        # Project to target dimension
        prompts = self.proj(prompts)

        # Normalize
        prompts = F.normalize(prompts, dim=-1)

        return prompts

    def forward(self) -> Dict[str, torch.Tensor]:
        """
        Get all hierarchical prompts.

        Returns:
            dict with 'prompts_P2', 'prompts_P3', 'prompts_P4', 'prompts_P5'
            Each tensor has shape [K, prompt_dim]
        """
        prompts = {}
        for scale in self.scale_names:
            prompts[f'prompts_{scale}'] = self.get_prompts_for_scale(scale)
        return prompts

    def compute_similarity(
        self,
        features: torch.Tensor,
        scale: str,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute similarity between features and prompts for a specific scale.

        Args:
            features: [..., D] features to compare
            scale: Which scale's prompts to use
            normalize: Whether to normalize features

        Returns:
            [..., K] similarity scores
        """
        prompts = self.get_prompts_for_scale(scale)

        if normalize:
            features = F.normalize(features, dim=-1)

        similarity = torch.matmul(features, prompts.t())
        similarity = similarity * self.logit_scale.exp()

        return similarity


class EfficientHierarchicalLSP(nn.Module):
    """
    Memory-efficient Hierarchical LSP using shared base + scale-specific offsets.

    Uses a single shared base prompt set with scale-specific learnable offsets.
    Significantly reduces parameter count.
    """

    def __init__(
        self,
        dataset: str = 'second_cc',
        prompt_dim: int = 256,
        num_scales: int = 4,
        clip_model_name: str = "openai/clip-vit-base-patch32"
    ):
        super().__init__()

        self.dataset = dataset
        self.prompt_dim = prompt_dim
        self.num_scales = num_scales
        self.scale_names = ['P2', 'P3', 'P4', 'P5'][:num_scales]

        # Set class names
        if dataset == 'second_cc':
            self.class_names = SECOND_CC_CLASSES
        else:
            self.class_names = LEVIR_MCI_CLASSES

        self.num_classes = len(self.class_names)
        clip_dim = 512

        # Shared base prompts
        base_prompts = self._init_shared_from_clip(clip_model_name)
        self.register_buffer('base_prompts', base_prompts)

        # Scale-specific offsets (smaller dimension)
        self.scale_offsets = nn.ParameterDict({
            scale: nn.Parameter(torch.zeros(self.num_classes, 64))
            for scale in self.scale_names
        })

        # Scale offset projection
        self.offset_proj = nn.Linear(64, clip_dim)

        # Main projection
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, prompt_dim),
            nn.LayerNorm(prompt_dim)
        )

        # Initialize scale offsets with small random values
        for scale in self.scale_names:
            nn.init.trunc_normal_(self.scale_offsets[scale], std=0.02)

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def _init_shared_from_clip(self, model_name: str) -> torch.Tensor:
        """Initialize shared base prompts from CLIP."""
        if not CLIP_AVAILABLE:
            return torch.randn(self.num_classes, 512) * 0.02

        try:
            clip_model = CLIPModel.from_pretrained(model_name)
            tokenizer = CLIPTokenizer.from_pretrained(model_name)

            # Use generic template for shared base
            text_prompts = [
                f"a satellite image showing {name}" for name in self.class_names
            ]

            with torch.no_grad():
                inputs = tokenizer(text_prompts, return_tensors="pt", padding=True)
                text_features = clip_model.get_text_features(**inputs)
                text_features = F.normalize(text_features, dim=-1)

            print(f"Initialized shared base prompts from CLIP")
            return text_features

        except Exception as e:
            print(f"Failed to load CLIP: {e}")
            return torch.randn(self.num_classes, 512) * 0.02

    def forward(self) -> Dict[str, torch.Tensor]:
        """Get all hierarchical prompts efficiently."""
        prompts = {}

        for scale in self.scale_names:
            # Get scale-specific offset
            offset = self.offset_proj(self.scale_offsets[scale])

            # Combine with base
            p = self.base_prompts + offset

            # Project and normalize
            p = self.proj(p)
            p = F.normalize(p, dim=-1)

            prompts[f'prompts_{scale}'] = p

        return prompts
