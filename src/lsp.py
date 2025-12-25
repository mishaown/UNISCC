"""
Learnable Semantic Prompts (LSP) - Shared Semantic Space

Creates learnable text embeddings that serve as the shared semantic space
for both change detection and caption generation.

Key features:
- CLIP-initialized for strong semantic priors
- Learnable offsets for task adaptation
- Used by both SemanticChangeHead and CaptionDecoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LearnableSemanticPrompts(nn.Module):
    """
    Learnable semantic prompts for shared semantic space.

    Creates embeddings for each semantic class that are used by:
    1. SemanticChangeHead: for similarity-based classification
    2. CaptionDecoder: as cross-attention context

    Args:
        dataset: 'second_cc' or 'levir_mci'
        prompt_dim: Dimension of prompt embeddings
        clip_model_name: CLIP model for initialization
        learnable: Whether prompts can be fine-tuned
    """

    def __init__(
        self,
        dataset: str = "second_cc",
        prompt_dim: int = 512,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        learnable: bool = True
    ):
        super().__init__()
        self.dataset = dataset
        self.prompt_dim = prompt_dim
        self.learnable = learnable

        # Set class names based on dataset
        if dataset == "second_cc":
            self.class_names = SECOND_CC_CLASSES
            self.num_classes = 7
        else:  # levir_mci
            self.class_names = LEVIR_MCI_CLASSES
            self.num_classes = 3

        # Initialize prompts from CLIP
        clip_dim = 512  # CLIP base model dimension
        base_prompts = self._init_from_clip(clip_model_name)

        # Register base prompts (frozen)
        self.register_buffer('base_prompts', base_prompts)

        # Learnable offsets for fine-tuning
        if learnable:
            self.prompt_offsets = nn.Parameter(torch.zeros(self.num_classes, clip_dim))
        else:
            self.register_buffer('prompt_offsets', torch.zeros(self.num_classes, clip_dim))

        # Project to desired dimension if needed
        if clip_dim != prompt_dim:
            self.proj = nn.Sequential(
                nn.Linear(clip_dim, prompt_dim),
                nn.LayerNorm(prompt_dim)
            )
        else:
            self.proj = nn.Identity()

        # Learnable scale for similarity computation
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(14.28) from CLIP

    def _init_from_clip(self, model_name: str) -> torch.Tensor:
        """Initialize prompts from CLIP text encoder."""
        if not CLIP_AVAILABLE:
            print(f"CLIP not available, using random initialization for {self.num_classes} classes")
            return torch.randn(self.num_classes, 512) * 0.02

        try:
            clip_model = CLIPModel.from_pretrained(model_name)
            tokenizer = CLIPTokenizer.from_pretrained(model_name)

            # Create descriptive prompts
            if self.dataset == "second_cc":
                text_prompts = [
                    f"a satellite image showing {name}" for name in self.class_names
                ]
            else:
                text_prompts = [
                    f"a remote sensing image with {name}" for name in self.class_names
                ]

            # Encode with CLIP
            with torch.no_grad():
                inputs = tokenizer(text_prompts, return_tensors="pt", padding=True)
                text_features = clip_model.get_text_features(**inputs)
                text_features = F.normalize(text_features, dim=-1)

            print(f"Initialized {self.num_classes} semantic prompts from CLIP")
            return text_features

        except Exception as e:
            print(f"Failed to load CLIP: {e}, using random initialization")
            return torch.randn(self.num_classes, 512) * 0.02

    def get_prompts(self) -> torch.Tensor:
        """
        Get the semantic prompts.

        Returns:
            [K, D] tensor of semantic prompts where K is num_classes
        """
        # Add learnable offset to base prompts
        prompts = self.base_prompts + self.prompt_offsets

        # Project to desired dimension
        prompts = self.proj(prompts)

        # Normalize for cosine similarity
        prompts = F.normalize(prompts, dim=-1)

        return prompts

    def get_prompt_for_class(self, class_idx: int) -> torch.Tensor:
        """Get prompt for a specific class."""
        prompts = self.get_prompts()
        return prompts[class_idx]

    def get_class_name(self, class_idx: int) -> str:
        """Get class name for a given index."""
        return self.class_names[class_idx]

    def compute_similarity(
        self,
        features: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute similarity between features and prompts.

        Args:
            features: [..., D] features to compare
            normalize: Whether to normalize features

        Returns:
            [..., K] similarity scores
        """
        prompts = self.get_prompts()  # [K, D]

        if normalize:
            features = F.normalize(features, dim=-1)

        # Compute cosine similarity scaled by learned temperature
        similarity = torch.matmul(features, prompts.t())
        similarity = similarity * self.logit_scale.exp()

        return similarity

    def forward(self) -> torch.Tensor:
        """Return semantic prompts."""
        return self.get_prompts()

    # Keep backward compatibility
    def get_transition_prompts(self) -> torch.Tensor:
        """Backward compatible method - returns class prompts."""
        return self.get_prompts()
