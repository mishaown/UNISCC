"""
Learnable Semantic Prompts (LSP) - Shared Semantic Space

Creates learnable text embeddings that serve as the shared semantic space
for both change detection and caption generation.

Key features:
- CLIP-initialized for strong semantic priors
- Learnable offsets for task adaptation
- Used by both SemanticChangeHead and CaptionDecoder

v3.0 Addition: TransitionLSP
- Dual prompts for before/after semantics
- Learned transition embeddings [K, K, D] for class pair transitions
- Enables caption decoder to attend to "what changed into what"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

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

# Temporal templates for TransitionLSP
SECOND_CC_TEMPORAL_TEMPLATES = {
    'before': [
        "a satellite image showing {class_name} before the change occurred",
    ],
    'after': [
        "a satellite image showing {class_name} after the change occurred",
    ],
    'transition': "land cover changed from {class_a} to {class_b}"
}

LEVIR_MCI_TEMPORAL_TEMPLATES = {
    'before': [
        "a remote sensing image showing the area before {class_name}",
    ],
    'after': [
        "a remote sensing image showing the area after {class_name}",
    ],
    'transition': "the area changed from {class_a} to {class_b}"
}


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


class TransitionLSP(nn.Module):
    """
    Transition-Aware Semantic Prompts (TASP) for UniSCC v3.0.

    Creates three types of embeddings:
    1. prompts_A: Semantic prompts for before-change classes [K, D]
    2. prompts_B: Semantic prompts for after-change classes [K, D]
    3. transition_embeddings: Learned embeddings for class transitions [K, K, D]

    The transition_embeddings[i, j] represents the semantic meaning of
    "class i changed to class j", which the caption decoder uses to
    generate transition descriptions like "vegetation was replaced by building".

    Args:
        dataset: 'second_cc' or 'levir_mci'
        prompt_dim: Dimension of prompt embeddings
        clip_model_name: CLIP model for initialization
        learnable: Whether prompts can be fine-tuned
        transition_hidden_dim: Hidden dimension for transition encoder
    """

    def __init__(
        self,
        dataset: str = "second_cc",
        prompt_dim: int = 512,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        learnable: bool = True,
        transition_hidden_dim: int = 256
    ):
        super().__init__()
        self.dataset = dataset
        self.prompt_dim = prompt_dim
        self.learnable = learnable
        self.transition_hidden_dim = transition_hidden_dim

        # Set class names and templates based on dataset
        if dataset == "second_cc":
            self.class_names = SECOND_CC_CLASSES
            self.num_classes = 7
            self.templates = SECOND_CC_TEMPORAL_TEMPLATES
        else:  # levir_mci
            self.class_names = LEVIR_MCI_CLASSES
            self.num_classes = 3
            self.templates = LEVIR_MCI_TEMPORAL_TEMPLATES

        clip_dim = 512  # CLIP base model dimension

        # Initialize temporal prompts from CLIP
        base_prompts_A = self._init_from_clip_temporal(clip_model_name, 'before')
        base_prompts_B = self._init_from_clip_temporal(clip_model_name, 'after')

        # Register base prompts (frozen)
        self.register_buffer('base_prompts_A', base_prompts_A)
        self.register_buffer('base_prompts_B', base_prompts_B)

        # Learnable offsets for fine-tuning
        if learnable:
            self.prompt_offsets_A = nn.Parameter(torch.zeros(self.num_classes, clip_dim))
            self.prompt_offsets_B = nn.Parameter(torch.zeros(self.num_classes, clip_dim))
        else:
            self.register_buffer('prompt_offsets_A', torch.zeros(self.num_classes, clip_dim))
            self.register_buffer('prompt_offsets_B', torch.zeros(self.num_classes, clip_dim))

        # Project to desired dimension if needed
        if clip_dim != prompt_dim:
            self.proj = nn.Sequential(
                nn.Linear(clip_dim, prompt_dim),
                nn.LayerNorm(prompt_dim)
            )
        else:
            self.proj = nn.Identity()

        # Transition embedding generator
        # Takes concatenated (prompt_A, prompt_B) and outputs transition embedding
        self.transition_encoder = nn.Sequential(
            nn.Linear(prompt_dim * 2, transition_hidden_dim),
            nn.LayerNorm(transition_hidden_dim),
            nn.GELU(),
            nn.Linear(transition_hidden_dim, transition_hidden_dim),
            nn.LayerNorm(transition_hidden_dim),
            nn.GELU(),
            nn.Linear(transition_hidden_dim, prompt_dim),
            nn.LayerNorm(prompt_dim)
        )

        # Learnable "no change" embedding (when class_A == class_B)
        self.no_change_embed = nn.Parameter(torch.zeros(1, prompt_dim))
        nn.init.trunc_normal_(self.no_change_embed, std=0.02)

        # Learnable scale for similarity computation
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(14.28) from CLIP

    def _init_from_clip_temporal(self, model_name: str, temporal: str) -> torch.Tensor:
        """
        Initialize prompts from CLIP text encoder with temporal context.

        Args:
            model_name: CLIP model name
            temporal: 'before' or 'after'

        Returns:
            [K, 512] tensor of CLIP text embeddings
        """
        if not CLIP_AVAILABLE:
            print(f"CLIP not available, using random initialization for {temporal} prompts")
            return torch.randn(self.num_classes, 512) * 0.02

        try:
            clip_model = CLIPModel.from_pretrained(model_name)
            tokenizer = CLIPTokenizer.from_pretrained(model_name)

            # Create descriptive prompts with temporal context
            template = self.templates[temporal][0]
            text_prompts = [
                template.format(class_name=name) for name in self.class_names
            ]

            # Encode with CLIP
            with torch.no_grad():
                inputs = tokenizer(text_prompts, return_tensors="pt", padding=True)
                text_features = clip_model.get_text_features(**inputs)
                text_features = F.normalize(text_features, dim=-1)

            print(f"Initialized {self.num_classes} {temporal}-change prompts from CLIP")
            return text_features

        except Exception as e:
            print(f"Failed to load CLIP for {temporal} prompts: {e}, using random initialization")
            return torch.randn(self.num_classes, 512) * 0.02

    def get_prompts_A(self) -> torch.Tensor:
        """
        Get before-change semantic prompts.

        Returns:
            [K, D] tensor of normalized semantic prompts for before-change
        """
        prompts = self.base_prompts_A + self.prompt_offsets_A
        prompts = self.proj(prompts)
        return F.normalize(prompts, dim=-1)

    def get_prompts_B(self) -> torch.Tensor:
        """
        Get after-change semantic prompts.

        Returns:
            [K, D] tensor of normalized semantic prompts for after-change
        """
        prompts = self.base_prompts_B + self.prompt_offsets_B
        prompts = self.proj(prompts)
        return F.normalize(prompts, dim=-1)

    def get_prompts(self) -> torch.Tensor:
        """
        Backward compatible method - returns after-change prompts.

        Returns:
            [K, D] tensor of semantic prompts (same as get_prompts_B)
        """
        return self.get_prompts_B()

    def get_transition_embeddings(self) -> torch.Tensor:
        """
        Compute transition embeddings for all class pairs.

        Creates a [K, K, D] tensor where entry [i, j] represents the
        semantic embedding for "class i changed to class j".

        For diagonal entries (i == j, no change), uses the special
        no_change_embed instead.

        Returns:
            [K, K, D] tensor of transition embeddings
        """
        prompts_A = self.get_prompts_A()  # [K, D]
        prompts_B = self.get_prompts_B()  # [K, D]

        K, D = prompts_A.shape

        # Create all pairs: expand to [K, K, D] then concatenate
        A_expanded = prompts_A.unsqueeze(1).expand(K, K, D)  # [K, K, D]
        B_expanded = prompts_B.unsqueeze(0).expand(K, K, D)  # [K, K, D]
        pairs = torch.cat([A_expanded, B_expanded], dim=-1)  # [K, K, 2D]

        # Flatten for batch processing through transition encoder
        pairs_flat = pairs.view(K * K, D * 2)  # [K*K, 2D]
        transitions_flat = self.transition_encoder(pairs_flat)  # [K*K, D]
        transitions = transitions_flat.view(K, K, D)  # [K, K, D]

        # For diagonal (no change), use special embedding
        diag_mask = torch.eye(K, device=transitions.device, dtype=torch.bool)
        no_change_expanded = self.no_change_embed.expand(K, D)  # [K, D]
        transitions[diag_mask] = no_change_expanded

        return F.normalize(transitions, dim=-1)

    def get_transition_for_caption(
        self,
        sem_A: torch.Tensor,
        sem_B: torch.Tensor
    ) -> torch.Tensor:
        """
        Get transition embeddings based on predicted semantic maps.
        Used by caption decoder to attend to "what changed into what".

        Args:
            sem_A: [B, H, W] predicted before-change classes (argmax of logits)
            sem_B: [B, H, W] predicted after-change classes

        Returns:
            [B, H*W, D] transition embeddings for each spatial location
        """
        B, H, W = sem_A.shape
        device = sem_A.device

        transitions = self.get_transition_embeddings()  # [K, K, D]
        D = transitions.shape[-1]

        # Flatten spatial dims
        sem_A_flat = sem_A.view(B, -1)  # [B, H*W]
        sem_B_flat = sem_B.view(B, -1)  # [B, H*W]

        # Gather transition embeddings for each spatial location
        # transitions[sem_A[b,n], sem_B[b,n]] for each batch b and position n
        N = H * W
        trans_embeds = torch.zeros(B, N, D, device=device)

        for b in range(B):
            # Index into transition matrix for this batch
            trans_embeds[b] = transitions[sem_A_flat[b], sem_B_flat[b]]

        return trans_embeds

    def compute_similarity(
        self,
        features: torch.Tensor,
        prompts: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute similarity between features and prompts.

        Args:
            features: [..., D] features to compare
            prompts: [K, D] prompts to compare against
            normalize: Whether to normalize features

        Returns:
            [..., K] similarity scores
        """
        if normalize:
            features = F.normalize(features, dim=-1)

        # Compute cosine similarity scaled by learned temperature
        similarity = torch.matmul(features, prompts.t())
        similarity = similarity * self.logit_scale.exp()

        return similarity

    def forward(self) -> Dict[str, torch.Tensor]:
        """
        Return all prompts and transitions.

        Returns:
            dict with:
                - prompts_A: [K, D] before-change prompts
                - prompts_B: [K, D] after-change prompts
                - transitions: [K, K, D] transition embeddings
        """
        return {
            'prompts_A': self.get_prompts_A(),
            'prompts_B': self.get_prompts_B(),
            'transitions': self.get_transition_embeddings()
        }

    def get_class_name(self, class_idx: int) -> str:
        """Get class name for a given index."""
        return self.class_names[class_idx]

    def get_transition_description(self, class_a_idx: int, class_b_idx: int) -> str:
        """Get text description for a transition."""
        class_a = self.class_names[class_a_idx]
        class_b = self.class_names[class_b_idx]
        return self.templates['transition'].format(class_a=class_a, class_b=class_b)
