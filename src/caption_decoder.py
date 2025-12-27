"""
Semantic-Guided Caption Decoder

Caption decoder that uses shared semantic prompts for generation.
The semantic prompts create alignment between change detection classes
and caption vocabulary.

v3.0 Addition: TransitionCaptionDecoder
- Attends to transition embeddings for "what changed into what"
- Gated fusion of visual and transition attention
- Enables generating descriptions like "vegetation was replaced by building"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SemanticGuidedDecoderLayer(nn.Module):
    """
    Decoder layer with semantic prompt guidance.

    Attention flow:
    1. Self-attention (causal) on caption tokens
    2. Cross-attention to visual features
    3. Cross-attention to semantic prompts (shared space)
    4. Feed-forward network
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention to visual features
        self.visual_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # Cross-attention to semantic prompts
        self.semantic_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        visual_memory: torch.Tensor,
        semantic_prompts: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] decoder input
            visual_memory: [B, N, D] visual features
            semantic_prompts: [K, D] semantic prompts (shared with change head)
            tgt_mask: [T, T] causal mask
            tgt_key_padding_mask: [B, T] padding mask

        Returns:
            [B, T, D] decoder output
        """
        B = x.shape[0]

        # 1. Masked self-attention
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Cross-attention to visual features
        visual_out, _ = self.visual_attn(x, visual_memory, visual_memory)
        x = self.norm2(x + self.dropout(visual_out))

        # 3. Cross-attention to semantic prompts
        # Expand prompts for batch: [K, D] -> [B, K, D]
        prompts_expanded = semantic_prompts.unsqueeze(0).expand(B, -1, -1)
        semantic_out, _ = self.semantic_attn(x, prompts_expanded, prompts_expanded)
        x = self.norm3(x + self.dropout(semantic_out))

        # 4. Feed-forward
        x = self.norm4(x + self.ffn(x))

        return x


class SemanticCaptionDecoder(nn.Module):
    """
    Caption decoder with semantic prompt guidance.

    Uses the shared semantic space from LSP to:
    1. Attend to semantic class prompts during decoding
    2. Create alignment between class names and caption words
    3. Generate captions that naturally describe semantic changes
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_length: int = 50,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)

        # Visual feature projection
        self.visual_proj = nn.Linear(d_model, d_model)

        # Decoder layers with semantic guidance
        self.layers = nn.ModuleList([
            SemanticGuidedDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_prompts: torch.Tensor,
        captions: torch.Tensor = None,
        caption_lengths: torch.Tensor = None,
        teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            visual_features: [B, C, H, W] or [B, N, D] visual features
            semantic_prompts: [K, D] semantic prompts from LSP
            captions: [B, T] caption tokens (for training)
            caption_lengths: [B] caption lengths
            teacher_forcing: Use teacher forcing

        Returns:
            [B, T, V] vocabulary logits (training) or [B, T] tokens (inference)
        """
        # Handle spatial features
        if visual_features.dim() == 4:
            B, C, H, W = visual_features.shape
            visual_features = visual_features.flatten(2).transpose(1, 2)

        # Project visual features
        memory = self.visual_proj(visual_features)

        if teacher_forcing and captions is not None:
            return self._forward_train(memory, semantic_prompts, captions, caption_lengths)
        else:
            return self._forward_inference(memory, semantic_prompts)

    def _forward_train(
        self,
        memory: torch.Tensor,
        semantic_prompts: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """Training forward with teacher forcing."""
        B, T = captions.shape
        device = captions.device

        # Embed tokens
        token_embeds = self.token_embed(captions)
        token_embeds = self.pos_encoding(token_embeds)

        # Causal mask
        causal_mask = self._generate_causal_mask(T, device)

        # Padding mask
        padding_mask = None
        if caption_lengths is not None:
            padding_mask = torch.arange(T, device=device).expand(B, T) >= caption_lengths.unsqueeze(1)

        # Apply decoder layers with semantic guidance
        hidden = token_embeds
        for layer in self.layers:
            hidden = layer(hidden, memory, semantic_prompts, causal_mask, padding_mask)

        # Output projection
        logits = self.output_proj(hidden)
        return logits

    def _forward_inference(
        self,
        memory: torch.Tensor,
        semantic_prompts: torch.Tensor
    ) -> torch.Tensor:
        """Inference with autoregressive generation."""
        B = memory.shape[0]
        device = memory.device

        # Start with BOS token
        generated = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)

        for _ in range(self.max_length - 1):
            # Embed tokens
            token_embeds = self.token_embed(generated)
            token_embeds = self.pos_encoding(token_embeds)

            # Causal mask
            T = generated.shape[1]
            causal_mask = self._generate_causal_mask(T, device)

            # Apply decoder layers
            hidden = token_embeds
            for layer in self.layers:
                hidden = layer(hidden, memory, semantic_prompts, causal_mask)

            # Get next token
            logits = self.output_proj(hidden[:, -1:, :])
            next_token = logits.argmax(dim=-1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == self.eos_token_id).all():
                break

        return generated

    def generate(
        self,
        visual_features: torch.Tensor,
        semantic_prompts: torch.Tensor,
        max_length: int = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate captions."""
        max_length = max_length or self.max_length

        # Handle spatial features
        if visual_features.dim() == 4:
            visual_features = visual_features.flatten(2).transpose(1, 2)

        memory = self.visual_proj(visual_features)

        return self._forward_inference(memory, semantic_prompts)


# Backward compatibility alias
ChangeGuidedCaptionDecoder = SemanticCaptionDecoder


class TransitionGuidedDecoderLayer(nn.Module):
    """
    Decoder layer with transition-aware semantic guidance.

    Attention flow:
    1. Self-attention (causal) on caption tokens
    2. Cross-attention to visual features
    3. Cross-attention to TRANSITION embeddings (key innovation)
    4. Gated fusion of visual and transition attention
    5. Feed-forward network

    The transition attention allows the decoder to directly attend to
    "what changed from what to what" information.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention to visual features
        self.visual_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # Cross-attention to transition embeddings
        self.transition_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)

        # Gating between visual and transition
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        visual_memory: torch.Tensor,
        transition_memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, D] decoder input
            visual_memory: [B, N, D] visual features (fused A+B features)
            transition_memory: [B, M, D] transition embeddings for changed regions
            tgt_mask: [T, T] causal mask
            tgt_key_padding_mask: [B, T] padding mask

        Returns:
            [B, T, D] decoder output
        """
        # 1. Self-attention
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Visual cross-attention
        visual_out, _ = self.visual_attn(x, visual_memory, visual_memory)

        # 3. Transition cross-attention
        trans_out, _ = self.transition_attn(x, transition_memory, transition_memory)

        # 4. Gated fusion of visual and transition
        gate_input = torch.cat([visual_out, trans_out], dim=-1)
        gate_weights = self.gate(gate_input)
        fused = gate_weights * visual_out + (1 - gate_weights) * trans_out

        x = self.norm2(x + self.dropout(fused))

        # 5. FFN
        x = self.norm4(x + self.ffn(x))

        return x


class TransitionCaptionDecoder(nn.Module):
    """
    Caption decoder with transition-aware semantic guidance.

    Key difference from SemanticCaptionDecoder:
    - Receives transition embeddings based on predicted (sem_A, sem_B) pairs
    - Can generate descriptions like "X was replaced by Y" by attending to
      the transition embedding for (X, Y)
    - Uses gated fusion to balance visual and transition information

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of decoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        max_length: Maximum caption length
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sentence token ID
        eos_token_id: End of sentence token ID
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_length: int = 50,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)

        # Visual feature projection
        self.visual_proj = nn.Linear(d_model, d_model)

        # Transition embedding projection
        self.transition_proj = nn.Linear(d_model, d_model)

        # Decoder layers with transition guidance
        self.layers = nn.ModuleList([
            TransitionGuidedDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        visual_features: torch.Tensor,
        transition_embeddings: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        caption_lengths: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with transition embeddings.

        Args:
            visual_features: [B, D, H, W] or [B, N, D] fused visual features
            transition_embeddings: [B, H*W, D] per-pixel transition embeddings
            captions: [B, T] caption tokens (for training)
            caption_lengths: [B] caption lengths
            teacher_forcing: Use teacher forcing

        Returns:
            [B, T, V] vocabulary logits (training) or [B, T] tokens (inference)
        """
        # Handle spatial features
        if visual_features.dim() == 4:
            B, C, H, W = visual_features.shape
            visual_features = visual_features.flatten(2).transpose(1, 2)

        # Project visual features
        memory = self.visual_proj(visual_features)

        # Project transition embeddings
        trans_memory = self.transition_proj(transition_embeddings)

        if teacher_forcing and captions is not None:
            return self._forward_train(memory, trans_memory, captions, caption_lengths)
        else:
            return self._forward_inference(memory, trans_memory)

    def _forward_train(
        self,
        memory: torch.Tensor,
        trans_memory: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Training forward with teacher forcing."""
        B, T = captions.shape
        device = captions.device

        # Embed tokens
        token_embeds = self.token_embed(captions)
        token_embeds = self.pos_encoding(token_embeds)

        # Causal mask
        causal_mask = self._generate_causal_mask(T, device)

        # Padding mask
        padding_mask = None
        if caption_lengths is not None:
            padding_mask = torch.arange(T, device=device).expand(B, T) >= caption_lengths.unsqueeze(1)

        # Apply decoder layers with transition guidance
        hidden = token_embeds
        for layer in self.layers:
            hidden = layer(hidden, memory, trans_memory, causal_mask, padding_mask)

        # Output projection
        logits = self.output_proj(hidden)
        return logits

    def _forward_inference(
        self,
        memory: torch.Tensor,
        trans_memory: torch.Tensor
    ) -> torch.Tensor:
        """Inference with autoregressive generation."""
        B = memory.shape[0]
        device = memory.device

        # Start with BOS token
        generated = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)

        for _ in range(self.max_length - 1):
            # Embed tokens
            token_embeds = self.token_embed(generated)
            token_embeds = self.pos_encoding(token_embeds)

            # Causal mask
            T = generated.shape[1]
            causal_mask = self._generate_causal_mask(T, device)

            # Apply decoder layers
            hidden = token_embeds
            for layer in self.layers:
                hidden = layer(hidden, memory, trans_memory, causal_mask)

            # Get next token
            logits = self.output_proj(hidden[:, -1:, :])
            next_token = logits.argmax(dim=-1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == self.eos_token_id).all():
                break

        return generated

    def generate(
        self,
        visual_features: torch.Tensor,
        transition_embeddings: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate captions with transition awareness."""
        max_length = max_length or self.max_length

        # Handle spatial features
        if visual_features.dim() == 4:
            visual_features = visual_features.flatten(2).transpose(1, 2)

        memory = self.visual_proj(visual_features)
        trans_memory = self.transition_proj(transition_embeddings)

        return self._forward_inference(memory, trans_memory)
