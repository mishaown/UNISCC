"""
Multi-Level Caption Decoder for UniSCC v5.0

Caption decoder that attends to all pyramid levels and hierarchical prompts.

Key features:
    1. Multi-scale visual cross-attention (to all diff_pyramid levels)
    2. Hierarchical prompt cross-attention
    3. Change attention masking (focus on changed regions)
    4. Gated fusion of visual and semantic information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, List


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiLevelDecoderLayer(nn.Module):
    """
    Single decoder layer with multi-scale visual attention.

    Architecture:
        1. Self-attention (causal) on caption tokens
        2. Multi-scale visual cross-attention (one per pyramid level)
        3. Visual fusion (concatenate and project)
        4. Hierarchical prompt cross-attention
        5. Gated fusion of visual and semantic
        6. Feed-forward network
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_scales: int = 4
    ):
        super().__init__()

        self.d_model = d_model
        self.num_scales = num_scales
        self.scale_names = ['P2', 'P3', 'P4', 'P5'][:num_scales]

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Multi-scale visual attention (one per scale)
        self.visual_attns = nn.ModuleDict({
            name: nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for name in self.scale_names
        })

        # Visual fusion
        self.visual_fusion = nn.Linear(d_model * num_scales, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Prompt attention
        self.prompt_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(d_model)

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # Feed-forward
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
        visual_features: Dict[str, torch.Tensor],
        prompt_features: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoder layer forward pass.

        Args:
            x: [B, T, D] decoder input
            visual_features: dict of [B, N_i, D] per scale
            prompt_features: [4K, D] all hierarchical prompts stacked
            tgt_mask: [T, T] causal mask
            tgt_key_padding_mask: [B, T] padding mask

        Returns:
            [B, T, D] decoder output
        """
        B = x.shape[0]

        # 1. Self-attention
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Multi-scale visual attention
        visual_outs = []
        for name in self.scale_names:
            vis = visual_features[name]
            out, _ = self.visual_attns[name](x, vis, vis)
            visual_outs.append(out)

        # Fuse visual outputs
        visual_cat = torch.cat(visual_outs, dim=-1)
        visual_fused = self.visual_fusion(visual_cat)
        x = self.norm2(x + self.dropout(visual_fused))

        # 3. Prompt attention
        # Expand prompts for batch
        prompts_expanded = prompt_features.unsqueeze(0).expand(B, -1, -1)
        prompt_out, _ = self.prompt_attn(x, prompts_expanded, prompts_expanded)

        # 4. Gated fusion
        gate_input = torch.cat([visual_fused, prompt_out], dim=-1)
        gate_weights = self.gate(gate_input)
        fused = gate_weights * visual_fused + (1 - gate_weights) * prompt_out
        x = self.norm3(x + self.dropout(fused))

        # 5. FFN
        x = self.norm4(x + self.ffn(x))

        return x


class MultiLevelCaptionDecoder(nn.Module):
    """
    Multi-Level Caption Decoder for UniSCC v5.0.

    Attends to all pyramid levels and hierarchical prompts for
    comprehensive change captioning.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension (512)
        nhead: Number of attention heads
        num_layers: Number of decoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        max_length: Maximum caption length
        num_scales: Number of pyramid levels (4)
        prompt_dim: Dimension of prompts (256)
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
        num_scales: int = 4,
        prompt_dim: int = 256,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.num_scales = num_scales
        self.scale_names = ['P2', 'P3', 'P4', 'P5'][:num_scales]
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)

        # Per-scale visual projections (256 -> 512)
        self.visual_projs = nn.ModuleDict({
            name: nn.Linear(256, d_model)
            for name in self.scale_names
        })

        # Prompt projection (256 -> 512)
        self.prompt_proj = nn.Linear(prompt_dim, d_model)

        # Decoder layers
        self.layers = nn.ModuleList([
            MultiLevelDecoderLayer(d_model, nhead, dim_feedforward, dropout, num_scales)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _prepare_visual_features(
        self,
        diff_pyramid: Dict[str, torch.Tensor],
        change_attention: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Convert spatial features to sequences with attention masking.

        Args:
            diff_pyramid: dict of [B, 256, Hi, Wi] per scale
            change_attention: [B, 1, 256, 256] attention map

        Returns:
            dict of [B, Ni, d_model] per scale
        """
        visual_features = {}

        for name in self.scale_names:
            feat = diff_pyramid[name]  # [B, 256, Hi, Wi]
            B, C, H, W = feat.shape

            # Flatten to sequence
            feat_seq = feat.flatten(2).transpose(1, 2)  # [B, H*W, 256]

            # Downsample attention mask to match scale
            attn = F.interpolate(
                change_attention,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )  # [B, 1, H, W]
            attn_seq = attn.flatten(2).transpose(1, 2)  # [B, H*W, 1]

            # Apply attention mask
            feat_seq = feat_seq * attn_seq

            # Project to d_model
            feat_seq = self.visual_projs[name](feat_seq)  # [B, H*W, d_model]

            visual_features[name] = feat_seq

        return visual_features

    def _prepare_prompts(
        self,
        hierarchical_prompts: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Stack all prompts into single sequence.

        Args:
            hierarchical_prompts: dict of [K, 256] per scale

        Returns:
            [4K, d_model] stacked prompts
        """
        all_prompts = []
        for name in self.scale_names:
            prompts = hierarchical_prompts[f'prompts_{name}']  # [K, 256]
            prompts = self.prompt_proj(prompts)  # [K, d_model]
            all_prompts.append(prompts)

        all_prompts = torch.cat(all_prompts, dim=0)  # [4K, d_model]
        return all_prompts

    def forward(
        self,
        diff_pyramid: Dict[str, torch.Tensor],
        hierarchical_prompts: Dict[str, torch.Tensor],
        change_attention: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        caption_lengths: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Multi-level caption generation.

        Args:
            diff_pyramid: dict of difference features per scale
            hierarchical_prompts: dict of prompts per scale
            change_attention: [B, 1, 256, 256] spatial attention map
            captions: [B, T] caption tokens (for training)
            caption_lengths: [B] caption lengths
            teacher_forcing: Whether to use teacher forcing

        Returns:
            [B, T, V] caption logits (training) or [B, T] tokens (inference)
        """
        # Prepare visual features
        visual_features = self._prepare_visual_features(diff_pyramid, change_attention)

        # Prepare prompts
        prompt_features = self._prepare_prompts(hierarchical_prompts)

        if teacher_forcing and captions is not None:
            return self._forward_train(
                visual_features, prompt_features, captions, caption_lengths
            )
        else:
            return self._forward_inference(visual_features, prompt_features)

    def _forward_train(
        self,
        visual_features: Dict[str, torch.Tensor],
        prompt_features: torch.Tensor,
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

        # Apply decoder layers
        hidden = token_embeds
        for layer in self.layers:
            hidden = layer(
                hidden, visual_features, prompt_features,
                causal_mask, padding_mask
            )

        # Output projection
        logits = self.output_proj(hidden)
        return logits

    def _forward_inference(
        self,
        visual_features: Dict[str, torch.Tensor],
        prompt_features: torch.Tensor
    ) -> torch.Tensor:
        """Inference with autoregressive generation."""
        B = next(iter(visual_features.values())).shape[0]
        device = next(iter(visual_features.values())).device

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
                hidden = layer(hidden, visual_features, prompt_features, causal_mask)

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
        diff_pyramid: Dict[str, torch.Tensor],
        hierarchical_prompts: Dict[str, torch.Tensor],
        change_attention: torch.Tensor,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Generate captions."""
        max_length = max_length or self.max_length

        # Prepare features
        visual_features = self._prepare_visual_features(diff_pyramid, change_attention)
        prompt_features = self._prepare_prompts(hierarchical_prompts)

        return self._forward_inference(visual_features, prompt_features)


class EfficientMultiLevelDecoder(nn.Module):
    """
    Memory-efficient Multi-Level Decoder.

    Uses a single shared visual attention with scale embeddings
    instead of per-scale attention modules.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_length: int = 50,
        num_scales: int = 4,
        prompt_dim: int = 256,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.num_scales = num_scales
        self.scale_names = ['P2', 'P3', 'P4', 'P5'][:num_scales]
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)

        # Scale embeddings
        self.scale_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, d_model))
            for name in self.scale_names
        })
        for name in self.scale_names:
            nn.init.trunc_normal_(self.scale_embeddings[name], std=0.02)

        # Shared projections
        self.visual_proj = nn.Linear(256, d_model)
        self.prompt_proj = nn.Linear(prompt_dim, d_model)

        # Standard transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        diff_pyramid: Dict[str, torch.Tensor],
        hierarchical_prompts: Dict[str, torch.Tensor],
        change_attention: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        caption_lengths: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> torch.Tensor:
        """Efficient forward pass."""
        # Concatenate all visual features with scale embeddings
        all_visual = []
        for name in self.scale_names:
            feat = diff_pyramid[name]
            B, C, H, W = feat.shape

            # Flatten
            feat_seq = feat.flatten(2).transpose(1, 2)  # [B, H*W, 256]

            # Project
            feat_seq = self.visual_proj(feat_seq)  # [B, H*W, d_model]

            # Add scale embedding
            feat_seq = feat_seq + self.scale_embeddings[name]

            all_visual.append(feat_seq)

        # Concatenate all scales
        memory = torch.cat(all_visual, dim=1)  # [B, sum(N_i), d_model]

        # Add prompts to memory
        all_prompts = []
        for name in self.scale_names:
            prompts = hierarchical_prompts[f'prompts_{name}']
            prompts = self.prompt_proj(prompts)
            all_prompts.append(prompts)
        prompts_cat = torch.cat(all_prompts, dim=0)  # [4K, d_model]
        prompts_expanded = prompts_cat.unsqueeze(0).expand(memory.shape[0], -1, -1)
        memory = torch.cat([memory, prompts_expanded], dim=1)

        if teacher_forcing and captions is not None:
            B, T = captions.shape
            device = captions.device

            token_embeds = self.token_embed(captions)
            token_embeds = self.pos_encoding(token_embeds)

            # Causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

            # Padding mask
            padding_mask = None
            if caption_lengths is not None:
                padding_mask = torch.arange(T, device=device).expand(B, T) >= caption_lengths.unsqueeze(1)

            hidden = self.decoder(
                token_embeds, memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask
            )
            return self.output_proj(hidden)
        else:
            # Autoregressive generation
            B = memory.shape[0]
            device = memory.device
            generated = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)

            for _ in range(self.max_length - 1):
                token_embeds = self.token_embed(generated)
                token_embeds = self.pos_encoding(token_embeds)

                T = generated.shape[1]
                causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

                hidden = self.decoder(token_embeds, memory, tgt_mask=causal_mask)
                logits = self.output_proj(hidden[:, -1:, :])
                next_token = logits.argmax(dim=-1)

                generated = torch.cat([generated, next_token], dim=1)

                if (next_token == self.eos_token_id).all():
                    break

            return generated
