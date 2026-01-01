"""Transformer block implementation.

Clean room implementation based on:
- MobileLLM-R1 paper (arXiv:2509.24945)
- Pre-normalization (GPT-2 style) for training stability
"""

from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from cheapertraining._legacy.models.attention import (
    MultiHeadAttention,
    FeedForward,
    RMSNorm,
)
from cheapertraining._legacy.models.checkpoint_utils import checkpoint_fn
from cheapertraining._legacy.models.config import MobileLLMConfig


class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization.

    Architecture:
        x -> norm1 -> attention -> + -> norm2 -> ffn -> +
             |______________________|     |______________|

    Uses pre-normalization (norm before attention/ffn) for training stability.
    """

    def __init__(
        self,
        config: MobileLLMConfig,
        layer_idx: int,
    ):
        """Initialize transformer block.

        Args:
            config: Model configuration
            layer_idx: Index of this layer (for debugging/profiling)
        """
        super().__init__()

        self.layer_idx = layer_idx

        # Pre-attention normalization
        self.attention_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)

        # Self-attention
        self.attention = MultiHeadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            use_qk_norm=config.use_qk_norm,
            dropout=config.attention_dropout,
            rope_base=config.rope_base,
            max_seq_len=config.max_seq_len,
            norm_eps=config.norm_eps,
        )

        # Pre-FFN normalization
        self.ffn_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)

        # Feed-forward network
        self.ffn = FeedForward(
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        # Gradient checkpointing settings (set via model.gradient_checkpointing_enable())
        self.gradient_checkpointing = False
        self.checkpointing_mode = "quantized"

    def _forward_impl(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
        past_key_value: Optional[Tuple[Tensor, Tensor]],
        use_cache: bool,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Internal forward implementation (used by checkpointing)."""
        # Self-attention with residual connection
        residual = x
        x = self.attention_norm(x)
        x, new_kv_cache = self.attention(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = residual + x

        # Feed-forward with residual connection
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, new_kv_cache

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cached KV from previous steps
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (output, optional_kv_cache)
        """
        # Use gradient checkpointing during training if enabled
        # Note: Checkpointing is incompatible with KV cache (use_cache=True)
        if self.gradient_checkpointing and self.training and not use_cache:
            # Create a wrapper that works with checkpoint_fn
            def create_forward_fn(
                attn_norm, attention, ffn_norm, ffn, attn_mask, pos_ids, past_kv, cache
            ):
                def forward_fn(hidden_states):
                    # Self-attention with residual connection
                    residual = hidden_states
                    hidden_states = attn_norm(hidden_states)
                    hidden_states, kv_cache = attention(
                        hidden_states,
                        attention_mask=attn_mask,
                        position_ids=pos_ids,
                        past_key_value=past_kv,
                        use_cache=cache,
                    )
                    hidden_states = residual + hidden_states

                    # Feed-forward with residual connection
                    residual = hidden_states
                    hidden_states = ffn_norm(hidden_states)
                    hidden_states = ffn(hidden_states)
                    hidden_states = residual + hidden_states

                    return hidden_states
                return forward_fn

            forward_fn = create_forward_fn(
                self.attention_norm,
                self.attention,
                self.ffn_norm,
                self.ffn,
                attention_mask,
                position_ids,
                past_key_value,
                use_cache,
            )
            x = checkpoint_fn(forward_fn, x, mode=self.checkpointing_mode)
            return x, None  # No KV cache when checkpointing
        else:
            return self._forward_impl(x, attention_mask, position_ids, past_key_value, use_cache)


class TransformerDecoder(nn.Module):
    """Stack of transformer blocks for decoder-only architecture."""

    def __init__(self, config: MobileLLMConfig):
        """Initialize transformer decoder.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[list]]:
        """Forward pass through all transformer layers.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_values: Optional list of cached KV per layer
            use_cache: Whether to return updated KV caches

        Returns:
            Tuple of (output, optional_list_of_kv_caches)
        """
        new_kv_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            x, new_kv = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if use_cache:
                new_kv_caches.append(new_kv)

        return x, new_kv_caches
