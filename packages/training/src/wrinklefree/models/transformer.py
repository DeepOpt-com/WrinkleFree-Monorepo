"""BitNet Transformer block and decoder layer."""

from typing import Optional

import torch
import torch.nn as nn

from wrinklefree.models.attention import BitNetAttention, BitNetFlashAttention
from wrinklefree.models.ffn import BitNetFFN
from wrinklefree.models.subln import RMSNorm


class BitNetDecoderLayer(nn.Module):
    """
    BitNet Transformer decoder layer.

    Architecture (Pre-LN style):
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))

    Where Attention and FFN contain SubLN before their output projections.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of KV heads for GQA (None = MHA)
        head_dim: Dimension per head
        rope_theta: Base for RoPE frequencies
        max_position_embeddings: Max sequence length for RoPE
        attention_dropout: Dropout for attention weights
        hidden_act: FFN activation function
        use_flash_attention: Whether to use Flash Attention
        layer_idx: Layer index (for debugging)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        rope_theta: float = 500000.0,
        max_position_embeddings: int = 4096,
        attention_dropout: float = 0.0,
        hidden_act: str = "relu2",
        use_flash_attention: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        # Input layernorm (before attention)
        self.input_layernorm = RMSNorm(hidden_size)

        # Self-attention
        attention_cls = BitNetFlashAttention if use_flash_attention else BitNetAttention
        self.self_attn = attention_cls(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            attention_dropout=attention_dropout,
        )

        # Post-attention layernorm (before FFN)
        self.post_attention_layernorm = RMSNorm(hidden_size)

        # Feed-forward network
        self.mlp = BitNetFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through decoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output tensor, attention weights if requested)
        """
        residual = hidden_states

        # Pre-LN + Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # Pre-LN + FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights


class BitNetTransformer(nn.Module):
    """
    Stack of BitNet decoder layers.

    This is the core transformer component without embeddings or LM head.

    Args:
        num_layers: Number of decoder layers
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of KV heads for GQA
        head_dim: Dimension per head
        rope_theta: Base for RoPE frequencies
        max_position_embeddings: Max sequence length
        attention_dropout: Dropout for attention
        hidden_act: FFN activation
        use_flash_attention: Whether to use Flash Attention
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        rope_theta: float = 500000.0,
        max_position_embeddings: int = 4096,
        attention_dropout: float = 0.0,
        hidden_act: str = "relu2",
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BitNetDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                attention_dropout=attention_dropout,
                hidden_act=hidden_act,
                use_flash_attention=use_flash_attention,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list]:
        """
        Forward pass through all decoder layers.

        Args:
            hidden_states: Input tensor from embeddings
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            output_attentions: Whether to collect attention weights
            output_hidden_states: Whether to collect hidden states

        Returns:
            Dictionary with:
                - last_hidden_state: Final hidden states
                - hidden_states: List of hidden states per layer (if requested)
                - attentions: List of attention weights per layer (if requested)
        """
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions.append(attn_weights)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }
