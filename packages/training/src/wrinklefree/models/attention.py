"""BitNet Multi-Head Attention with SubLN and RoPE."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wrinklefree.models.bitlinear import BitLinear
from wrinklefree.models.subln import SubLN


def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    theta: float = 500000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Precompute rotary position embeddings (RoPE) frequencies.

    Args:
        dim: Dimension of the embeddings (head_dim)
        seq_len: Maximum sequence length
        theta: Base for the exponential frequency calculation
        device: Device to create tensors on

    Returns:
        Complex tensor of shape (seq_len, dim//2) containing rotation frequencies
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i * theta)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        xq: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim//2)

    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Reshape to complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Apply rotation
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim//2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match query heads for GQA.

    Args:
        x: Tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        n_rep: Number of times to repeat each KV head

    Returns:
        Tensor of shape (batch, seq_len, num_kv_heads * n_rep, head_dim)
    """
    if n_rep == 1:
        return x
    batch, seq_len, num_kv_heads, head_dim = x.shape
    x = x.unsqueeze(3).expand(batch, seq_len, num_kv_heads, n_rep, head_dim)
    return x.reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)


class BitNetAttention(nn.Module):
    """
    BitNet Multi-Head Attention with SubLN.

    Architecture:
    - Q, K, V projections (BitLinear)
    - Rotary Position Embeddings (RoPE)
    - Grouped-Query Attention (GQA) support
    - SubLN before output projection (key BitDistill modification)
    - Output projection (BitLinear)

    Args:
        hidden_size: Model hidden dimension
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA, None = MHA)
        head_dim: Dimension per head (default: hidden_size // num_attention_heads)
        rope_theta: Base for RoPE frequency calculation
        max_position_embeddings: Maximum sequence length for RoPE cache
        attention_dropout: Dropout probability for attention weights
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        rope_theta: float = 500000.0,
        max_position_embeddings: int = 4096,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads or num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        # Projections
        self.q_proj = BitLinear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = BitLinear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = BitLinear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        # SubLN before output projection (key BitDistill modification)
        self.subln = SubLN(self.num_heads * self.head_dim)
        self.o_proj = BitLinear(self.num_heads * self.head_dim, hidden_size, bias=False)

        # RoPE cache
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, max_position_embeddings, rope_theta),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for attention layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch, 1, seq_len, seq_len)
            position_ids: Optional position indices for RoPE
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output tensor, attention weights if requested)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        # Use -1 instead of num_heads to infer actual local heads from tensor size.
        # This is the TorchTitan pattern: TP may have sharded the projection outputs,
        # so we dynamically compute local heads from the actual tensor dimensions.
        # See: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model/model.py
        query_states = query_states.view(batch_size, seq_len, -1, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, -1, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, -1, self.head_dim)

        # Apply RoPE
        if position_ids is None:
            freqs_cis = self.freqs_cis[:seq_len]
        else:
            freqs_cis = self.freqs_cis[position_ids]
        query_states, key_states = apply_rotary_emb(query_states, key_states, freqs_cis)

        # Transpose for attention: (batch, heads, seq, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Repeat KV heads for GQA
        key_states = repeat_kv(key_states.transpose(1, 2), self.num_kv_groups).transpose(1, 2)
        value_states = repeat_kv(value_states.transpose(1, 2), self.num_kv_groups).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Dropout
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        # Attention output
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and apply SubLN before output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.subln(attn_output)  # SubLN here!
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


class BitNetFlashAttention(BitNetAttention):
    """
    BitNet Attention using Flash Attention 2 for efficiency.

    Falls back to standard attention if Flash Attention is not available.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with Flash Attention when possible.

        Uses SDPA (scaled_dot_product_attention) for performance when available.
        Falls back to standard attention when attention weights are requested.
        """
        # Use SDPA when available and attention weights not needed
        if not output_attentions and hasattr(F, "scaled_dot_product_attention"):
            return self._forward_flash(hidden_states, attention_mask, position_ids)

        # Fall back to standard attention
        return super().forward(hidden_states, attention_mask, position_ids, output_attentions)

    def _forward_flash(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass using PyTorch's scaled_dot_product_attention."""
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        # Use -1 instead of num_heads to infer actual local heads from tensor size.
        # This is the TorchTitan pattern: TP may have sharded the projection outputs,
        # so we dynamically compute local heads from the actual tensor dimensions.
        # See: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model/model.py
        query_states = query_states.view(batch_size, seq_len, -1, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, -1, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, -1, self.head_dim)

        # Apply RoPE
        if position_ids is None:
            freqs_cis = self.freqs_cis[:seq_len]
        else:
            freqs_cis = self.freqs_cis[position_ids]
        query_states, key_states = apply_rotary_emb(query_states, key_states, freqs_cis)

        # Transpose for attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Repeat KV for GQA
        key_states = repeat_kv(key_states.transpose(1, 2), self.num_kv_groups).transpose(1, 2)
        value_states = repeat_kv(value_states.transpose(1, 2), self.num_kv_groups).transpose(1, 2)

        # Flash Attention
        # Always use is_causal=True for autoregressive LM (causal language modeling).
        # This avoids dtype issues with DTensor (tensor parallelism) when explicit masks
        # are passed. The explicit attention_mask from the model is a causal mask anyway.
        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,  # Don't pass mask - use is_causal instead
            dropout_p=dropout_p,
            is_causal=True,  # Always use causal for autoregressive LM
        )

        # Reshape and apply SubLN
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.subln(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output, None
