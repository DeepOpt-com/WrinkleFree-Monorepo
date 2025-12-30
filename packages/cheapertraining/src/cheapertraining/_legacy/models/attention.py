"""Multi-head attention with QK-norm and Grouped Query Attention (GQA).

Clean room implementation based on:
- MobileLLM-R1 paper (arXiv:2509.24945) - QK-norm for training stability
- Grouped Query Attention (Ainslie et al.) - efficient KV cache
- RoPE (Su et al.) - rotary position embeddings
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it doesn't require mean computation.
    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Compute RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    base: float = 500_000.0,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """Precompute RoPE sin/cos frequencies.

    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (500k for long context)
        device: Device to create tensors on

    Returns:
        Tuple of (cos, sin) tensors of shape (max_seq_len, dim)
    """
    assert dim % 2 == 0, "Head dimension must be even for RoPE"

    # Compute frequency bands: theta_i = base^(-2i/dim) for i in [0, dim/2)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Compute position indices
    positions = torch.arange(max_seq_len, device=device).float()

    # Outer product: (seq_len, dim/2)
    freqs = torch.outer(positions, inv_freq)

    # Expand to full dimension by repeating: (seq_len, dim)
    freqs = torch.cat([freqs, freqs], dim=-1)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return cos, sin


def apply_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Optional[Tensor] = None,
) -> Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
        cos: Cosine frequencies (seq_len, head_dim) or (1, seq_len, 1, head_dim)
        sin: Sine frequencies (seq_len, head_dim) or (1, seq_len, 1, head_dim)
        position_ids: Optional position indices for non-contiguous positions

    Returns:
        Tensor with RoPE applied
    """
    # Handle position_ids for non-contiguous sequences (e.g., KV cache)
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(2)  # (batch, seq, 1, dim)
        sin = sin[position_ids].unsqueeze(2)
    else:
        seq_len = x.size(1)
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)

    # Split into pairs and rotate
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Interleave rotated pairs
    x_rotated = torch.cat([-x2, x1], dim=-1)

    # Apply rotation
    return x * cos + x_rotated * sin


class MultiHeadAttention(nn.Module):
    """Multi-head attention with QK-norm and Grouped Query Attention (GQA).

    Features from MobileLLM-R1:
    - QK-norm: Apply RMSNorm to Q and K after projection for training stability
    - GQA: Use fewer KV heads than query heads for efficiency
    - RoPE: Rotary position embeddings for position encoding

    Reference: https://arxiv.org/abs/2509.24945
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        use_qk_norm: bool = True,
        dropout: float = 0.0,
        rope_base: float = 500_000.0,
        max_seq_len: int = 32_768,
        norm_eps: float = 1e-5,
    ):
        """Initialize multi-head attention.

        Args:
            embed_dim: Model embedding dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (for GQA)
            head_dim: Dimension per head (default: embed_dim // num_heads)
            use_qk_norm: Whether to apply RMSNorm to Q and K
            dropout: Attention dropout probability
            rope_base: Base frequency for RoPE
            max_seq_len: Maximum sequence length for RoPE precomputation
            norm_eps: Epsilon for normalization layers
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.use_qk_norm = use_qk_norm
        self.dropout = dropout

        # Ensure num_heads is divisible by num_kv_heads for GQA
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
        self.num_kv_groups = num_heads // num_kv_heads

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

        # QK-norm layers (applied after projection, before RoPE)
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        # Precompute RoPE frequencies
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        self._init_rope()

    def _init_rope(self):
        """Initialize RoPE frequencies."""
        cos, sin = precompute_rope_frequencies(
            self.head_dim,
            self.max_seq_len,
            self.rope_base,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

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
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional attention mask (batch, 1, seq_len, kv_seq_len)
            position_ids: Optional position indices (batch, seq_len)
            past_key_value: Optional cached KV for incremental decoding
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (output, optional_kv_cache)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape: (batch, seq, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK-norm (per-head normalization)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE to Q and K
        q = apply_rope(q, self.rope_cos, self.rope_sin, position_ids)
        k = apply_rope(k, self.rope_cos, self.rope_sin, position_ids)

        # Handle KV cache for incremental decoding
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)

        new_kv_cache = (k, v) if use_cache else None

        # Expand KV heads for GQA: repeat each KV head for its group of Q heads
        # (batch, kv_seq, num_kv_heads, head_dim) -> (batch, kv_seq, num_heads, head_dim)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # Transpose for attention: (batch, num_heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention for efficiency
        # This automatically selects FlashAttention2, Memory-Efficient, or Math backend
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None,  # Use causal mask if no mask provided
        )

        # Reshape back: (batch, seq, embed_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection
        output = self.o_proj(output)

        return output, new_kv_cache


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation.

    Uses gated linear units with SiLU (Swish) activation.
    Reference: GLU Variants Improve Transformer (Shazeer, 2020)
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        """Initialize feed-forward network.

        Args:
            embed_dim: Input/output embedding dimension
            hidden_dim: Intermediate hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with SwiGLU activation."""
        # SwiGLU: down(silu(gate(x)) * up(x))
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))
