"""CPU fallback implementations for vllm._custom_ops.

These are PyTorch-native implementations that work on CPU without CUDA.
"""

import torch
import torch.nn.functional as F


def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """SiLU activation with gating: out = silu(x[:d]) * x[d:]."""
    d = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :d]) * x[..., d:])


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """GELU activation with gating: out = gelu(x[:d]) * x[d:]."""
    d = x.shape[-1] // 2
    out.copy_(F.gelu(x[..., :d]) * x[..., d:])


def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """GELU-tanh activation with gating."""
    d = x.shape[-1] // 2
    out.copy_(F.gelu(x[..., :d], approximate="tanh") * x[..., d:])


def rms_norm(
    out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out.copy_(x_normed * weight)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """Fused residual add + RMS norm (in-place)."""
    x.add_(residual)
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    x.copy_(x_normed * weight)


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    """Apply rotary positional embeddings to query and key tensors.

    This is a simplified CPU implementation.
    """
    # Extract cos and sin from cache based on positions
    # cos_sin_cache shape: [max_seq_len, head_size]
    # positions shape: [batch_size] or [batch_size, seq_len]

    positions_flat = positions.flatten()
    cos = cos_sin_cache[positions_flat, :head_size // 2]
    sin = cos_sin_cache[positions_flat, head_size // 2:]

    # Reshape for broadcasting
    # query/key shape: [batch, seq, num_heads, head_size] or similar
    original_shape = query.shape

    if len(original_shape) == 3:
        # [batch*seq, num_heads, head_size]
        batch_seq, num_heads, _ = query.shape
        cos = cos.view(batch_seq, 1, -1)
        sin = sin.view(batch_seq, 1, -1)
    elif len(original_shape) == 4:
        batch, seq, num_heads, _ = query.shape
        cos = cos.view(batch, seq, 1, -1)
        sin = sin.view(batch, seq, 1, -1)
    else:
        # Fallback - just reshape to match
        cos = cos.view(*positions.shape, 1, -1)
        sin = sin.view(*positions.shape, 1, -1)

    # Split head_size into two halves
    q1, q2 = query[..., :head_size // 2], query[..., head_size // 2:head_size]
    k1, k2 = key[..., :head_size // 2], key[..., head_size // 2:head_size]

    # Apply rotation
    if is_neox:
        # GPT-NeoX style: interleaved
        query[..., :head_size // 2] = q1 * cos - q2 * sin
        query[..., head_size // 2:head_size] = q2 * cos + q1 * sin
        key[..., :head_size // 2] = k1 * cos - k2 * sin
        key[..., head_size // 2:head_size] = k2 * cos + k1 * sin
    else:
        # GPT-J style: separate halves
        query[..., :head_size // 2] = q1 * cos - q2 * sin
        query[..., head_size // 2:head_size] = q1 * sin + q2 * cos
        key[..., :head_size // 2] = k1 * cos - k2 * sin
        key[..., head_size // 2:head_size] = k1 * sin + k2 * cos


# Alias for ops module pattern
class ops:
    """Namespace for vllm ops."""
    silu_and_mul = staticmethod(silu_and_mul)
    gelu_and_mul = staticmethod(gelu_and_mul)
    gelu_tanh_and_mul = staticmethod(gelu_tanh_and_mul)
    rms_norm = staticmethod(rms_norm)
    fused_add_rms_norm = staticmethod(fused_add_rms_norm)
    rotary_embedding = staticmethod(rotary_embedding)
