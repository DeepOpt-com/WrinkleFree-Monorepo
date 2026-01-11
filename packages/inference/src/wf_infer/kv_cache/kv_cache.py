"""KV cache implementation for BitNet CPU inference.

Supports multiple precision levels for memory-quality tradeoffs:
- BF16: Default, highest quality
- FP8 E4M3: 50% memory savings, minimal quality loss
- INT8: 50% memory savings, symmetric quantization

For long context (>4K tokens), quantized KV cache enables:
- 2x more tokens in same memory
- Acceptable quality for most tasks

References:
- https://docs.sglang.io/advanced_features/quantized_kv_cache.html
- https://arxiv.org/abs/2208.07339 (LLM.int8())
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

class KVCacheDtype(Enum):
    """Supported KV cache data types."""
    BF16 = "bfloat16"
    FP16 = "float16"
    FP32 = "float32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"


@dataclass
class KVCacheConfig:
    """Configuration for KV cache."""

    max_seq_len: int = 4096
    """Maximum sequence length (context window)."""

    num_layers: int = 32
    """Number of transformer layers."""

    num_heads: int = 32
    """Number of attention heads."""

    head_dim: int = 128
    """Dimension per attention head."""

    dtype: KVCacheDtype = KVCacheDtype.INT8
    """Data type for KV cache storage. INT8 default for 50% memory savings."""

    page_size: int = 16
    """Token page size for paged attention."""

    @property
    def hidden_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def bytes_per_token_per_layer(self) -> int:
        """Memory per token per layer (K + V)."""
        if self.dtype in (KVCacheDtype.BF16, KVCacheDtype.FP16):
            bytes_per_elem = 2
        elif self.dtype == KVCacheDtype.FP32:
            bytes_per_elem = 4
        elif self.dtype in (KVCacheDtype.FP8_E4M3, KVCacheDtype.FP8_E5M2, KVCacheDtype.INT8):
            bytes_per_elem = 1
        else:
            bytes_per_elem = 2

        # K + V, each is (num_heads * head_dim)
        return 2 * self.hidden_dim * bytes_per_elem

    @property
    def total_bytes(self) -> int:
        """Total memory for full context."""
        return self.max_seq_len * self.num_layers * self.bytes_per_token_per_layer

    def __repr__(self) -> str:
        total_mb = self.total_bytes / (1024 * 1024)
        return (
            f"KVCacheConfig(max_seq_len={self.max_seq_len}, "
            f"layers={self.num_layers}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, dtype={self.dtype.value}, "
            f"total={total_mb:.1f}MB)"
        )


class KVCache:
    """Key-Value cache for transformer attention.

    Stores K and V tensors for all layers to avoid recomputation
    during autoregressive generation.

    Supports quantized storage (FP8, INT8) for memory efficiency.
    """

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.current_seq_len = 0

        # Determine storage dtype
        if config.dtype == KVCacheDtype.BF16:
            self.storage_dtype = torch.bfloat16
        elif config.dtype == KVCacheDtype.FP16:
            self.storage_dtype = torch.float16
        elif config.dtype == KVCacheDtype.FP32:
            self.storage_dtype = torch.float32
        elif config.dtype in (KVCacheDtype.FP8_E4M3, KVCacheDtype.FP8_E5M2):
            # Use uint8 for FP8 storage (manual quant/dequant)
            self.storage_dtype = torch.uint8
            self._init_fp8_scales()
        elif config.dtype == KVCacheDtype.INT8:
            self.storage_dtype = torch.int8
            self._init_int8_scales()
        else:
            self.storage_dtype = torch.bfloat16

        # Allocate cache: [num_layers, 2, max_seq_len, num_heads, head_dim]
        # 2 for K and V
        cache_shape = (
            config.num_layers,
            2,  # K and V
            config.max_seq_len,
            config.num_heads,
            config.head_dim,
        )

        self.cache = torch.zeros(cache_shape, dtype=self.storage_dtype)

        # For quantized formats, store scales per layer
        self.k_scales = torch.ones(config.num_layers, dtype=torch.float32)
        self.v_scales = torch.ones(config.num_layers, dtype=torch.float32)

        logger.info(
            f"Initialized KVCache: {config.max_seq_len} tokens, "
            f"{config.num_layers} layers, {self.config.total_bytes / 1e6:.1f}MB"
        )

    def _init_fp8_scales(self):
        """Initialize FP8 scaling factors."""
        # FP8 E4M3 has max value ~448, E5M2 has max value ~57344
        if self.config.dtype == KVCacheDtype.FP8_E4M3:
            self.fp8_max = 448.0
        else:  # E5M2
            self.fp8_max = 57344.0

    def _init_int8_scales(self):
        """Initialize INT8 scaling factors."""
        self.int8_max = 127.0

    def _quantize_to_fp8(
        self, tensor: torch.Tensor, layer_idx: int, is_key: bool
    ) -> torch.Tensor:
        """Quantize BF16/FP32 tensor to simulated FP8.

        Uses symmetric quantization with per-tensor scaling.
        Stores as int8 internally (not true FP8 format).
        """
        # Compute per-tensor scale
        abs_max = tensor.abs().max().item()
        if abs_max < 1e-6:
            abs_max = 1.0

        # Scale to fit in [-127, 127] range (symmetric int8)
        scale = 127.0 / abs_max

        # Store scale for dequantization
        if is_key:
            self.k_scales[layer_idx] = abs_max / 127.0
        else:
            self.v_scales[layer_idx] = abs_max / 127.0

        # Quantize: scale, round, clamp to int8 range
        quantized = (tensor.float() * scale).round().clamp(-127, 127)

        # Store as int8 (reinterpret uint8 storage)
        return quantized.to(torch.int8).view(torch.uint8)

    def _dequantize_from_fp8(
        self, tensor: torch.Tensor, layer_idx: int, is_key: bool
    ) -> torch.Tensor:
        """Dequantize simulated FP8 tensor to BF16."""
        scale = self.k_scales[layer_idx] if is_key else self.v_scales[layer_idx]

        # Convert back from uint8 (as int8) and apply scale
        dequantized = tensor.view(torch.int8).to(torch.float32) * scale

        return dequantized.to(torch.bfloat16)

    def _quantize_to_int8(
        self, tensor: torch.Tensor, layer_idx: int, is_key: bool
    ) -> torch.Tensor:
        """Quantize BF16/FP32 tensor to INT8."""
        # Compute per-tensor scale
        abs_max = tensor.abs().max().item()
        if abs_max < 1e-6:
            abs_max = 1.0

        scale = self.int8_max / abs_max

        # Store scale for dequantization
        if is_key:
            self.k_scales[layer_idx] = 1.0 / scale
        else:
            self.v_scales[layer_idx] = 1.0 / scale

        # Quantize: clamp and round to int8
        quantized = (tensor * scale).round().clamp(-127, 127)

        return quantized.to(torch.int8)

    def _dequantize_from_int8(
        self, tensor: torch.Tensor, layer_idx: int, is_key: bool
    ) -> torch.Tensor:
        """Dequantize INT8 tensor to BF16."""
        scale = self.k_scales[layer_idx] if is_key else self.v_scales[layer_idx]

        dequantized = tensor.to(torch.float32) * scale

        return dequantized.to(torch.bfloat16)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_pos: int,
    ) -> None:
        """Update KV cache with new key-value pairs.

        Args:
            layer_idx: Layer index
            key: Key tensor [batch, new_tokens, num_heads, head_dim]
            value: Value tensor [batch, new_tokens, num_heads, head_dim]
            seq_pos: Starting position in sequence
        """
        new_tokens = key.shape[1]
        end_pos = seq_pos + new_tokens

        if end_pos > self.config.max_seq_len:
            raise ValueError(
                f"Sequence position {end_pos} exceeds max {self.config.max_seq_len}"
            )

        # Quantize if needed
        if self.config.dtype in (KVCacheDtype.FP8_E4M3, KVCacheDtype.FP8_E5M2):
            key_store = self._quantize_to_fp8(key, layer_idx, is_key=True)
            value_store = self._quantize_to_fp8(value, layer_idx, is_key=False)
        elif self.config.dtype == KVCacheDtype.INT8:
            key_store = self._quantize_to_int8(key, layer_idx, is_key=True)
            value_store = self._quantize_to_int8(value, layer_idx, is_key=False)
        else:
            key_store = key.to(self.storage_dtype)
            value_store = value.to(self.storage_dtype)

        # Store in cache (squeeze batch dim for single-batch)
        if key_store.shape[0] == 1:
            key_store = key_store.squeeze(0)
            value_store = value_store.squeeze(0)

        self.cache[layer_idx, 0, seq_pos:end_pos] = key_store
        self.cache[layer_idx, 1, seq_pos:end_pos] = value_store

        self.current_seq_len = max(self.current_seq_len, end_pos)

    def get(
        self,
        layer_idx: int,
        end_pos: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached key-value pairs.

        Args:
            layer_idx: Layer index
            end_pos: End position (default: current_seq_len)

        Returns:
            Tuple of (key, value) tensors
        """
        if end_pos is None:
            end_pos = self.current_seq_len

        key = self.cache[layer_idx, 0, :end_pos]
        value = self.cache[layer_idx, 1, :end_pos]

        # Dequantize if needed
        if self.config.dtype in (KVCacheDtype.FP8_E4M3, KVCacheDtype.FP8_E5M2):
            key = self._dequantize_from_fp8(key, layer_idx, is_key=True)
            value = self._dequantize_from_fp8(value, layer_idx, is_key=False)
        elif self.config.dtype == KVCacheDtype.INT8:
            key = self._dequantize_from_int8(key, layer_idx, is_key=True)
            value = self._dequantize_from_int8(value, layer_idx, is_key=False)

        return key, value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.zero_()
        self.k_scales.fill_(1.0)
        self.v_scales.fill_(1.0)
        self.current_seq_len = 0

    def memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        return self.cache.numel() * self.cache.element_size() / (1024 * 1024)


def attention_with_kv_cache(
    query: torch.Tensor,
    kv_cache: KVCache,
    layer_idx: int,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    seq_pos: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention with KV cache.

    Args:
        query: Query tensor [batch, num_heads, 1, head_dim] for single token
        kv_cache: KV cache instance
        layer_idx: Layer index
        new_key: New key tensor [batch, 1, num_heads, head_dim]
        new_value: New value tensor [batch, 1, num_heads, head_dim]
        seq_pos: Current sequence position
        scale: Attention scale (default: 1/sqrt(head_dim))

    Returns:
        Attention output [batch, 1, num_heads, head_dim]
    """
    # Update cache with new KV
    kv_cache.update(layer_idx, new_key, new_value, seq_pos)

    # Get all cached KV
    key, value = kv_cache.get(layer_idx, end_pos=seq_pos + 1)

    # Add batch dimension if needed
    if key.dim() == 3:
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

    # Transpose for attention: [batch, num_heads, seq_len, head_dim]
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Ensure consistent dtype with query
    key = key.to(query.dtype)
    value = value.to(query.dtype)

    # Compute attention scale
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    # Attention scores: [batch, num_heads, 1, seq_len]
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)

    # Attention output: [batch, num_heads, 1, head_dim]
    output = torch.matmul(attn_weights, value)

    return output


def compute_kv_cache_memory(
    model_config: dict,
    context_length: int,
    dtype: KVCacheDtype = KVCacheDtype.BF16,
) -> dict:
    """Compute KV cache memory requirements.

    Args:
        model_config: Dict with num_layers, num_heads, head_dim
        context_length: Maximum context length
        dtype: KV cache data type

    Returns:
        Dict with memory statistics
    """
    config = KVCacheConfig(
        max_seq_len=context_length,
        num_layers=model_config.get("num_layers", 32),
        num_heads=model_config.get("num_heads", 32),
        head_dim=model_config.get("head_dim", 128),
        dtype=dtype,
    )

    total_mb = config.total_bytes / (1024 * 1024)
    per_token_kb = config.bytes_per_token_per_layer * config.num_layers / 1024

    return {
        "config": str(config),
        "total_mb": total_mb,
        "per_token_kb": per_token_kb,
        "context_length": context_length,
        "dtype": dtype.value,
    }
