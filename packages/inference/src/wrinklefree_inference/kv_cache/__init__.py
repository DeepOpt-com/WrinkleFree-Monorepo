"""KV cache utilities for BitNet inference.

Includes:
- KV cache implementation with quantized storage (FP8, INT8)
- Validation utilities for testing cache behavior
"""

from wrinklefree_inference.kv_cache.kv_cache import (
    KVCache,
    KVCacheConfig,
    KVCacheDtype,
    attention_with_kv_cache,
    compute_kv_cache_memory,
)
from wrinklefree_inference.kv_cache.validator import KVCacheValidator

__all__ = [
    "KVCache",
    "KVCacheConfig",
    "KVCacheDtype",
    "KVCacheValidator",
    "attention_with_kv_cache",
    "compute_kv_cache_memory",
]
