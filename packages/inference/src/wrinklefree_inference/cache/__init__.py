"""GCS-backed cache for packed BitNet model weights.

This module provides caching infrastructure for converted BitNet models,
storing packed weight files in Google Cloud Storage to avoid repeated
conversion on each server start.

Components:
    compute_cache_key: Generate deterministic cache keys from model config
    GCSModelCache: Client for storing/retrieving cached models from GCS
    get_cached_or_convert: High-level API that fetches from cache or converts

Usage:
    >>> from wrinklefree_inference.cache import get_cached_or_convert
    >>> model_path = get_cached_or_convert(
    ...     checkpoint_path="models/dlm-bitnet-2b",
    ...     bucket="wrinklefree-cache",
    ... )
"""

from .cache_key import compute_cache_key
from .gcs_client import GCSModelCache
from .loader import get_cached_or_convert

__all__ = ["compute_cache_key", "GCSModelCache", "get_cached_or_convert"]
