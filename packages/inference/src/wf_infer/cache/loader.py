"""Main caching logic for BitNet model loading."""

import logging
from pathlib import Path
from typing import Optional

from .bitnet_converter import convert_and_save_bitnet
from .cache_key import compute_cache_key
from .gcs_client import GCSModelCache

logger = logging.getLogger(__name__)


def get_cached_or_convert(
    model_path: str,
    revision: Optional[str] = None,
    cache: Optional[GCSModelCache] = None,
    skip_gcs: bool = False,
) -> Path:
    """Get cached model or convert from source.

    Args:
        model_path: HuggingFace model ID or local path
        revision: Model revision
        cache: GCS cache instance (created if None)
        skip_gcs: If True, skip GCS check (for local-only testing)

    Returns:
        Path to model ready for loading
    """
    if cache is None:
        cache = GCSModelCache()

    cache_key = compute_cache_key(model_path, revision)
    logger.info(f"Cache key: {cache_key}")

    # 1. Check local cache first
    local_path = cache.cache_key_to_local_path(cache_key)
    if (local_path / "config.json").exists():
        logger.info(f"Found in local cache: {local_path}")
        return local_path

    # 2. Check GCS cache
    if not skip_gcs:
        try:
            if cache.exists(cache_key):
                logger.info("Found in GCS cache, downloading...")
                return cache.download(cache_key)
        except Exception as e:
            logger.warning(f"GCS check failed: {e}, proceeding with conversion")

    # 3. Convert from source
    logger.info("Cache miss - converting from source...")
    local_path.mkdir(parents=True, exist_ok=True)
    convert_and_save_bitnet(model_path, local_path, revision)

    # 4. Upload to GCS
    if not skip_gcs:
        try:
            logger.info("Uploading to GCS cache...")
            cache.upload(local_path, cache_key)
        except Exception as e:
            logger.warning(f"GCS upload failed: {e}")

    return local_path
