"""Compute deterministic cache keys for BitNet models."""

import hashlib
import json
from pathlib import Path
from typing import Optional


def compute_cache_key(
    model_path: str,
    revision: Optional[str] = None,
) -> str:
    """Compute cache key from model source and conversion parameters.

    Cache key format: {model_name_safe}_{hash[:12]}

    Args:
        model_path: HuggingFace model ID or local path
        revision: Git revision/commit hash (for HF models)

    Returns:
        Unique cache key string
    """
    key_data = {
        "model_path": model_path,
        "revision": revision or "main",
        "pack_format": "blocked_uint8_v1",  # Our packing format version
    }

    # Add source model hash for local paths
    local_path = Path(model_path)
    if local_path.exists():
        key_data["source_hash"] = _compute_local_model_hash(local_path)

    key_json = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:12]

    # Create safe model name
    model_name_safe = model_path.replace("/", "_").replace(".", "_").replace("-", "_")
    return f"{model_name_safe}_{key_hash}"


def _compute_local_model_hash(model_path: Path) -> str:
    """Compute hash of local model files for cache invalidation."""
    hasher = hashlib.sha256()

    # Hash safetensors files (sorted for determinism)
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    for sf in safetensor_files:
        hasher.update(sf.name.encode())
        hasher.update(str(sf.stat().st_size).encode())
        hasher.update(str(sf.stat().st_mtime_ns).encode())

    # Also hash config.json if present
    config_file = model_path / "config.json"
    if config_file.exists():
        hasher.update(config_file.read_bytes())

    return hasher.hexdigest()[:16]
