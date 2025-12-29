"""Run fingerprinting for deterministic run identification.

Generates a SHA256 hash from training config + git state to uniquely identify
training runs. Infrastructure-only changes (num_workers, logging, etc.) don't
affect the fingerprint, allowing resume across different execution environments.

Example:
    >>> from omegaconf import OmegaConf
    >>> config = OmegaConf.load("configs/training/stage2_pretrain.yaml")
    >>> fingerprint, metadata = generate_fingerprint(config)
    >>> print(f"Run fingerprint: {fingerprint[:8]}...")
    Run fingerprint: a3f7c2d1...
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


# Infrastructure keys to exclude from fingerprint
# These affect HOW training runs (resources, logging) but not WHAT is trained
IGNORE_KEYS: set[str] = {
    # Hydra internals
    "hydra",
    # Paths and I/O (can vary by machine)
    "output_dir",
    "checkpoint.resume_from",
    "checkpoint.hub.repo_id",
    # Logging infrastructure (doesn't affect model)
    "logging.wandb.entity",
    "logging.wandb.project",
    "logging.wandb.tags",
    "logging.wandb.name",
    "logging.log_interval",
    # Checkpointing frequency (doesn't affect training math)
    "checkpoint.save_interval",
    "checkpoint.keep_last_n",
    "checkpoint.hub.upload_interval",
    # System resources
    "num_workers",
    "data.num_workers",
    "data.dataloader.num_workers",
    "data.dataloader.prefetch_factor",
    "data.dataloader.pin_memory",
    # GCS and resume configuration (meta-settings)
    "gcs",
    "resume",
    "audit",
    # Experiment naming (doesn't affect training)
    "experiment_name",
}


def get_git_info(repo_path: Path | None = None) -> tuple[str, bool]:
    """Get git commit hash and dirty status.

    Args:
        repo_path: Path to git repository. If None, uses current directory.

    Returns:
        Tuple of (commit_hash, is_dirty).
        Returns ("unknown", False) if not a git repo or git unavailable.
    """
    cwd = str(repo_path) if repo_path else None

    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode != 0:
            return ("unknown", False)

        commit_hash = result.stdout.strip()

        # Check if working directory is dirty
        result = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            capture_output=True,
            cwd=cwd,
            timeout=5,
        )
        is_dirty = result.returncode != 0

        return (commit_hash, is_dirty)

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ("unknown", False)


def _should_ignore_key(key_path: str, ignore_keys: set[str]) -> bool:
    """Check if a key path should be ignored.

    Supports both exact matches and prefix matches for nested keys.
    Also matches leaf keys (e.g., "logging" matches "level1.level2.logging").

    Args:
        key_path: Dot-separated key path (e.g., "logging.wandb.project")
        ignore_keys: Set of keys to ignore

    Returns:
        True if key should be ignored
    """
    # Exact match
    if key_path in ignore_keys:
        return True

    # Check if any ignore key is a prefix (for nested structures)
    # e.g., "logging.wandb" ignores "logging.wandb.project"
    for ignore_key in ignore_keys:
        if key_path.startswith(f"{ignore_key}."):
            return True

    # Check if the leaf key (last part of path) matches
    # e.g., "level1.level2.logging" should match ignore_key "logging"
    leaf_key = key_path.rsplit(".", 1)[-1]
    if leaf_key in ignore_keys:
        return True

    return False


def clean_config_for_hashing(
    cfg_dict: dict[str, Any],
    ignore_keys: set[str] | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Recursively clean config dict for deterministic hashing.

    Removes ignored keys and sorts remaining keys for determinism.

    Args:
        cfg_dict: Config dictionary to clean
        ignore_keys: Set of keys to ignore (uses IGNORE_KEYS if None)
        prefix: Current key path prefix for nested keys

    Returns:
        Cleaned dictionary with sorted keys
    """
    if ignore_keys is None:
        ignore_keys = IGNORE_KEYS

    result = {}
    for key, value in sorted(cfg_dict.items()):
        # Build full key path
        key_path = f"{prefix}.{key}" if prefix else key

        # Skip ignored keys
        if _should_ignore_key(key_path, ignore_keys):
            continue

        # Recurse into nested dicts
        if isinstance(value, dict):
            cleaned = clean_config_for_hashing(value, ignore_keys, key_path)
            if cleaned:  # Only add non-empty dicts
                result[key] = cleaned
        elif isinstance(value, (list, tuple)):
            # Handle lists (e.g., tags, layer configs)
            result[key] = [
                clean_config_for_hashing(v, ignore_keys, f"{key_path}[{i}]")
                if isinstance(v, dict)
                else v
                for i, v in enumerate(value)
            ]
        else:
            result[key] = value

    return result


def generate_fingerprint(
    cfg: DictConfig | dict[str, Any],
    ignore_keys: set[str] | None = None,
    include_git: bool = True,
    repo_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate a deterministic fingerprint for a training run.

    The fingerprint is a SHA256 hash of:
    - Cleaned config (infrastructure keys removed)
    - Git commit hash (if include_git=True)

    Note: Git dirty flag is NOT included in the fingerprint to allow
    resuming runs during development. It's tracked in metadata for auditing.

    Args:
        cfg: Training configuration (DictConfig or dict)
        ignore_keys: Keys to exclude from hash (uses IGNORE_KEYS if None)
        include_git: Whether to include git commit in fingerprint
        repo_path: Path to git repository

    Returns:
        Tuple of (fingerprint, metadata) where metadata contains:
        - fingerprint: The SHA256 hash
        - git_commit: Commit hash
        - git_dirty: Whether repo has uncommitted changes
        - config_hash: Hash of just the config (without git)
        - timestamp: When fingerprint was generated
    """
    # Convert OmegaConf to dict and resolve interpolations
    # Use throw_on_missing=False to handle missing interpolation keys gracefully
    if isinstance(cfg, DictConfig):
        try:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        except Exception:
            # If resolution fails, try without resolution
            # This can happen with complex interpolations
            cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    else:
        cfg_dict = cfg

    # Clean config for hashing
    if ignore_keys is None:
        ignore_keys = IGNORE_KEYS
    cleaned_config = clean_config_for_hashing(cfg_dict, ignore_keys)

    # Generate config-only hash
    config_json = json.dumps(cleaned_config, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()

    # Get git info
    git_commit, git_dirty = get_git_info(repo_path) if include_git else ("none", False)

    # Generate full fingerprint (config + git commit, but NOT dirty flag)
    fingerprint_data = {
        "config": cleaned_config,
        "git_commit": git_commit,
    }
    fingerprint_json = json.dumps(
        fingerprint_data, sort_keys=True, separators=(",", ":")
    )
    fingerprint = hashlib.sha256(fingerprint_json.encode()).hexdigest()

    # Build metadata
    metadata = {
        "fingerprint": fingerprint,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash": config_hash,
        "timestamp": datetime.now().isoformat(),
    }

    return fingerprint, metadata


def fingerprint_matches(
    cfg: DictConfig | dict[str, Any],
    expected_fingerprint: str,
    ignore_keys: set[str] | None = None,
    include_git: bool = True,
    repo_path: Path | None = None,
) -> bool:
    """Check if a config produces the expected fingerprint.

    Args:
        cfg: Training configuration
        expected_fingerprint: Fingerprint to match against
        ignore_keys: Keys to exclude from hash
        include_git: Whether to include git commit
        repo_path: Path to git repository

    Returns:
        True if fingerprints match
    """
    fingerprint, _ = generate_fingerprint(cfg, ignore_keys, include_git, repo_path)
    return fingerprint == expected_fingerprint
