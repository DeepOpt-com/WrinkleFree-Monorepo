"""Auto-setup utilities for model loading, conversion, and checkpoint resolution.

This module provides utilities for:
1. Resolving checkpoint paths (local or GCS)
2. Auto-converting models to BitNet if needed
3. Loading models with appropriate configuration
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def resolve_checkpoint_path(
    checkpoint_path: Optional[str],
    gcs_bucket: Optional[str] = None,
    local_cache_dir: str = "./checkpoints",
) -> Optional[Path]:
    """Resolve a checkpoint path, downloading from GCS if needed.

    Args:
        checkpoint_path: Local path, GCS URL (gs://...), or HuggingFace model name
        gcs_bucket: Optional GCS bucket for checkpoint discovery
        local_cache_dir: Local directory to cache downloaded checkpoints

    Returns:
        Path to the resolved checkpoint, or None if not found
    """
    if checkpoint_path is None:
        return None

    # Case 1: Local path exists
    local_path = Path(checkpoint_path)
    if local_path.exists():
        logger.info(f"Using local checkpoint: {local_path}")
        return local_path

    # Case 2: GCS URL
    if checkpoint_path.startswith("gs://"):
        return _download_from_gcs_url(checkpoint_path, local_cache_dir)

    # Case 3: HuggingFace model name (will be loaded directly, return as-is)
    if "/" in checkpoint_path and not checkpoint_path.startswith("/"):
        logger.info(f"Using HuggingFace model: {checkpoint_path}")
        return Path(checkpoint_path)  # Treated as HF model name

    logger.warning(f"Checkpoint not found: {checkpoint_path}")
    return None


def _download_from_gcs_url(gcs_url: str, local_cache_dir: str) -> Optional[Path]:
    """Download a checkpoint from a GCS URL.

    Args:
        gcs_url: GCS URL (gs://bucket/path/to/checkpoint)
        local_cache_dir: Local directory to cache downloads

    Returns:
        Path to downloaded checkpoint, or None on failure
    """
    # Parse GCS URL
    match = re.match(r"gs://([^/]+)/(.+)", gcs_url)
    if not match:
        logger.error(f"Invalid GCS URL: {gcs_url}")
        return None

    bucket_name, gcs_path = match.groups()

    # Create unique local directory based on GCS path
    local_dir = Path(local_cache_dir) / bucket_name / gcs_path.replace("/", "_")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    checkpoint_file = local_dir / "checkpoint.pt"
    if checkpoint_file.exists():
        logger.info(f"Using cached checkpoint: {checkpoint_file}")
        return local_dir

    # Download using gcloud CLI
    logger.info(f"Downloading from GCS: {gcs_url}")
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", "-r", gcs_url, str(local_dir)],
            check=True,
            capture_output=True,
        )
        logger.info(f"Downloaded checkpoint to: {local_dir}")
        return local_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download from GCS: {e.stderr.decode()}")
        return None
    except FileNotFoundError:
        logger.error("gcloud CLI not found. Please install Google Cloud SDK.")
        return None


def auto_setup_model(
    config: DictConfig,
    device: torch.device,
    tokenizer: Optional[AutoTokenizer] = None,
) -> tuple[nn.Module, Optional[AutoTokenizer]]:
    """Auto-setup model with checkpoint resolution and BitNet conversion.

    This function handles the complete model setup pipeline:
    1. Resolve checkpoint path (local, GCS, or HuggingFace)
    2. Load model from checkpoint or pretrained
    3. Auto-convert to BitNet if needed and enabled
    4. Return model and tokenizer

    Args:
        config: Full Hydra configuration
        device: Device to load model on
        tokenizer: Optional pre-loaded tokenizer

    Returns:
        Tuple of (model, tokenizer)
    """
    training_config = getattr(config, "training", config)
    model_config = getattr(config, "model", {})

    # Get checkpoint path from config
    checkpoint_path = getattr(model_config, "checkpoint_path", None)
    pretrained_name = getattr(model_config, "pretrained_name", None)

    # Get auto-convert settings
    auto_convert_config = getattr(training_config, "auto_convert", {})
    auto_convert_enabled = getattr(auto_convert_config, "enabled", True)
    gcs_bucket = getattr(auto_convert_config, "gcs_bucket", None)
    exclude_layers = list(getattr(auto_convert_config, "exclude_layers", []))

    # Resolve checkpoint path
    resolved_path = resolve_checkpoint_path(
        checkpoint_path or pretrained_name,
        gcs_bucket=gcs_bucket,
    )

    # Load model
    if resolved_path is None:
        raise ValueError(
            f"No checkpoint found. Specify model.checkpoint_path or model.pretrained_name"
        )

    model, tokenizer = _load_model(resolved_path, device, tokenizer)

    # Auto-convert to BitNet if needed and enabled
    if auto_convert_enabled:
        model = _auto_convert_if_needed(model, exclude_layers)

    return model, tokenizer


def _load_model(
    path: Path,
    device: torch.device,
    tokenizer: Optional[AutoTokenizer] = None,
) -> tuple[nn.Module, AutoTokenizer]:
    """Load a model from a path (checkpoint or HuggingFace).

    Args:
        path: Path to checkpoint directory or HuggingFace model name
        device: Device to load model on
        tokenizer: Optional pre-loaded tokenizer

    Returns:
        Tuple of (model, tokenizer)
    """
    path_str = str(path)

    # Check if this is a HuggingFace model name (contains / but not a local path)
    if "/" in path_str and not path.exists():
        logger.info(f"Loading from HuggingFace: {path_str}")
        model = AutoModelForCausalLM.from_pretrained(
            path_str,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                path_str,
                trust_remote_code=True,
            )
        return model, tokenizer

    # Load from local checkpoint
    checkpoint_file = path / "checkpoint.pt" if path.is_dir() else path

    if not checkpoint_file.exists():
        # Try loading as HuggingFace format (safetensors/bin files)
        if (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists():
            logger.info(f"Loading HuggingFace format checkpoint: {path}")
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)

            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            return model, tokenizer
        else:
            raise FileNotFoundError(f"No checkpoint found at: {path}")

    logger.info(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)

    # Extract model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Get model config from checkpoint or config file
    model_config = checkpoint.get("model_config", {})
    pretrained_name = model_config.get("pretrained_name", model_config.get("_name_or_path"))

    if pretrained_name:
        # Create model from pretrained config
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_name, trust_remote_code=True)
    else:
        raise ValueError(
            "Cannot determine model architecture. "
            "Checkpoint must include model_config.pretrained_name"
        )

    return model, tokenizer


def _auto_convert_if_needed(
    model: nn.Module,
    exclude_layers: Optional[list[str]] = None,
) -> nn.Module:
    """Convert model to BitNet if not already converted.

    Args:
        model: Model to potentially convert
        exclude_layers: Layer name patterns to exclude from conversion

    Returns:
        Converted model (or original if already BitNet)
    """
    try:
        from wf_arch import auto_convert_if_needed, is_bitnet_model

        if is_bitnet_model(model):
            logger.info("Model is already BitNet, skipping conversion")
            return model

        logger.info("Converting model to BitNet (auto_convert.enabled=true)")
        model = auto_convert_if_needed(
            model,
            exclude_layers=exclude_layers or ["embed_tokens", "lm_head"],
        )
        logger.info("Model converted to BitNet successfully")
        return model

    except ImportError:
        logger.warning(
            "wf_arch not available. Install with: "
            "pip install bitnet-arch"
        )
        return model


def get_mask_token_id(
    tokenizer: AutoTokenizer,
    config: Optional[DictConfig] = None,
) -> int:
    """Get the mask token ID for DLM training.

    Priority:
    1. Config value (training.objectives.dlm.mask_token_id)
    2. Tokenizer's mask_token_id
    3. UNK token ID
    4. Pad token ID
    5. Error

    Args:
        tokenizer: The tokenizer to get mask token from
        config: Optional config with explicit mask_token_id

    Returns:
        Token ID to use for masking
    """
    # Check config first
    if config is not None:
        training_config = getattr(config, "training", config)
        objectives = getattr(training_config, "objectives", {})
        dlm_config = getattr(objectives, "dlm", {})
        mask_token_id = getattr(dlm_config, "mask_token_id", None)
        if mask_token_id is not None:
            return mask_token_id

    # Try tokenizer's mask token
    if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id

    # Fallback to UNK or PAD token
    if hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None:
        logger.warning(f"Using UNK token ({tokenizer.unk_token_id}) for DLM masking")
        return tokenizer.unk_token_id

    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        logger.warning(f"Using PAD token ({tokenizer.pad_token_id}) for DLM masking")
        return tokenizer.pad_token_id

    raise ValueError(
        "No mask token ID found. Set training.objectives.dlm.mask_token_id "
        "or use a tokenizer with a mask token."
    )
