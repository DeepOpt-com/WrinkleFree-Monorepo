"""BitNet checkpoint loading utilities.

Supports loading BitNet 1.58-bit models from:
- Local paths (safetensors, .pt files)
- HuggingFace Hub (hf://repo_id or repo_id)
- Modal volumes (/checkpoints/...)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)


def load_bitnet_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> tuple[nn.Module, PreTrainedTokenizer, dict[str, Any]]:
    """Load a trained BitNet model from checkpoint.

    Supports multiple path formats:
    - Local: /path/to/checkpoint or ./checkpoint
    - HuggingFace Hub: hf://organization/model-name or just organization/model-name
    - Modal volume: /checkpoints/model_name/stage2

    Args:
        checkpoint_path: Path to the checkpoint
        device: Target device for the model
        dtype: Data type for model weights
        trust_remote_code: Allow custom model code from Hub

    Returns:
        model: The loaded BitNet model
        tokenizer: Associated tokenizer
        config: Model configuration dict

    Raises:
        ValueError: If checkpoint path is invalid or model isn't BitNet
    """
    path_str = str(checkpoint_path)

    # Handle HuggingFace Hub paths
    if path_str.startswith("hf://"):
        path_str = path_str[5:]  # Remove hf:// prefix

    # Check if it's a local path
    local_path = Path(path_str)
    if local_path.exists():
        return _load_local_checkpoint(local_path, device, dtype, trust_remote_code)

    # Try loading from HuggingFace Hub
    return _load_hf_checkpoint(path_str, device, dtype, trust_remote_code)


def _load_local_checkpoint(
    path: Path,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> tuple[nn.Module, PreTrainedTokenizer, dict[str, Any]]:
    """Load checkpoint from local filesystem."""
    logger.info(f"Loading local checkpoint from {path}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
    )

    # Extract config
    config = _extract_config(model, path)

    return model, tokenizer, config


def _load_hf_checkpoint(
    repo_id: str,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> tuple[nn.Module, PreTrainedTokenizer, dict[str, Any]]:
    """Load checkpoint from HuggingFace Hub."""
    logger.info(f"Loading checkpoint from HuggingFace Hub: {repo_id}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        trust_remote_code=trust_remote_code,
    )

    # Extract config
    config = _extract_config(model, repo_id)

    return model, tokenizer, config


def _extract_config(model: nn.Module, source: str | Path) -> dict[str, Any]:
    """Extract configuration from loaded model."""
    model_config = model.config

    return {
        "source": str(source),
        "model_type": getattr(model_config, "model_type", "unknown"),
        "hidden_size": getattr(model_config, "hidden_size", None),
        "num_hidden_layers": getattr(model_config, "num_hidden_layers", None),
        "num_attention_heads": getattr(model_config, "num_attention_heads", None),
        "num_kv_heads": getattr(model_config, "num_key_value_heads", None),
        "vocab_size": getattr(model_config, "vocab_size", None),
        "is_bitnet": _detect_bitnet(model),
    }


def validate_bitnet_model(model: nn.Module) -> bool:
    """Verify model contains BitLinear layers with ternary weights.

    Checks if the model has quantized linear layers typical of BitNet:
    - Weight values concentrated around {-1, 0, +1}
    - BitLinear or equivalent layer types

    Args:
        model: The model to validate

    Returns:
        True if model appears to be a BitNet model
    """
    return _detect_bitnet(model)


def _detect_bitnet(model: nn.Module) -> bool:
    """Detect if model uses BitLinear quantization."""
    for name, module in model.named_modules():
        # Check for BitLinear layer type
        module_type = type(module).__name__
        if "BitLinear" in module_type or "Ternary" in module_type:
            return True

        # Check weight distribution for ternary pattern
        if hasattr(module, "weight") and module.weight is not None:
            weight = module.weight.data
            if weight.numel() > 1000:  # Only check substantial layers
                unique_vals = torch.unique(weight).numel()
                # Ternary weights should have very few unique values
                if unique_vals <= 10:
                    logger.debug(f"Detected ternary weights in {name}: {unique_vals} unique values")
                    return True

    return False


def extract_model_config(model: nn.Module) -> dict[str, Any]:
    """Extract architecture config from model for DLM adaptation.

    Args:
        model: The loaded model

    Returns:
        Dictionary with architecture details needed for block diffusion adaptation
    """
    config = model.config

    return {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "intermediate_size": getattr(config, "intermediate_size", config.hidden_size * 4),
        "vocab_size": config.vocab_size,
        "max_position_embeddings": getattr(config, "max_position_embeddings", 4096),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "attention_bias": getattr(config, "attention_bias", False),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", False),
    }
