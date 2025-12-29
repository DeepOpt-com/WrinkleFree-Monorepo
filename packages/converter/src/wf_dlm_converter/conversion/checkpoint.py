"""Checkpoint saving and loading for converted DLM models.

Saves models in a format compatible with Fast-dLLM inference.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
from transformers import PreTrainedTokenizer, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DLMConfig:
    """Configuration for DLM model inference.

    Stored alongside model weights for Fast-dLLM compatibility.
    """

    # Block diffusion parameters
    block_size: int = 32
    num_diffusion_steps: int = 8
    noise_schedule: str = "cosine"

    # Model info
    source_model: str = ""
    source_checkpoint: str = ""
    is_bitnet: bool = True

    # Inference settings
    confidence_threshold: float = 0.9
    use_kv_cache: bool = True
    parallel_decode: bool = True

    # Training metadata
    total_tokens_trained: int = 0
    training_loss: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DLMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def save_dlm_checkpoint(
    model: nn.Module,
    output_path: str | Path,
    tokenizer: PreTrainedTokenizer,
    dlm_config: Optional[DLMConfig] = None,
    format: str = "safetensors",
) -> Path:
    """Save converted model in Fast-dLLM compatible format.

    Output structure:
        output_path/
        ├── config.json           # HuggingFace model config
        ├── model.safetensors     # Model weights
        ├── tokenizer/            # Tokenizer files
        │   ├── tokenizer.json
        │   ├── tokenizer_config.json
        │   └── special_tokens_map.json
        └── dlm_config.json       # Block diffusion parameters

    Args:
        model: The converted DLM model
        output_path: Directory to save the checkpoint
        tokenizer: Model tokenizer
        dlm_config: DLM configuration (created from model if not provided)
        format: Weight format ('safetensors' or 'pytorch')

    Returns:
        Path to the saved checkpoint directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving DLM checkpoint to {output_path}")

    # Create DLM config from model if not provided
    if dlm_config is None:
        dlm_config = _create_dlm_config(model)

    # Save model weights
    if format == "safetensors":
        weights_path = output_path / "model.safetensors"
        state_dict = model.state_dict()
        save_safetensors(state_dict, weights_path)
    else:
        weights_path = output_path / "pytorch_model.bin"
        torch.save(model.state_dict(), weights_path)

    logger.info(f"Saved weights to {weights_path}")

    # Save model config (HuggingFace format)
    if hasattr(model, "config"):
        config_dict = model.config.to_dict()
        # Add DLM-specific fields
        config_dict["is_dlm"] = True
        config_dict["block_size"] = dlm_config.block_size
        config_dict["num_diffusion_steps"] = dlm_config.num_diffusion_steps

        with open(output_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    # Save DLM config
    with open(output_path / "dlm_config.json", "w") as f:
        json.dump(dlm_config.to_dict(), f, indent=2)

    # Save tokenizer
    tokenizer_path = output_path / "tokenizer"
    tokenizer_path.mkdir(exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    # Also save to root for HF compatibility
    tokenizer.save_pretrained(output_path)

    logger.info(f"DLM checkpoint saved to {output_path}")
    return output_path


def load_dlm_checkpoint(
    path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, PreTrainedTokenizer, DLMConfig]:
    """Load a converted DLM model for inference.

    Args:
        path: Path to checkpoint directory
        device: Target device
        dtype: Model dtype

    Returns:
        model: Loaded DLM model
        tokenizer: Model tokenizer
        dlm_config: DLM configuration
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading DLM checkpoint from {path}")

    # Load DLM config
    dlm_config_path = path / "dlm_config.json"
    if dlm_config_path.exists():
        with open(dlm_config_path) as f:
            dlm_config = DLMConfig.from_dict(json.load(f))
    else:
        dlm_config = DLMConfig()

    # Load model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer_path = path / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path)

    logger.info(f"Loaded DLM model: block_size={dlm_config.block_size}")

    return model, tokenizer, dlm_config


def _create_dlm_config(model: nn.Module) -> DLMConfig:
    """Create DLM config from model attributes."""
    config = DLMConfig()

    # Extract from model config if available
    if hasattr(model, "config"):
        if hasattr(model.config, "block_size"):
            config.block_size = model.config.block_size
        if hasattr(model.config, "num_diffusion_steps"):
            config.num_diffusion_steps = model.config.num_diffusion_steps
        if hasattr(model.config, "_name_or_path"):
            config.source_model = model.config._name_or_path

    # Check for adapter
    if hasattr(model, "_dlm_adapter"):
        adapter = model._dlm_adapter
        config.block_size = adapter.block_size
        config.num_diffusion_steps = adapter.num_diffusion_steps
        config.noise_schedule = adapter.noise_schedule

    return config


def export_for_inference(
    model: nn.Module,
    output_path: str | Path,
    tokenizer: PreTrainedTokenizer,
    optimize: bool = True,
) -> Path:
    """Export model optimized for Fast-dLLM inference.

    This creates a checkpoint specifically optimized for the
    Fast-dLLM inference pipeline with KV caching and parallel decode.

    Args:
        model: The converted DLM model
        output_path: Directory to save
        tokenizer: Model tokenizer
        optimize: Apply inference optimizations

    Returns:
        Path to exported checkpoint
    """
    output_path = Path(output_path)

    # First save standard checkpoint
    save_dlm_checkpoint(
        model=model,
        output_path=output_path,
        tokenizer=tokenizer,
    )

    if optimize:
        logger.info("Applying inference optimizations...")

        # Create inference config
        inference_config = {
            "use_kv_cache": True,
            "block_kv_cache": True,
            "parallel_decode": True,
            "confidence_threshold": 0.9,
            "torch_compile": True,
        }

        with open(output_path / "inference_config.json", "w") as f:
            json.dump(inference_config, f, indent=2)

    return output_path
