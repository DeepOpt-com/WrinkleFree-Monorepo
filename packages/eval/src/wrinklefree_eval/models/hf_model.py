"""HuggingFace model wrapper for lm-evaluation-harness."""

from typing import Any
import logging

import torch
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)


class HuggingFaceModel(HFLM):
    """Extended HuggingFace LM wrapper with WrinkleFree-specific features.

    Supports:
    - Standard HuggingFace model loading
    - Quantized model loading (GPTQ, AWQ, int8)
    - Custom dtype configuration
    - Flash Attention 2
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int | str = "auto",
        trust_remote_code: bool = True,
        use_flash_attention: bool = True,
        quantization: str | None = None,
        device_map: str = "auto",
        **kwargs,
    ):
        """Initialize the HuggingFace model wrapper.

        Args:
            model_path: HuggingFace model ID or local path
            device: Device to run on (cuda, cpu)
            dtype: Model dtype (float16, bfloat16, float32)
            batch_size: Batch size for evaluation (auto for automatic)
            trust_remote_code: Trust remote code in model config
            use_flash_attention: Use Flash Attention 2 if available
            quantization: Quantization type (int8, int4, gptq, awq, None)
            device_map: Device mapping for multi-GPU
        """
        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # Build model kwargs
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "device_map": device_map,
        }

        # Add Flash Attention 2 if requested
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Handle quantization
        if quantization:
            model_kwargs.update(self._get_quantization_config(quantization))

        # Initialize parent class
        super().__init__(
            pretrained=model_path,
            device=device,
            batch_size=batch_size,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            **model_kwargs,
            **kwargs,
        )

        logger.info(f"Loaded model from {model_path}")
        logger.info(f"  Device: {device}, Dtype: {dtype}")
        if quantization:
            logger.info(f"  Quantization: {quantization}")

    def _get_quantization_config(self, quantization: str) -> dict:
        """Get quantization configuration for different methods."""
        if quantization == "int8":
            from transformers import BitsAndBytesConfig
            return {
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True)
            }
        elif quantization == "int4":
            from transformers import BitsAndBytesConfig
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            }
        elif quantization in ("gptq", "awq"):
            # These are auto-detected from model config
            logger.info(f"Using {quantization.upper()} quantization from model config")
            return {}
        else:
            logger.warning(f"Unknown quantization type: {quantization}")
            return {}


def create_hf_model(cfg) -> HuggingFaceModel:
    """Factory function to create HuggingFace model from Hydra config.

    Args:
        cfg: Hydra config with model settings

    Returns:
        Configured HuggingFaceModel instance
    """
    return HuggingFaceModel(
        model_path=cfg.model_path,
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", "bfloat16"),
        batch_size=cfg.get("batch_size", "auto"),
        trust_remote_code=cfg.model.get("trust_remote_code", True),
        use_flash_attention=cfg.model.get("use_flash_attention", True),
        quantization=cfg.model.get("quantization", None),
        device_map=cfg.model.get("device_map", "auto"),
    )
