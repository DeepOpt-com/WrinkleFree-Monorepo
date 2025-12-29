"""FP8 GEMM acceleration for BitLinear using TorchAO (DeepSeek-V3 style).

This module provides:
- Hardware detection for FP8 capability (H100+ vs A100)
- FP8Config dataclass for configuration
- Utility functions for layer-level FP8 decisions

References:
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- TorchAO Float8: https://github.com/pytorch/ao/tree/main/torchao/float8
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class FP8Capability(Enum):
    """GPU FP8 capability levels."""

    NONE = "none"  # No FP8 support (pre-Hopper: A100, etc.)
    HOPPER = "hopper"  # H100/H200 - native FP8 tensor cores


@dataclass
class FP8Config:
    """Configuration for FP8 GEMM acceleration (DeepSeek-V3 style).

    Attributes:
        enabled: Whether FP8 is enabled (still requires hardware support)
        recipe: Scaling recipe - "rowwise" (DeepSeek-V3 style, more accurate)
                or "tensorwise" (faster, single scale per tensor)
        accumulator_dtype: GEMM accumulator precision - "float32" (safer) or "bfloat16"
        exclude_patterns: Layer name patterns to exclude from FP8
                         (following DeepSeek-V3: embedding, output head, norm)
        min_gemm_size: Minimum K/N dimension for FP8 to be beneficial
    """

    enabled: bool = True
    recipe: str = "rowwise"
    accumulator_dtype: str = "float32"
    exclude_patterns: tuple[str, ...] = field(
        default_factory=lambda: ("embed_tokens", "lm_head", "norm", "subln")
    )
    min_gemm_size: int = 512


@functools.lru_cache(maxsize=1)
def detect_fp8_capability() -> FP8Capability:
    """Detect GPU FP8 capability at runtime.

    Uses CUDA compute capability to determine support:
    - sm_90+ (Hopper): Full FP8 tensor core support
    - sm_89 (Ada): Limited FP8 support (not recommended for training)
    - sm_80 (Ampere): No FP8 support

    Returns:
        FP8Capability enum indicating hardware support level.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available: FP8 disabled")
        return FP8Capability.NONE

    # Get compute capability of first GPU
    device_props = torch.cuda.get_device_properties(0)
    major, minor = device_props.major, device_props.minor
    gpu_name = device_props.name

    # Hopper (sm_90) and later support FP8 training
    if major >= 9:
        logger.info(f"Detected Hopper+ GPU ({gpu_name}, sm_{major}{minor}): FP8 enabled")
        return FP8Capability.HOPPER

    # Ada Lovelace (sm_89) has FP8 but not well-suited for training
    # Ampere (sm_80) has no FP8 support
    logger.info(f"Detected pre-Hopper GPU ({gpu_name}, sm_{major}{minor}): FP8 disabled")
    return FP8Capability.NONE


def should_use_fp8_for_layer(
    layer_name: str,
    in_features: int,
    out_features: int,
    config: FP8Config,
) -> bool:
    """Determine if a specific layer should use FP8 GEMM.

    Following DeepSeek-V3 pattern:
    - FP8 for linear layer GEMMs only
    - Exclude embedding, output head, normalization layers
    - Skip small GEMMs where FP8 overhead > benefit

    Args:
        layer_name: Full name of the layer (e.g., "model.layers.0.self_attn.q_proj")
        in_features: Input dimension (K in GEMM)
        out_features: Output dimension (N in GEMM)
        config: FP8 configuration

    Returns:
        True if layer should use FP8, False otherwise.
    """
    if not config.enabled:
        return False

    # Check hardware capability
    if detect_fp8_capability() == FP8Capability.NONE:
        return False

    # Check exclusion patterns (embedding, lm_head, normalization)
    layer_name_lower = layer_name.lower()
    for pattern in config.exclude_patterns:
        if pattern.lower() in layer_name_lower:
            logger.debug(f"Layer {layer_name} excluded from FP8 (pattern: {pattern})")
            return False

    # Check minimum GEMM size (FP8 overhead not worth it for small GEMMs)
    if min(in_features, out_features) < config.min_gemm_size:
        logger.debug(
            f"Layer {layer_name} excluded from FP8 "
            f"(size {in_features}x{out_features} < {config.min_gemm_size})"
        )
        return False

    return True


def get_accumulator_dtype(config: FP8Config) -> torch.dtype:
    """Get the torch dtype for FP8 GEMM accumulation.

    Args:
        config: FP8 configuration

    Returns:
        torch.float32 or torch.bfloat16
    """
    if config.accumulator_dtype == "float32":
        return torch.float32
    elif config.accumulator_dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid accumulator_dtype: {config.accumulator_dtype}")


def check_torchao_available() -> bool:
    """Check if TorchAO FP8 functionality is available.

    Returns:
        True if torchao.float8 is importable, False otherwise.
    """
    try:
        import torchao.float8  # noqa: F401

        return True
    except ImportError:
        return False


def log_fp8_config(config: FP8Config) -> None:
    """Log FP8 configuration for debugging.

    Args:
        config: FP8 configuration to log
    """
    capability = detect_fp8_capability()
    torchao_available = check_torchao_available()

    logger.info("=" * 50)
    logger.info("FP8 Configuration (DeepSeek-V3 Style)")
    logger.info("=" * 50)
    logger.info(f"  Enabled: {config.enabled}")
    logger.info(f"  Hardware capability: {capability.value}")
    logger.info(f"  TorchAO available: {torchao_available}")
    logger.info(f"  Recipe: {config.recipe}")
    logger.info(f"  Accumulator dtype: {config.accumulator_dtype}")
    logger.info(f"  Min GEMM size: {config.min_gemm_size}")
    logger.info(f"  Exclude patterns: {config.exclude_patterns}")

    if config.enabled and capability == FP8Capability.NONE:
        logger.warning("FP8 requested but hardware does not support it - using BF16 fallback")
    if config.enabled and not torchao_available:
        logger.warning("FP8 requested but TorchAO not installed - using BF16 fallback")

    logger.info("=" * 50)
