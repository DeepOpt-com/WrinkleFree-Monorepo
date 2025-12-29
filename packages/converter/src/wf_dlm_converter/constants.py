"""Centralized constants for WrinkleFree DLM Converter.

All magic strings, default values, and configuration constants live here.
"""

from enum import Enum
from typing import Final

# =============================================================================
# Modal App Configuration
# =============================================================================

MODAL_APP_NAME: Final[str] = "wrinklefree-dlm-converter"
"""Modal application name for DLM conversion jobs."""

MODAL_VOLUME_DLM_OUTPUTS: Final[str] = "wrinklefree-dlm-outputs"
"""Modal volume for storing converted DLM model outputs."""

# Shared volumes (from wf_deployer if available, otherwise standalone)
MODAL_VOLUME_CHECKPOINTS: Final[str] = "wrinklefree-checkpoints"
"""Modal volume for BitNet source checkpoints."""

MODAL_VOLUME_HF_CACHE: Final[str] = "wrinklefree-hf-cache"
"""Modal volume for HuggingFace model/tokenizer cache."""

# =============================================================================
# Repository URLs
# =============================================================================

REPO_DLM_CONVERTER: Final[str] = "https://github.com/DeepOpt-com/WrinkleFree-DLM-Converter.git"
"""This repository."""

REPO_FAST_DLLM: Final[str] = "https://github.com/NVlabs/Fast-dLLM.git"
"""Fast-dLLM reference implementation (Apache 2.0)."""

# =============================================================================
# GCS Configuration
# =============================================================================

GCS_BUCKET: Final[str] = "wrinklefree-checkpoints"
"""GCS bucket for checkpoint storage."""

GCS_DLM_PREFIX: Final[str] = "dlm-checkpoints"
"""Prefix for DLM converted model checkpoints in GCS."""

# =============================================================================
# Default Values
# =============================================================================

DEFAULT_MODEL: Final[str] = "smollm2_135m"
"""Default model config for conversion."""

DEFAULT_BLOCK_SIZE: Final[int] = 32
"""Default block size for block diffusion."""

DEFAULT_DIFFUSION_STEPS: Final[int] = 8
"""Default number of diffusion steps per block."""

DEFAULT_TOTAL_TOKENS: Final[int] = 1_000_000_000
"""Default fine-tuning tokens (1B)."""

DEFAULT_WANDB_PROJECT: Final[str] = "wrinklefree-dlm"
"""Default Weights & Biases project name."""

# =============================================================================
# Training Hyperparameters
# =============================================================================

DEFAULT_LEARNING_RATE: Final[float] = 5e-5
"""Default learning rate for diffusion fine-tuning."""

DEFAULT_BATCH_SIZE: Final[int] = 8
"""Default batch size."""

DEFAULT_MAX_SEQ_LENGTH: Final[int] = 512
"""Default maximum sequence length."""

# =============================================================================
# Timeouts (in seconds)
# =============================================================================

CONVERSION_TIMEOUT: Final[int] = 24 * 60 * 60  # 24 hours
"""Maximum conversion job duration on Modal."""

VALIDATION_TIMEOUT: Final[int] = 30 * 60  # 30 minutes
"""Maximum validation job duration."""

# =============================================================================
# Run ID Prefixes
# =============================================================================


class RunIdPrefix(str, Enum):
    """Prefixes for run IDs to identify job type."""

    CONVERT = "dlm-convert-"
    VALIDATE = "dlm-validate-"


# =============================================================================
# GPU Configuration
# =============================================================================

SUPPORTED_GPU_TYPES: Final[frozenset[str]] = frozenset({
    "H100", "A100", "A100-80GB", "A10G", "L4"
})
"""Supported GPU types for conversion."""

DEFAULT_GPU_TYPE: Final[str] = "H100"
"""Default GPU for conversion jobs."""

# =============================================================================
# Model Architecture Constants
# =============================================================================


class MaskToken:
    """Special tokens for diffusion masking."""

    ID: Final[int] = 32000  # Typical mask token ID
    TEXT: Final[str] = "[MASK]"
