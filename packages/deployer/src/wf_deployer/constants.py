"""Centralized constants for WrinkleFree Deployer.

All magic strings, default values, and configuration constants live here.
Import from this module instead of hardcoding values.

Example:
    from wf_deployer.constants import SCALES, RunIdPrefix, get_scale_for_model

    scale = get_scale_for_model("qwen3_4b")
    run_id = f"{RunIdPrefix.SKYPILOT.value}{model}-s{stage}"
"""

import os
from enum import Enum
from typing import Final

# =============================================================================
# Repository URLs
# =============================================================================

REPO_1_58_QUANT: Final[str] = "https://github.com/DeepOpt-com/WrinkleFree-1.58Quant.git"
"""Main training code repository (1.58-bit quantization)."""

REPO_CHEAPER_TRAINING: Final[str] = "https://github.com/DeepOpt-com/WrinkleFree-CheaperTraining.git"
"""Training optimization library repository."""

# =============================================================================
# Default Values
# =============================================================================

DEFAULT_DATA: Final[str] = "fineweb"
"""Default data config for training."""

DEFAULT_WANDB_PROJECT: Final[str] = "wrinklefree"
"""Default Weights & Biases project name."""

DEFAULT_CONTEXT_SIZE: Final[int] = 4096
"""Default context window size for inference."""

DEFAULT_SMOKE_TEST_MODEL: Final[str] = "smollm2_135m"
"""Default model for smoke tests (smallest, fastest)."""

# =============================================================================
# Timeouts (in seconds)
# =============================================================================

TRAINING_TIMEOUT: Final[int] = 24 * 60 * 60  # 24 hours
"""Maximum training job duration on Modal (Modal's hard limit)."""

SMOKE_TEST_TIMEOUT: Final[int] = 30 * 60  # 30 minutes
"""Maximum smoke test duration."""

DEBUG_TIMEOUT: Final[int] = 5 * 60  # 5 minutes
"""Maximum debug/clone test duration."""

# =============================================================================
# Run ID Prefixes
# =============================================================================


class RunIdPrefix(str, Enum):
    """Prefixes for run IDs to identify the backend.

    Run IDs are prefixed to make backend detection self-documenting:
    - sky-qwen3_4b-s2:123 -> SkyPilot backend

    Example:
        if run_id.startswith(RunIdPrefix.SKYPILOT.value):
            _logs_skypilot(run_id, follow)
    """

    SKYPILOT = "sky-"


# =============================================================================
# GPU Configuration
# =============================================================================

SUPPORTED_GPU_TYPES: Final[frozenset[str]] = frozenset({
    "H100", "A100", "A100-80GB", "A10G", "L4", "T4"
})
"""Supported GPU types for training."""

# GPU profile mapping for Hydra configs
GPU_PROFILES: Final[dict[str, str]] = {
    "H100": "h100_80gb",
    "A100": "a100_40gb",
    "A100-80GB": "a100_80gb",
    "A10G": "a10g_24gb",
    "L4": "l4_24gb",
    "T4": "t4_16gb",
}
"""Maps GPU type to Hydra GPU profile config name."""

# =============================================================================
# Scale Profiles (GPU T-shirt Sizing)
# =============================================================================

SCALES: Final[dict[str, dict]] = {
    "dev": {"gpus": 1, "type": "H100", "gpu_profile": "h100_80gb"},  # H100:1 on Nebius
    "small": {"gpus": 1, "type": "H100", "gpu_profile": "h100_80gb"},
    "medium": {"gpus": 2, "type": "H100", "gpu_profile": "h100_80gb"},
    "large": {"gpus": 4, "type": "H100", "gpu_profile": "h100_80gb"},
    "xlarge": {"gpus": 8, "type": "H100", "gpu_profile": "h100_80gb"},
}
"""GPU scale profiles for training jobs."""

# Model-specific default scales
MODEL_SCALES: Final[dict[str, str]] = {
    "smollm2_135m": "dev",     # Small model, single cheap GPU
    "qwen3_4b": "medium",      # 4B model, 2x H100 recommended
}
"""Recommended scale profiles for specific models."""

DEFAULT_SCALE: Final[str] = "dev"
"""Default scale profile (uses A10G, cheaper for testing)."""


def get_scale_for_model(model: str) -> str:
    """Get the recommended scale for a model.

    Args:
        model: Model config name (e.g., "qwen3_4b", "smollm2_135m")

    Returns:
        Scale profile name (e.g., "dev", "medium", "large")
    """
    return MODEL_SCALES.get(model, DEFAULT_SCALE)

# =============================================================================
# Stage to Config Mapping
# =============================================================================
# Training stages map to Hydra config files in WrinkleFree-1.58Quant/configs/training/
# Stage 1: Convert FP16 model to initial 1.58-bit (SubLN initialization)
# Stage 1.9: Layer-wise distillation to refine initial conversion
# Stage 2: Pre-training on large corpus (main training)
# Stage 3: Distillation from teacher model (final refinement)

STAGE_CONFIG_MAP: Final[dict[float, str]] = {
    1: "stage1_subln",
    1.9: "stage1_9_layerwise",
    2: "stage2_pretrain",
    3: "stage3_distill",
}
"""Maps training stage number to Hydra training config name."""

# =============================================================================
# Environment Variables
# =============================================================================
# Centralized env var names for consistency


class EnvVars:
    """Environment variable names used by the deployer."""

    # API tokens
    GH_TOKEN = "GH_TOKEN"
    WANDB_API_KEY = "WANDB_API_KEY"
    WANDB_ENTITY = "WANDB_ENTITY"

    # HuggingFace
    HF_HOME = "HF_HOME"
    TRANSFORMERS_CACHE = "TRANSFORMERS_CACHE"
    HF_HUB_ENABLE_HF_TRANSFER = "HF_HUB_ENABLE_HF_TRANSFER"


def get_wandb_entity() -> str | None:
    """Get W&B entity from environment variable.

    Returns:
        W&B entity (user or team name), or None if not set.
    """
    return os.environ.get(EnvVars.WANDB_ENTITY)
