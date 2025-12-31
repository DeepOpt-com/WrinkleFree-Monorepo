"""Stage 1.9: Layer-wise Distillation - DEPRECATED.

This module is deprecated. Use run_stage2 with pre_stage_2.enabled=true instead.

The functionality has been merged into Stage 2 for a unified training pipeline.
To enable layer-wise distillation in Stage 2:

    training.pre_stage_2.enabled=true

This wrapper is kept for backward compatibility only.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def run_stage1_9(
    student_model: nn.Module,
    teacher_model_name: str,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    config: DictConfig,
    layerwise_config: DictConfig,
    output_dir: Path,
    resume_from: Optional[Path] = None,
    run_manager: Optional[Any] = None,
    experiment_name: Optional[str] = None,
) -> nn.Module:
    """
    Run Stage 1.9: Layer-wise distillation.

    DEPRECATED: This function is deprecated. Use run_stage2 with
    pre_stage_2.enabled=true instead.

    This wrapper converts the old stage1_9 config format to the new
    unified stage2 format and calls run_stage2.
    """
    warnings.warn(
        "run_stage1_9 is deprecated. Use run_stage2 with "
        "training.pre_stage_2.enabled=true instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    logger.warning(
        "=" * 60 + "\n"
        "DEPRECATION WARNING: Stage 1.9 is deprecated.\n"
        "Use run_stage2 with training.pre_stage_2.enabled=true instead.\n"
        "This call is being redirected to run_stage2 with pre_stage_2 mode.\n"
        + "=" * 60
    )

    # Import here to avoid circular imports
    from wrinklefree.training.continued_pretraining import run_stage2

    # Convert old layerwise_config to new pre_stage_2 format
    # The new config expects pre_stage_2 under config.training
    if not hasattr(config, "training"):
        # Config is already training config
        training_config = config
    else:
        training_config = config.training

    # Build pre_stage_2 config from layerwise_config
    pre_stage_2_dict = {
        "enabled": True,
        "teacher": {
            "fp16": layerwise_config.get("teacher_fp16", True),
            "offload_to_cpu": layerwise_config.get("teacher_offload", False),
            "load_in_4bit": layerwise_config.get("teacher_4bit", False),
            "use_flash_attention": layerwise_config.get("teacher_flash_attention", False),
        },
        "layerwise": {
            "loss_type": layerwise_config.get("loss_type", "mse_normalized"),
            "layer_weights": layerwise_config.get("layer_weights", "progressive"),
            "normalize": layerwise_config.get("normalize", True),
            "temperature": layerwise_config.get("temperature", 1.0),
            "lm_loss_weight": layerwise_config.get("lm_loss_weight", 0.5),
            "hidden_size": layerwise_config.get("hidden_size"),
            "vocab_size": layerwise_config.get("vocab_size"),
        },
        "distill_schedule": layerwise_config.get("distill_schedule", {
            "enabled": True,
            "type": "cosine",
            "initial_weight": 0.5,
            "final_weight": 0.0,
            "warmup_steps": 0,
        }),
    }

    # Merge pre_stage_2 config into training config
    if hasattr(training_config, "_content"):
        # OmegaConf DictConfig
        with OmegaConf.read_write(training_config):
            training_config.pre_stage_2 = OmegaConf.create(pre_stage_2_dict)
    else:
        training_config.pre_stage_2 = pre_stage_2_dict

    # Call run_stage2 with the converted config
    return run_stage2(
        model=student_model,
        train_dataloader=train_dataloader,
        config=config,
        output_dir=output_dir,
        resume_from=resume_from,
        run_manager=run_manager,
        experiment_name=experiment_name,
        teacher_model_name=teacher_model_name,
    )
