"""Factory function for creating ObjectiveManager from Hydra config.

This provides the main entry point for creating the objectives system
from a Hydra configuration dictionary.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from wrinklefree.objectives.base import Objective
from wrinklefree.objectives.continue_pretrain import ContinuePretrainObjective
from wrinklefree.objectives.dlm import DLMObjective
from wrinklefree.objectives.layerwise import LayerwiseDistillationObjective, LayerwiseLossType
from wrinklefree.objectives.logits_distill import LogitsDistillationObjective
from wrinklefree.objectives.attention_distill import AttentionRelationDistillationObjective
from wrinklefree.objectives.tcs_distill import TCSDistillationObjective
from wrinklefree.objectives.block_attention_distill import BlockAttentionDistillationObjective
from wrinklefree.objectives.bitdistill import BitDistillObjective
from wrinklefree.objectives.lrc_reconstruction import LRCReconstructionObjective, LRCLossType
from wrinklefree.objectives.manager import (
    CurriculumPhase,
    CurriculumScheduler,
    ObjectiveManager,
)

logger = logging.getLogger(__name__)


def create_objective(name: str, config: dict[str, Any]) -> Objective:
    """Create a single objective from config.

    Args:
        name: Objective type name
        config: Objective-specific configuration

    Returns:
        Initialized Objective instance
    """
    if name == "continue_pretrain":
        return ContinuePretrainObjective(
            ignore_index=config.get("ignore_index", -100),
            label_smoothing=config.get("label_smoothing", 0.0),
        )
    elif name == "dlm":
        return DLMObjective(
            mask_token_id=config["mask_token_id"],
            mask_prob=config.get("mask_prob", 0.15),
            ignore_index=config.get("ignore_index", -100),
            use_complementary_masks=config.get("use_complementary_masks", True),
        )
    elif name == "layerwise_distill":
        loss_type = config.get("loss_type", "mse_normalized")
        return LayerwiseDistillationObjective(
            loss_type=LayerwiseLossType(loss_type) if isinstance(loss_type, str) else loss_type,
            layer_weights=config.get("layer_weights"),
            normalize=config.get("normalize", True),
        )
    elif name == "logits_distill":
        return LogitsDistillationObjective(
            temperature=config.get("temperature", 5.0),
            ignore_index=config.get("ignore_index", -100),
            shift_labels=config.get("shift_labels", True),
        )
    elif name == "attention_distill":
        return AttentionRelationDistillationObjective(
            distill_layer=config.get("distill_layer", -1),
            temperature=config.get("temperature", 1.0),
            ignore_index=config.get("ignore_index", -100),
        )
    elif name == "tcs_distill":
        return TCSDistillationObjective(
            temperature=config.get("temperature", 5.0),
            top_k=config.get("top_k", 100),
            ignore_index=config.get("ignore_index", -100),
        )
    elif name == "block_attention_distill":
        return BlockAttentionDistillationObjective(
            block_size=config.get("block_size", 32),
            distill_layer=config.get("distill_layer", -1),
            ignore_index=config.get("ignore_index", -100),
        )
    elif name == "bitdistill":
        return BitDistillObjective(
            lambda_logits=config.get("lambda_logits", 10.0),
            gamma_attention=config.get("gamma_attention", 1e-5),
            temperature=config.get("temperature", 5.0),
            distill_layer=config.get("distill_layer", -1),
            ignore_index=config.get("ignore_index", -100),
        )
    elif name == "lrc_reconstruction":
        loss_type = config.get("loss_type", "mse")
        return LRCReconstructionObjective(
            loss_type=LRCLossType(loss_type) if isinstance(loss_type, str) else loss_type,
            layer_weights=config.get("layer_weights"),
            temperature=config.get("temperature", 1.0),
            normalize=config.get("normalize", False),
        )
    else:
        raise ValueError(f"Unknown objective type: {name}")


def create_curriculum_scheduler(
    curriculum_config: dict[str, Any],
    total_steps: int,
) -> Optional[CurriculumScheduler]:
    """Create CurriculumScheduler from config.

    Args:
        curriculum_config: Curriculum configuration with 'phases' list
        total_steps: Total training steps

    Returns:
        CurriculumScheduler or None if curriculum disabled
    """
    if not curriculum_config.get("enabled", False):
        return None

    phases_config = curriculum_config.get("phases", [])
    if not phases_config:
        return None

    phases = []
    for phase_cfg in phases_config:
        phases.append(
            CurriculumPhase(
                name=phase_cfg["name"],
                end_ratio=phase_cfg["end_ratio"],
                objective_weights=dict(phase_cfg.get("objectives", {})),
                data_config=phase_cfg.get("data_config"),
            )
        )

    return CurriculumScheduler(
        phases=phases,
        total_steps=total_steps,
        interpolation=curriculum_config.get("interpolation", "linear"),
    )


def create_objective_manager(
    config: DictConfig | dict[str, Any],
    total_steps: int,
) -> ObjectiveManager:
    """Create ObjectiveManager from Hydra config.

    Expected config structure:
    ```yaml
    objectives:
      continue_pretrain:
        enabled: true
        weight: 1.0
        # objective-specific params...
      layerwise_distill:
        enabled: true
        weight: 0.5
        loss_type: mse_normalized
        layer_weights: progressive

    curriculum:
      enabled: false
      interpolation: linear
      phases:
        - name: warmup
          end_ratio: 0.2
          objectives: {continue_pretrain: 1.0}
        - name: main
          end_ratio: 1.0
          objectives: {continue_pretrain: 1.0, layerwise_distill: 0.5}
    ```

    Args:
        config: Hydra configuration (DictConfig or dict)
        total_steps: Total training steps

    Returns:
        Configured ObjectiveManager
    """
    # Convert DictConfig to dict for easier handling
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    objectives_config = config.get("objectives", {})
    curriculum_config = config.get("curriculum", {})

    # Create enabled objectives
    objectives: dict[str, Objective] = {}
    weights: dict[str, float] = {}

    for name, obj_config in objectives_config.items():
        if not obj_config.get("enabled", True):
            continue

        obj = create_objective(name, obj_config)
        objectives[name] = obj
        weights[name] = obj_config.get("weight", 1.0)

    if not objectives:
        # Default to continue_pretrain if nothing specified
        logger.warning("No objectives configured, using default continue_pretrain")
        objectives["continue_pretrain"] = ContinuePretrainObjective()
        weights["continue_pretrain"] = 1.0

    # Create curriculum scheduler
    curriculum = create_curriculum_scheduler(curriculum_config, total_steps)

    return ObjectiveManager(
        objectives=objectives,
        weights=weights,
        curriculum=curriculum,
    )

