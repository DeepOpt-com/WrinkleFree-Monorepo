"""Factory function for creating ObjectiveManager from Hydra config.

This provides the main entry point for creating the objectives system
from a Hydra configuration dictionary.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from wf_train.objectives.base import Objective
from wf_train.objectives.continue_pretrain import ContinuePretrainObjective
from wf_train.objectives.dlm import DLMObjective
from wf_train.objectives.distill import (
    DistillObjective,
    HiddenConfig,
    LogitsConfig,
    AttentionConfig,
    LRCConfig,
)
from wf_train.objectives.manager import (
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
    elif name == "distill":
        return _create_distill_objective(config)
    else:
        raise ValueError(f"Unknown objective type: {name}")


def _create_distill_objective(config: dict[str, Any]) -> DistillObjective:
    """Create DistillObjective from config.

    Args:
        config: Distill config with hidden, logits, attention, lrc sub-configs

    Returns:
        Configured DistillObjective
    """
    hidden = None
    if "hidden" in config:
        hidden_cfg = config["hidden"]
        hidden = HiddenConfig(
            enabled=hidden_cfg.get("enabled", False),
            weight=hidden_cfg.get("weight", 1.0),
            loss_type=hidden_cfg.get("loss_type", "mse_normalized"),
            layer_weights=hidden_cfg.get("layer_weights"),
            normalize=hidden_cfg.get("normalize", True),
        )

    logits = None
    if "logits" in config:
        logits_cfg = config["logits"]
        logits = LogitsConfig(
            enabled=logits_cfg.get("enabled", False),
            weight=logits_cfg.get("weight", 10.0),
            temperature=logits_cfg.get("temperature", 5.0),
            mode=logits_cfg.get("mode", "full"),
            top_k=logits_cfg.get("top_k", 100),
            shift_labels=logits_cfg.get("shift_labels", True),
            ignore_index=logits_cfg.get("ignore_index", -100),
        )

    attention = None
    if "attention" in config:
        attn_cfg = config["attention"]
        attention = AttentionConfig(
            enabled=attn_cfg.get("enabled", False),
            weight=attn_cfg.get("weight", 1.0e-5),
            distill_layer=attn_cfg.get("distill_layer", -1),
            mode=attn_cfg.get("mode", "relation"),
            block_size=attn_cfg.get("block_size", 32),
            temperature=attn_cfg.get("temperature", 1.0),
        )

    lrc = None
    if "lrc" in config:
        lrc_cfg = config["lrc"]
        lrc = LRCConfig(
            enabled=lrc_cfg.get("enabled", False),
            weight=lrc_cfg.get("weight", 1.0),
            loss_type=lrc_cfg.get("loss_type", "mse"),
            layer_weights=lrc_cfg.get("layer_weights"),
            temperature=lrc_cfg.get("temperature", 1.0),
            normalize=lrc_cfg.get("normalize", False),
        )

    return DistillObjective(
        hidden=hidden,
        logits=logits,
        attention=attention,
        lrc=lrc,
        ignore_index=config.get("ignore_index", -100),
    )


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
        interpolation=curriculum_config.get("interpolation", "step"),
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
      interpolation: step  # "step" (immediate), "linear", or "cosine"
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

