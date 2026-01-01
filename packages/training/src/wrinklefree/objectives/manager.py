"""Objective manager and curriculum scheduler.

Manages multiple training objectives with configurable weights,
and supports curriculum learning with phase-based weight adjustments.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch
import torch.nn as nn

from wrinklefree.objectives.base import Objective, ObjectiveOutput

logger = logging.getLogger(__name__)


@dataclass
class CurriculumPhase:
    """A phase in the curriculum schedule.

    Attributes:
        name: Phase identifier
        end_ratio: Training progress ratio when phase ends (0.0-1.0)
        objective_weights: Weight for each objective in this phase
        data_config: Optional data configuration name for this phase
    """

    name: str
    end_ratio: float
    objective_weights: dict[str, float]
    data_config: Optional[str] = None


@dataclass
class ManagerOutput:
    """Output from ObjectiveManager.forward().

    Attributes:
        loss: Combined weighted loss
        objective_outputs: Individual objective outputs
        weights_used: Weights applied to each objective
        ce_loss: Unweighted cross-entropy loss (for logging)
        perplexity: Perplexity from CE only (for logging)
    """

    loss: torch.Tensor
    objective_outputs: dict[str, ObjectiveOutput]
    weights_used: dict[str, float]
    ce_loss: Optional[torch.Tensor] = None
    perplexity: Optional[torch.Tensor] = None


class CurriculumScheduler:
    """Manages training curriculum with phase-based objective weights.

    Supports smooth interpolation between phases using linear or cosine
    schedules. Can also switch data configurations between phases.

    Args:
        phases: List of CurriculumPhase defining the curriculum
        total_steps: Total training steps
        interpolation: "linear" or "cosine" between phases
    """

    def __init__(
        self,
        phases: list[CurriculumPhase],
        total_steps: int,
        interpolation: Literal["linear", "cosine"] = "linear",
    ):
        self.phases = sorted(phases, key=lambda p: p.end_ratio)
        self.total_steps = total_steps
        self.interpolation = interpolation
        self._current_step = 0

        # Validate phases
        if not phases:
            raise ValueError("At least one phase required")
        if self.phases[-1].end_ratio != 1.0:
            logger.warning(
                f"Last phase ends at {self.phases[-1].end_ratio}, not 1.0. "
                "Final phase weights will be used for remaining training."
            )

    @property
    def current_step(self) -> int:
        return self._current_step

    def step(self) -> None:
        """Advance the scheduler by one step."""
        self._current_step += 1

    def get_current_phase(self) -> CurriculumPhase:
        """Get the current phase based on training progress."""
        progress = self._current_step / self.total_steps

        for phase in self.phases:
            if progress <= phase.end_ratio:
                return phase

        return self.phases[-1]  # Return last phase if past all phases

    def get_weights(self) -> dict[str, float]:
        """Get current objective weights, with interpolation between phases.

        For smooth transitions, weights are interpolated between the
        previous and current phase based on progress within the phase.
        """
        progress = self._current_step / self.total_steps

        # Find current and previous phases
        current_phase = None
        prev_phase = None

        for i, phase in enumerate(self.phases):
            if progress <= phase.end_ratio:
                current_phase = phase
                if i > 0:
                    prev_phase = self.phases[i - 1]
                break

        if current_phase is None:
            return self.phases[-1].objective_weights

        # If no previous phase, use current weights directly
        if prev_phase is None:
            return current_phase.objective_weights

        # Calculate interpolation factor within current phase
        phase_start = prev_phase.end_ratio
        phase_end = current_phase.end_ratio
        phase_progress = (progress - phase_start) / (phase_end - phase_start)

        if self.interpolation == "cosine":
            # Smooth cosine interpolation
            factor = (1 - math.cos(math.pi * phase_progress)) / 2
        else:
            # Linear interpolation
            factor = phase_progress

        # Interpolate weights
        weights = {}
        all_objectives = set(prev_phase.objective_weights.keys()) | set(
            current_phase.objective_weights.keys()
        )

        for obj_name in all_objectives:
            prev_weight = prev_phase.objective_weights.get(obj_name, 0.0)
            curr_weight = current_phase.objective_weights.get(obj_name, 0.0)
            weights[obj_name] = prev_weight + factor * (curr_weight - prev_weight)

        return weights

    def get_data_config(self) -> Optional[str]:
        """Get current data configuration name."""
        return self.get_current_phase().data_config

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {"current_step": self._current_step}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self._current_step = state_dict["current_step"]


class ObjectiveManager(nn.Module):
    """Manages multiple training objectives with weighted combination.

    Handles:
    - Combining multiple objectives with configurable weights
    - Batch preprocessing for objectives that modify inputs
    - Tracking which objectives need teacher/hidden states
    - Generating wandb-compatible metrics

    Args:
        objectives: Dictionary of name -> Objective
        weights: Dictionary of name -> weight (default: 1.0 for all)
        curriculum: Optional CurriculumScheduler for dynamic weights
    """

    def __init__(
        self,
        objectives: dict[str, Objective],
        weights: Optional[dict[str, float]] = None,
        curriculum: Optional[CurriculumScheduler] = None,
    ):
        super().__init__()
        self.objectives = nn.ModuleDict(objectives)
        self.base_weights = weights or {name: 1.0 for name in objectives}
        self.curriculum = curriculum

        # Track capabilities
        self._requires_teacher = any(obj.requires_teacher for obj in objectives.values())
        self._requires_hidden_states = any(
            obj.requires_hidden_states for obj in objectives.values()
        )
        self._requires_attentions = any(
            getattr(obj, "requires_attentions", False) for obj in objectives.values()
        )
        self._has_input_modifiers = any(obj.modifies_input for obj in objectives.values())

        logger.info(f"ObjectiveManager initialized with {len(objectives)} objectives:")
        for name, obj in objectives.items():
            weight = self.base_weights.get(name, 1.0)
            logger.info(f"  - {name}: weight={weight}, {obj.extra_repr()}")

    @property
    def requires_teacher(self) -> bool:
        """Whether any objective requires teacher outputs."""
        return self._requires_teacher

    @property
    def requires_hidden_states(self) -> bool:
        """Whether any objective requires hidden states."""
        return self._requires_hidden_states

    @property
    def requires_attentions(self) -> bool:
        """Whether any objective requires attention weights."""
        return self._requires_attentions

    @property
    def any_modifies_input(self) -> bool:
        """Whether any objective modifies the input batch."""
        return self._has_input_modifiers

    def get_current_weights(self) -> dict[str, float]:
        """Get current objective weights (from curriculum or base)."""
        if self.curriculum is not None:
            return self.curriculum.get_weights()
        return self.base_weights

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply all objective preprocessing to batch.

        Objectives with modifies_input=True get to modify the batch.
        If multiple objectives modify inputs, they are applied in order.

        Args:
            batch: Input batch dictionary

        Returns:
            Preprocessed batch (may be modified in place)
        """
        if not self._has_input_modifiers:
            return batch

        for name, obj in self.objectives.items():
            if obj.modifies_input:
                batch = obj.preprocess_batch(batch)

        return batch

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ManagerOutput:
        """Compute combined loss from all objectives.

        Args:
            model_outputs: Model forward pass outputs
            batch: Input batch (should be preprocessed)
            teacher_outputs: Optional teacher outputs

        Returns:
            ManagerOutput with combined loss and per-objective details
        """
        weights = self.get_current_weights()
        device = model_outputs["logits"].device

        objective_outputs: dict[str, ObjectiveOutput] = {}
        total_loss = torch.tensor(0.0, device=device, dtype=model_outputs["logits"].dtype)
        ce_loss = None
        perplexity = None

        for name, obj in self.objectives.items():
            weight = weights.get(name, 0.0)
            if weight <= 0:
                continue  # Skip disabled objectives

            # Compute objective
            output = obj(model_outputs, batch, teacher_outputs)
            objective_outputs[name] = output

            # Add weighted loss
            total_loss = total_loss + weight * output.loss

            # Track CE loss for unweighted logging
            if output.ce_loss is not None and ce_loss is None:
                ce_loss = output.ce_loss
                perplexity = output.metrics.get("perplexity")

        return ManagerOutput(
            loss=total_loss,
            objective_outputs=objective_outputs,
            weights_used=weights,
            ce_loss=ce_loss,
            perplexity=perplexity,
        )

    def step_curriculum(self) -> None:
        """Advance curriculum scheduler if present."""
        if self.curriculum is not None:
            self.curriculum.step()

    def get_wandb_metrics(self, output: ManagerOutput, prefix: str = "train") -> dict[str, float]:
        """Generate wandb-compatible metrics dictionary.

        Includes:
        - Combined loss
        - Unweighted CE loss and perplexity (for fair comparison)
        - Per-objective losses and metrics
        - Current objective weights

        Args:
            output: ManagerOutput from forward()
            prefix: Metric prefix (default: "train")

        Returns:
            Dictionary of metric_name -> value
        """
        metrics = {
            f"{prefix}/loss": output.loss.item(),
        }

        # Unweighted CE loss (important for comparing runs with different objectives)
        if output.ce_loss is not None:
            metrics[f"{prefix}/loss_unweighted_ce"] = output.ce_loss.item()
        if output.perplexity is not None:
            metrics[f"{prefix}/perplexity"] = output.perplexity.item()

        # Per-objective losses and metrics
        for name, obj_output in output.objective_outputs.items():
            metrics[f"{prefix}/{name}_loss"] = obj_output.loss.item()
            for metric_name, value in obj_output.metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[f"{prefix}/{name}_{metric_name}"] = value.item()
                else:
                    metrics[f"{prefix}/{name}_{metric_name}"] = value

        # Current weights (for tracking curriculum)
        for name, weight in output.weights_used.items():
            metrics[f"schedule/{name}_weight"] = weight

        return metrics

    def state_dict(self, destination=None, prefix='', keep_vars=False) -> dict:
        """Get state for checkpointing (PyTorch-compatible signature)."""
        # Get standard nn.Module state (for any registered parameters/buffers)
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Add custom objective state
        state["_objectives"] = {name: obj.state_dict() for name, obj in self.objectives.items()}
        if self.curriculum is not None:
            state["_curriculum"] = self.curriculum.state_dict()
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load state from checkpoint (PyTorch-compatible signature)."""
        # Load custom objective state
        if "_objectives" in state_dict:
            for name, obj_state in state_dict["_objectives"].items():
                if name in self.objectives:
                    self.objectives[name].load_state_dict(obj_state)
        if "_curriculum" in state_dict and self.curriculum is not None:
            self.curriculum.load_state_dict(state_dict["_curriculum"])
        # Load standard nn.Module state (filter out custom keys)
        module_state = {k: v for k, v in state_dict.items() if not k.startswith("_")}
        if module_state:
            super().load_state_dict(module_state, strict=strict)
