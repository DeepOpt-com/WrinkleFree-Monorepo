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

        # Validate objective compatibility
        self._validate_objective_compatibility()

    def _validate_objective_compatibility(self) -> None:
        """Check for incompatible objective combinations and warn."""
        # Check for DLM + logits_distill with shift_labels (incompatible)
        has_dlm = "dlm" in self.objectives
        has_logits_distill = "logits_distill" in self.objectives

        if has_dlm and has_logits_distill:
            logits_distill = self.objectives["logits_distill"]
            # Check if shift_labels is True (default for AR-to-AR distillation)
            shift_labels = getattr(logits_distill, "shift_labels", True)
            if shift_labels:
                logger.warning(
                    "Incompatible objectives detected: 'dlm' + 'logits_distill' with shift_labels=True. "
                    "AR-style logits distillation with token shifting is incompatible with DLM's "
                    "masked prediction paradigm. Consider using 'tcs_distill' instead (no shifting), "
                    "or disable one of the objectives."
                )

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

        Objectives with modifies_input=True get to modify the batch,
        but ONLY if their current weight is > 0. This prevents DLM from
        masking inputs during warmup phases when DLM weight is 0.

        Args:
            batch: Input batch dictionary

        Returns:
            Preprocessed batch (may be modified in place)
        """
        if not self._has_input_modifiers:
            return batch

        # Get current weights to check which objectives are active
        weights = self.get_current_weights()

        for name, obj in self.objectives.items():
            if obj.modifies_input:
                # Only apply preprocessing if objective weight > 0
                obj_weight = weights.get(name, self.base_weights.get(name, 1.0))
                if obj_weight > 0:
                    batch = obj.preprocess_batch(batch)
                else:
                    logger.debug(f"Skipping {name} preprocessing (weight=0)")

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

    # Note: We don't override state_dict/load_state_dict because:
    # 1. ObjectiveManager doesn't have trainable parameters
    # 2. Custom keys (_objectives, _curriculum) cause Lightning checkpoint issues
    # 3. Objectives and curriculum can be recreated from config
    # 4. Curriculum step is synced via set_current_step() from Lightning's global_step

    # =========================================================================
    # Meta-optimization support
    # =========================================================================

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set objective weights from meta-optimizer.

        Used by MetaOptimizerCallback to update weights during training.
        This overrides the curriculum weights until the next curriculum step.

        Args:
            weights: Dictionary mapping objective name to weight
        """
        for name, weight in weights.items():
            if name in self.base_weights:
                self.base_weights[name] = weight
                logger.debug(f"Meta-optimizer set {name} weight to {weight}")

    def get_objective_names(self) -> list[str]:
        """Get list of objective names.

        Returns:
            List of objective names in this manager
        """
        return list(self.objectives.keys())

    def get_objective_gradients(self) -> dict[str, torch.Tensor]:
        """Get cached per-objective gradients.

        These are the gradients of each objective's loss w.r.t. model parameters,
        computed during the last forward pass. Used by meta-optimization to
        estimate how objective weights affect validation performance.

        Note: This returns gradients from the last forward pass. Call this
        after backward() but before optimizer.step().

        Returns:
            Dictionary mapping objective name to flattened gradient tensor
        """
        if not hasattr(self, "_cached_objective_gradients"):
            return {}
        return self._cached_objective_gradients

    def cache_objective_gradients(self, model: nn.Module) -> None:
        """Cache gradients from each objective for meta-optimization.

        This should be called after backward() on the combined loss.
        It's an approximation: we can't get exact per-objective gradients
        without multiple backward passes. Instead, we cache the current
        gradient and use it as a proxy.

        For more accurate per-objective gradients, one would need to:
        1. Compute each objective loss separately
        2. Call backward() on each
        3. Accumulate gradients

        This approximation uses the combined gradient, which is a weighted
        sum of the per-objective gradients.

        Args:
            model: Model with gradients computed
        """
        # For now, we cache a single gradient representing all objectives
        # A more sophisticated version would track per-objective contributions
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.flatten())

        if grads:
            combined = torch.cat(grads)
            # Store as single combined gradient (approximation)
            self._cached_objective_gradients = {
                name: combined for name in self.objectives
            }
        else:
            self._cached_objective_gradients = {}

