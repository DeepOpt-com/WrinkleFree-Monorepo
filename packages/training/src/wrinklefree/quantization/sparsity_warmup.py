"""Sparsity warmup for gradual activation sparsification.

Similar to lambda warmup for quantization, this gradually increases
sparsity ratio from 0 (dense) to target (e.g., 0.61 for 1-bit models).

This prevents the model from experiencing a sudden change in activation
patterns during training.

Reference: Q-Sparse (arxiv.org/abs/2407.10969)
"""

from __future__ import annotations

import logging
import math
from typing import Literal

logger = logging.getLogger(__name__)


class SparsityWarmup:
    """Manages sparsity warmup schedule for gradual activation sparsification.

    During training, sparsity starts at initial_sparsity (default 0, dense)
    and gradually increases to target_sparsity over warmup_steps.

    Args:
        warmup_steps: Number of steps to reach target sparsity
        schedule: "linear" or "cosine" warmup schedule
        initial_sparsity: Starting sparsity ratio (default 0 = dense)
        target_sparsity: Final sparsity ratio (default 0.61 for 1-bit models)
    """

    def __init__(
        self,
        warmup_steps: int = 1000,
        schedule: Literal["linear", "cosine"] = "linear",
        initial_sparsity: float = 0.0,
        target_sparsity: float = 0.61,
    ):
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self._current_step = 0
        self._sparsity = initial_sparsity

    def step(self) -> None:
        """Advance the warmup schedule by one step."""
        self._current_step += 1
        self._update_sparsity()

    def _update_sparsity(self) -> None:
        """Update sparsity based on current step and schedule."""
        if self._current_step >= self.warmup_steps:
            self._sparsity = self.target_sparsity
            return

        progress = self._current_step / self.warmup_steps

        if self.schedule == "linear":
            self._sparsity = self.initial_sparsity + progress * (
                self.target_sparsity - self.initial_sparsity
            )
        elif self.schedule == "cosine":
            # Cosine schedule: slower at start and end, faster in middle
            cosine_progress = (1 - math.cos(math.pi * progress)) / 2
            self._sparsity = self.initial_sparsity + cosine_progress * (
                self.target_sparsity - self.initial_sparsity
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    @property
    def sparsity(self) -> float:
        """Current sparsity ratio."""
        return self._sparsity

    @property
    def current_step(self) -> int:
        """Current step in the warmup schedule."""
        return self._current_step

    def is_warmup_complete(self) -> bool:
        """Check if warmup is complete (sparsity reached target)."""
        return self._current_step >= self.warmup_steps

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "current_step": self._current_step,
            "sparsity": self._sparsity,
            "warmup_steps": self.warmup_steps,
            "schedule": self.schedule,
            "initial_sparsity": self.initial_sparsity,
            "target_sparsity": self.target_sparsity,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self._current_step = state_dict["current_step"]
        self._sparsity = state_dict["sparsity"]
        # Optionally restore config (useful for resuming)
        if "warmup_steps" in state_dict:
            self.warmup_steps = state_dict["warmup_steps"]
        if "schedule" in state_dict:
            self.schedule = state_dict["schedule"]
        if "initial_sparsity" in state_dict:
            self.initial_sparsity = state_dict["initial_sparsity"]
        if "target_sparsity" in state_dict:
            self.target_sparsity = state_dict["target_sparsity"]


# Global warmup instance (shared across all BitLinear layers)
_global_sparsity_warmup: SparsityWarmup | None = None


def get_global_sparsity_warmup() -> SparsityWarmup | None:
    """Get the global sparsity warmup instance."""
    return _global_sparsity_warmup


def set_global_sparsity_warmup(warmup: SparsityWarmup | None) -> None:
    """Set the global sparsity warmup instance."""
    global _global_sparsity_warmup
    _global_sparsity_warmup = warmup
    if warmup is not None:
        logger.info(
            f"Sparsity warmup enabled: {warmup.warmup_steps} steps, "
            f"target={warmup.target_sparsity}, schedule={warmup.schedule}"
        )


def get_current_sparsity() -> float:
    """Get current sparsity ratio from global warmup, or 0.0 if not set."""
    if _global_sparsity_warmup is None:
        return 0.0  # No sparsity when warmup not enabled
    return _global_sparsity_warmup.sparsity
