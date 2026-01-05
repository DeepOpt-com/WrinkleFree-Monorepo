"""Lambda warmup for gradual quantization.

Implements the gradual quantization approach from HuggingFace's BitNet 1.58-bit fine-tuning:
- Start with lambda=0 (full precision)
- Gradually increase to lambda=1 (full quantization)
- Prevents catastrophic forgetting of pre-trained knowledge

Reference: https://huggingface.co/blog/1_58_llm_extreme_quantization
"""

from __future__ import annotations

import logging
import math
from typing import Literal

logger = logging.getLogger(__name__)


class LambdaWarmup:
    """
    Manages lambda warmup schedule for gradual quantization.

    During training, lambda starts at 0 (full precision) and gradually
    increases to 1 (full quantization) over warmup_steps.

    This prevents the model from "forgetting" its pre-trained knowledge
    when quantization is applied.

    Args:
        warmup_steps: Number of steps to reach full quantization (lambda=1)
        schedule: "linear" or "cosine" warmup schedule
        min_lambda: Minimum lambda value (default 0)
        max_lambda: Maximum lambda value (default 1)
    """

    def __init__(
        self,
        warmup_steps: int = 1000,
        schedule: Literal["linear", "cosine"] = "linear",
        min_lambda: float = 0.0,
        max_lambda: float = 1.0,
    ):
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self._current_step = 0
        self._lambda = min_lambda

    def step(self) -> None:
        """Advance the warmup schedule by one step."""
        self._current_step += 1
        self._update_lambda()

    def _update_lambda(self) -> None:
        """Update lambda based on current step and schedule."""
        if self._current_step >= self.warmup_steps:
            self._lambda = self.max_lambda
            return

        progress = self._current_step / self.warmup_steps

        if self.schedule == "linear":
            self._lambda = self.min_lambda + progress * (self.max_lambda - self.min_lambda)
        elif self.schedule == "cosine":
            # Cosine schedule: slower at start and end, faster in middle
            cosine_progress = (1 - math.cos(math.pi * progress)) / 2
            self._lambda = self.min_lambda + cosine_progress * (self.max_lambda - self.min_lambda)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    @property
    def lambda_val(self) -> float:
        """Current lambda value for quantization mixing."""
        return self._lambda

    @property
    def current_step(self) -> int:
        """Current step in the warmup schedule."""
        return self._current_step

    def is_warmup_complete(self) -> bool:
        """Check if warmup is complete (lambda reached max)."""
        return self._current_step >= self.warmup_steps

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "current_step": self._current_step,
            "lambda": self._lambda,
            "warmup_steps": self.warmup_steps,
            "schedule": self.schedule,
            "min_lambda": self.min_lambda,
            "max_lambda": self.max_lambda,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self._current_step = state_dict["current_step"]
        self._lambda = state_dict["lambda"]
        # Optionally restore config (useful for resuming)
        if "warmup_steps" in state_dict:
            self.warmup_steps = state_dict["warmup_steps"]
        if "schedule" in state_dict:
            self.schedule = state_dict["schedule"]


# Global warmup instance (shared across all BitLinear layers)
_global_lambda_warmup: LambdaWarmup | None = None


def get_global_lambda_warmup() -> LambdaWarmup | None:
    """Get the global lambda warmup instance."""
    return _global_lambda_warmup


def set_global_lambda_warmup(warmup: LambdaWarmup | None) -> None:
    """Set the global lambda warmup instance."""
    global _global_lambda_warmup
    _global_lambda_warmup = warmup
    if warmup is not None:
        logger.info(
            f"Lambda warmup enabled: {warmup.warmup_steps} steps, "
            f"schedule={warmup.schedule}"
        )


def get_current_lambda() -> float:
    """Get current lambda value from global warmup, or 1.0 if not set."""
    if _global_lambda_warmup is None:
        return 1.0  # Full quantization when warmup not enabled
    return _global_lambda_warmup.lambda_val
