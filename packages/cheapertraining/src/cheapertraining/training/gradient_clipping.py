"""Adaptive gradient clipping utilities for stable LLM training.

This module provides ZClip, an adaptive gradient clipping algorithm that uses
z-score based anomaly detection to prevent gradient spikes during training.

Reference:
    ZClip: Adaptive Gradient Clipping for Pre-Training LLMs
    arXiv:2504.02507 (2025)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ZClipStats:
    """Statistics tracked by ZClip for monitoring.

    Attributes:
        raw_norm: The unclipped gradient norm
        clipped_norm: The gradient norm after clipping (equals raw_norm if no clipping)
        z_score: The z-score of the gradient norm (-1 if first step)
        was_clipped: Whether clipping was applied
        ema_mean: Current EMA mean of gradient norms
        ema_std: Current EMA standard deviation of gradient norms
    """

    raw_norm: float
    clipped_norm: float
    z_score: float
    was_clipped: bool
    ema_mean: float
    ema_std: float


class ZClip:
    """Adaptive gradient clipping using z-score anomaly detection.

    ZClip dynamically adjusts the clipping threshold based on gradient norm
    statistics. Instead of using a fixed clipping threshold (e.g., 1.0), it:

    1. Maintains an EMA (exponential moving average) of gradient norms
    2. Computes a z-score for each gradient norm
    3. Only clips when the z-score exceeds a threshold (default: 3.0)

    This approach is more robust than fixed clipping because:
    - It adapts to the natural scale of gradients for each model
    - It only clips true anomalies (spikes), not normal large gradients
    - It prevents both under-clipping (spikes cause instability) and
      over-clipping (too aggressive clipping slows convergence)

    Example:
        >>> zclip = ZClip(z_threshold=3.0, ema_decay=0.99)
        >>> for batch in dataloader:
        ...     loss.backward()
        ...     stats = zclip.clip(model)
        ...     wandb.log({
        ...         "grad_norm_raw": stats.raw_norm,
        ...         "grad_norm_clipped": stats.clipped_norm,
        ...     })
        ...     optimizer.step()

    Args:
        z_threshold: Number of standard deviations above mean to trigger clipping.
            Default is 3.0 (clips ~0.3% of updates if normally distributed).
        ema_decay: Decay factor for EMA statistics. Higher = more stable estimates
            but slower adaptation. Default is 0.99.
        min_clip_value: Minimum allowed clipping value to prevent degenerate cases.
            Default is 0.01.

    Attributes:
        z_threshold: The z-score threshold for clipping
        ema_decay: The EMA decay factor
        ema_mean: Current EMA mean of gradient norms (None before first step)
        ema_var: Current EMA variance of gradient norms (None before first step)
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        ema_decay: float = 0.99,
        min_clip_value: float = 0.01,
    ):
        self.z_threshold = z_threshold
        self.ema_decay = ema_decay
        self.min_clip_value = min_clip_value
        self.ema_mean: float | None = None
        self.ema_var: float | None = None
        self._step_count: int = 0

    def clip(
        self,
        model_or_params: nn.Module | Iterator[nn.Parameter],
        norm_type: float = 2.0,
    ) -> ZClipStats:
        """Clip gradients and return statistics.

        This method computes the gradient norm, updates EMA statistics,
        and applies clipping if the z-score exceeds the threshold.

        Args:
            model_or_params: Either a PyTorch model or an iterator of parameters.
                If a model, uses model.parameters() to get all parameters.
            norm_type: The type of norm to compute (default: L2 norm).

        Returns:
            ZClipStats with raw_norm, clipped_norm, z_score, and other stats.
        """
        # Get parameters with gradients
        if isinstance(model_or_params, nn.Module):
            params = [p for p in model_or_params.parameters() if p.grad is not None]
        else:
            params = [p for p in model_or_params if p.grad is not None]

        if not params:
            return ZClipStats(
                raw_norm=0.0,
                clipped_norm=0.0,
                z_score=0.0,
                was_clipped=False,
                ema_mean=0.0,
                ema_std=0.0,
            )

        # Compute gradient norm (without clipping yet)
        # We use torch.no_grad() for efficiency
        with torch.no_grad():
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad, norm_type) for p in params]),
                norm_type,
            )
        raw_norm_val = total_norm.item()

        self._step_count += 1

        # Initialize EMA on first step
        if self.ema_mean is None:
            self.ema_mean = raw_norm_val
            self.ema_var = 0.0
            return ZClipStats(
                raw_norm=raw_norm_val,
                clipped_norm=raw_norm_val,
                z_score=-1.0,  # No z-score on first step
                was_clipped=False,
                ema_mean=self.ema_mean,
                ema_std=0.0,
            )

        # Update EMA statistics
        self.ema_mean = self.ema_decay * self.ema_mean + (1 - self.ema_decay) * raw_norm_val
        self.ema_var = (
            self.ema_decay * self.ema_var
            + (1 - self.ema_decay) * (raw_norm_val - self.ema_mean) ** 2
        )

        # Compute z-score
        std = (self.ema_var + 1e-8) ** 0.5
        z_score = (raw_norm_val - self.ema_mean) / std

        # Check if clipping is needed
        if z_score > self.z_threshold:
            # Clip to threshold
            clip_val = max(self.ema_mean + self.z_threshold * std, self.min_clip_value)
            torch.nn.utils.clip_grad_norm_(params, clip_val)
            return ZClipStats(
                raw_norm=raw_norm_val,
                clipped_norm=clip_val,
                z_score=z_score,
                was_clipped=True,
                ema_mean=self.ema_mean,
                ema_std=std,
            )

        return ZClipStats(
            raw_norm=raw_norm_val,
            clipped_norm=raw_norm_val,
            z_score=z_score,
            was_clipped=False,
            ema_mean=self.ema_mean,
            ema_std=std,
        )

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "z_threshold": self.z_threshold,
            "ema_decay": self.ema_decay,
            "min_clip_value": self.min_clip_value,
            "ema_mean": self.ema_mean,
            "ema_var": self.ema_var,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.z_threshold = state_dict.get("z_threshold", self.z_threshold)
        self.ema_decay = state_dict.get("ema_decay", self.ema_decay)
        self.min_clip_value = state_dict.get("min_clip_value", self.min_clip_value)
        self.ema_mean = state_dict.get("ema_mean")
        self.ema_var = state_dict.get("ema_var")
        self._step_count = state_dict.get("step_count", 0)


def clip_grad_with_zclip(
    model: nn.Module,
    zclip: ZClip | None,
    fallback_max_norm: float = 1.0,
) -> tuple[float, float, bool]:
    """Convenience function to clip gradients with ZClip or fallback to fixed clipping.

    Args:
        model: The model with computed gradients
        zclip: ZClip instance, or None to use fixed clipping
        fallback_max_norm: Max norm for fixed clipping when zclip is None

    Returns:
        Tuple of (raw_norm, clipped_norm, was_clipped)
    """
    if zclip is not None:
        stats = zclip.clip(model)
        return stats.raw_norm, stats.clipped_norm, stats.was_clipped
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), fallback_max_norm)
        raw_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
        was_clipped = raw_norm > fallback_max_norm
        return raw_norm, min(raw_norm, fallback_max_norm), was_clipped
