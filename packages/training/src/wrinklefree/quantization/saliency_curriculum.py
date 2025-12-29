"""Saliency Smoothing Curriculum for BitNet mixed-precision training.

This module implements a curriculum that gradually transitions from mixed-precision
(FP16 for salient columns) to fully ternary quantization during Stage 1.9 training.

Based on HBLLM's approach:
- L-infinity norm saliency detection per column
- EMA tracking for stable saliency estimates
- Gradual annealing of protection ratio to 0%
"""

import math
from typing import Optional

import torch


class SaliencyCurriculum:
    """
    Manages saliency-based mixed-precision curriculum for BitLinear layers.

    Tracks per-layer, per-column saliency using L-infinity norm (max abs value per column).
    Uses EMA to smooth saliency estimates over training steps.
    Anneals the top-k% threshold from initial value down to 0%.

    Args:
        initial_saliency_k: Start protecting top k% of columns (default: 0.1 = 10%)
        final_saliency_k: End with this protection ratio (default: 0.0 = fully ternary)
        ema_decay: EMA decay for saliency tracking (default: 0.99)
        schedule_type: Annealing schedule - "linear" or "cosine" (default: "cosine")
        warmup_steps: Keep initial_k constant for this many steps (default: 0)
        update_interval: Update saliency EMA every N steps to reduce overhead (default: 10)
    """

    def __init__(
        self,
        initial_saliency_k: float = 0.1,
        final_saliency_k: float = 0.0,
        ema_decay: float = 0.99,
        schedule_type: str = "cosine",
        warmup_steps: int = 0,
        update_interval: int = 10,
    ):
        if schedule_type not in ("linear", "cosine"):
            raise ValueError(f"schedule_type must be 'linear' or 'cosine', got {schedule_type}")
        if not 0.0 <= initial_saliency_k <= 1.0:
            raise ValueError(f"initial_saliency_k must be in [0, 1], got {initial_saliency_k}")
        if not 0.0 <= final_saliency_k <= 1.0:
            raise ValueError(f"final_saliency_k must be in [0, 1], got {final_saliency_k}")

        self.initial_k = initial_saliency_k
        self.final_k = final_saliency_k
        self.ema_decay = ema_decay
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval

        # Per-layer saliency tracking: layer_name -> EMA of L-inf per column
        self._saliency_ema: dict[str, torch.Tensor] = {}

        # Step tracking
        self._current_step = 0
        self._total_steps = 0

    def register_layer(self, name: str, weight: torch.Tensor) -> None:
        """
        Register a BitLinear layer for saliency tracking.

        Args:
            name: Unique identifier for this layer
            weight: Weight tensor of shape (out_features, in_features)
        """
        # weight shape: (out_features, in_features)
        # Track saliency per column (input feature dimension)
        num_columns = weight.shape[1]  # in_features
        self._saliency_ema[name] = torch.zeros(num_columns, device=weight.device, dtype=weight.dtype)

    def update_saliency(self, name: str, weight: torch.Tensor) -> None:
        """
        Update EMA saliency estimate for a layer.

        Only updates every `update_interval` steps to reduce computational overhead.

        Args:
            name: Layer identifier
            weight: Weight tensor of shape (out_features, in_features)
        """
        # Skip update if not at update interval
        if self._current_step % self.update_interval != 0:
            return

        # L-inf norm per column: max abs value across rows
        # weight shape: (out_features, in_features)
        # Memory-efficient: compute max and min separately to avoid creating full .abs() tensor
        with torch.no_grad():
            max_vals = weight.max(dim=0).values  # (in_features,)
            min_vals = weight.min(dim=0).values  # (in_features,)
            l_inf_per_col = torch.maximum(max_vals.abs(), min_vals.abs())  # (in_features,)

            if name not in self._saliency_ema:
                # First time seeing this layer - initialize with current values
                self._saliency_ema[name] = l_inf_per_col.clone()
            else:
                # EMA update
                ema = self._saliency_ema[name]
                # Ensure same device/dtype
                if ema.device != l_inf_per_col.device:
                    ema = ema.to(l_inf_per_col.device)
                if ema.dtype != l_inf_per_col.dtype:
                    ema = ema.to(l_inf_per_col.dtype)
                self._saliency_ema[name] = self.ema_decay * ema + (1 - self.ema_decay) * l_inf_per_col

    def get_saliency_mask(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        """
        Get boolean mask indicating which columns are salient (should NOT be quantized).

        Args:
            name: Layer identifier
            weight: Weight tensor of shape (out_features, in_features)

        Returns:
            mask: Shape (1, in_features), True = salient (keep FP16), False = quantize
        """
        num_columns = weight.shape[1]

        if name not in self._saliency_ema:
            # No tracking yet, quantize everything
            return torch.zeros(1, num_columns, dtype=torch.bool, device=weight.device)

        current_k = self._get_current_k()
        if current_k <= 0:
            # Fully ternary, no protection
            return torch.zeros(1, num_columns, dtype=torch.bool, device=weight.device)

        saliency = self._saliency_ema[name]
        if saliency.device != weight.device:
            saliency = saliency.to(weight.device)

        num_salient = max(1, int(current_k * num_columns))

        # Get threshold for top-k
        # topk returns (values, indices), we need the smallest value in top-k
        threshold = torch.topk(saliency, min(num_salient, num_columns)).values[-1]
        mask = (saliency >= threshold).unsqueeze(0)  # (1, in_features)
        return mask

    def _get_current_k(self) -> float:
        """
        Get current saliency threshold based on schedule.

        Returns:
            Current fraction of columns to protect (between initial_k and final_k)
        """
        if self._current_step < self.warmup_steps:
            return self.initial_k

        if self._total_steps <= self.warmup_steps:
            return self.final_k

        effective_step = self._current_step - self.warmup_steps
        effective_total = self._total_steps - self.warmup_steps

        if effective_total <= 0:
            return self.final_k

        progress = min(1.0, effective_step / effective_total)

        if self.schedule_type == "linear":
            return self.initial_k + (self.final_k - self.initial_k) * progress
        else:  # cosine
            # Cosine annealing: slow at start/end, fast in middle
            return self.final_k + (self.initial_k - self.final_k) * 0.5 * (1 + math.cos(math.pi * progress))

    def step(self) -> None:
        """Advance curriculum by one training step."""
        self._current_step += 1

    def set_total_steps(self, total_steps: int) -> None:
        """
        Set total training steps for schedule calculation.

        Args:
            total_steps: Total number of training steps
        """
        self._total_steps = total_steps

    def get_current_k(self) -> float:
        """Public accessor for current saliency protection ratio."""
        return self._get_current_k()

    def state_dict(self) -> dict:
        """
        Return state for checkpointing.

        Returns:
            Dictionary containing curriculum state
        """
        return {
            "saliency_ema": {k: v.cpu() for k, v in self._saliency_ema.items()},
            "current_step": self._current_step,
            "total_steps": self._total_steps,
            "initial_k": self.initial_k,
            "final_k": self.final_k,
            "ema_decay": self.ema_decay,
            "schedule_type": self.schedule_type,
            "warmup_steps": self.warmup_steps,
            "update_interval": self.update_interval,
        }

    def load_state_dict(self, state: dict, device: Optional[torch.device] = None) -> None:
        """
        Load state from checkpoint.

        Args:
            state: State dictionary from state_dict()
            device: Device to load tensors to (optional)
        """
        self._current_step = state["current_step"]
        self._total_steps = state["total_steps"]

        # Restore EMA tensors
        self._saliency_ema = {}
        for k, v in state["saliency_ema"].items():
            if device is not None:
                self._saliency_ema[k] = v.to(device)
            else:
                self._saliency_ema[k] = v

    def sync_across_ranks(self) -> None:
        """
        Synchronize saliency EMA across distributed ranks.

        Called before get_saliency_mask to ensure consistent masks across GPUs.
        Uses all-reduce with mean aggregation.
        """
        if not torch.distributed.is_initialized():
            return

        for name, ema in self._saliency_ema.items():
            torch.distributed.all_reduce(ema, op=torch.distributed.ReduceOp.AVG)
