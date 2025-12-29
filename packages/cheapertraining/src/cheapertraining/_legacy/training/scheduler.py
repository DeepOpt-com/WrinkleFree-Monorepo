"""Learning rate scheduler implementations.

Reference: MobileLLM-R1 paper (arXiv:2509.24945) uses:
- Pretraining: Linear warmup + linear decay to 10% of peak
- Mid-training: Linear decay from max to 0
- Post-training: Linear warmup + linear decay to 0
"""

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


class LinearWarmupLinearDecay(_LRScheduler):
    """Linear warmup followed by linear decay.

    Used for pretraining phases.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Final LR as ratio of initial LR (e.g., 0.1 = 10%)
            last_epoch: Last epoch for resumption
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Linear decay to min_lr_ratio
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 - progress)
            scale = max(scale, self.min_lr_ratio)

        return [base_lr * scale for base_lr in self.base_lrs]


class LinearWarmupLinearDecayToZero(_LRScheduler):
    """Linear warmup followed by linear decay to zero.

    Used for mid-training and post-training phases.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            last_epoch: Last epoch for resumption
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Linear decay to 0
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 1 - progress
            scale = max(scale, 0)

        return [base_lr * scale for base_lr in self.base_lrs]


class CosineWarmup(_LRScheduler):
    """Linear warmup followed by cosine decay.

    Alternative scheduler for smoother decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum LR as ratio of initial LR
            last_epoch: Last epoch for resumption
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        return [base_lr * scale for base_lr in self.base_lrs]


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    total_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> _LRScheduler:
    """Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler (linear_decay, linear_decay_to_zero, cosine)
        warmup_steps: Number of warmup steps (takes precedence over warmup_ratio)
        warmup_ratio: Warmup as ratio of total steps (used if warmup_steps=0)
        total_steps: Total training steps
        min_lr_ratio: Minimum LR ratio for non-zero-end schedulers

    Returns:
        Configured scheduler
    """
    # Calculate warmup steps from ratio if not provided directly
    if warmup_steps == 0 and warmup_ratio > 0:
        warmup_steps = int(total_steps * warmup_ratio)

    scheduler_type = scheduler_type.lower().replace("-", "_")

    if scheduler_type == "linear_decay":
        return LinearWarmupLinearDecay(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type in ("linear_decay_to_zero", "linear"):
        return LinearWarmupLinearDecayToZero(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    elif scheduler_type == "cosine":
        return CosineWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
