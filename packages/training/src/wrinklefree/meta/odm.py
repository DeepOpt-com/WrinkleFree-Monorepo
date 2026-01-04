"""ODM: Online Data Mixing via EXP3 multi-armed bandit.

Reference: https://arxiv.org/abs/2312.02406
    "Efficient Online Data Mixing For Language Model Pre-Training"

Optimizes dataset sampling probabilities using training loss as reward signal.
Adds ~0.000007% wall-clock overhead (virtually free). Published results show
19% fewer iterations to reach same perplexity.

Key idea: Each data domain is an "arm" in a multi-armed bandit. We use training
loss as a reward signal - higher loss means more to learn from that domain.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

from wrinklefree.meta.config import ODMConfig

logger = logging.getLogger(__name__)


class OnlineDataMixer:
    """EXP3-based online data mixing for dataset weight optimization.

    Each dataset is an "arm" in a multi-armed bandit. We use training
    loss per domain as the reward signal to learn optimal sampling
    probabilities dynamically during training.

    The EXP3 algorithm:
    1. Maintains estimated rewards for each arm (dataset)
    2. Samples according to softmax over rewards + exploration
    3. Updates rewards using importance-weighted estimator
    """

    def __init__(
        self,
        dataset_names: list[str],
        config: ODMConfig,
    ):
        """Initialize the online data mixer.

        Args:
            dataset_names: List of dataset/domain names
            config: ODM configuration
        """
        self.dataset_names = dataset_names
        self.config = config
        self.K = len(dataset_names)

        # EXP3 state: cumulative importance-weighted rewards
        self.avg_rewards: dict[str, float] = {name: 0.0 for name in dataset_names}

        # Step counter for exploration rate decay
        self.step_count = 0

        # Track warmup for logging
        self._in_warmup = True

        logger.info(
            f"OnlineDataMixer initialized with {self.K} datasets: {dataset_names}"
        )

    def get_exploration_rate(self) -> float:
        """Compute decaying exploration rate.

        From the paper:
            ε_t = min{1/K, √(ln K / (K·t))}

        Returns:
            Current exploration rate
        """
        if self.step_count == 0:
            return 1.0 / self.K
        return min(
            1.0 / self.K,
            math.sqrt(math.log(self.K) / (self.K * self.step_count))
        )

    def get_sampling_weights(self) -> dict[str, float]:
        """Compute current sampling distribution.

        From the paper:
            π_t(i) = (1 - K·ε_t) × softmax(ε_{t-1}·R̂) + ε_t

        This mixes the exploitation distribution (softmax over rewards)
        with uniform exploration.

        Returns:
            Dict mapping dataset names to sampling probabilities
        """
        eps = self.get_exploration_rate()

        # Softmax over rewards (scaled by exploration rate for temperature)
        rewards = torch.tensor(
            [self.avg_rewards[n] for n in self.dataset_names],
            dtype=torch.float32,
        )

        # Use previous epsilon as temperature for softmax
        # This makes exploitation sharper as training progresses
        if self.step_count > 0:
            prev_eps = min(
                1.0 / self.K,
                math.sqrt(math.log(self.K) / (self.K * (self.step_count - 1)))
                if self.step_count > 1 else 1.0 / self.K
            )
        else:
            prev_eps = 1.0 / self.K

        probs = F.softmax(prev_eps * rewards, dim=0)

        # Mix with uniform exploration
        probs = (1 - self.K * eps) * probs + eps

        # Apply min/max constraints
        probs = self._apply_constraints(probs)

        return {n: p.item() for n, p in zip(self.dataset_names, probs)}

    def _apply_constraints(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply min/max weight constraints via iterative projection.

        Clips probabilities to [min_weight, max_weight] and renormalizes.
        Uses iterative projection to ensure all constraints are satisfied.

        Args:
            probs: Raw probabilities tensor

        Returns:
            Constrained and renormalized probabilities (sum to 1)
        """
        min_w = self.config.min_weight
        max_w = self.config.max_weight

        # Iterative projection to satisfy both min/max AND sum-to-1
        # This handles edge cases where simple clamp+renormalize fails
        for _ in range(10):  # Usually converges in 2-3 iterations
            # Clamp to range
            probs = probs.clamp(min_w, max_w)
            # Renormalize
            probs = probs / probs.sum()
            # Check if constraints satisfied
            if probs.min() >= min_w - 1e-6 and probs.max() <= max_w + 1e-6:
                break

        return probs

    def update(self, losses_per_domain: dict[str, float]) -> None:
        """Update rewards based on observed domain losses.

        Reward = loss (higher loss = more to learn = higher priority)
        Uses importance-weighted update for unbiased estimation.

        Args:
            losses_per_domain: Dict mapping dataset names to their losses
        """
        self.step_count += 1

        # Get current weights for importance weighting
        weights = self.get_sampling_weights()

        for name, loss in losses_per_domain.items():
            if name not in self.avg_rewards:
                logger.warning(f"Unknown domain in ODM update: {name}")
                continue

            # Importance-weighted reward (unbiased estimator)
            # We divide by sampling probability to correct for sampling bias
            iw_reward = loss / max(weights[name], 1e-8)

            # Exponential moving average for stability
            alpha = self.config.reward_smoothing
            self.avg_rewards[name] = (
                alpha * self.avg_rewards[name] +
                (1 - alpha) * iw_reward
            )

    def is_in_warmup(self, current_step: int, total_steps: int) -> bool:
        """Check if we're in the warmup period.

        During warmup, we use uniform weights instead of learned weights.

        Args:
            current_step: Current training step
            total_steps: Total training steps

        Returns:
            True if in warmup period
        """
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self._in_warmup = current_step < warmup_steps
        return self._in_warmup

    def get_uniform_weights(self) -> dict[str, float]:
        """Get uniform weights (for warmup period).

        Returns:
            Dict with equal weight for each dataset
        """
        weight = 1.0 / self.K
        return {name: weight for name in self.dataset_names}

    def get_wandb_metrics(self, prefix: str = "meta/odm") -> dict[str, float]:
        """Get metrics for WandB logging.

        Args:
            prefix: Metric name prefix

        Returns:
            Dict of metric_name -> value
        """
        metrics = {}

        # Current weights
        weights = self.get_sampling_weights()
        for name, weight in weights.items():
            metrics[f"{prefix}/dataset_weight_{name}"] = weight

        # Exploration rate
        metrics[f"{prefix}/exploration_rate"] = self.get_exploration_rate()

        # Average rewards (for debugging)
        for name, reward in self.avg_rewards.items():
            metrics[f"{prefix}/avg_reward_{name}"] = reward

        return metrics

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            "avg_rewards": self.avg_rewards.copy(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        if "avg_rewards" in state:
            self.avg_rewards = state["avg_rewards"].copy()
        if "step_count" in state:
            self.step_count = state["step_count"]
        logger.info(
            f"OnlineDataMixer state restored: step={self.step_count}, "
            f"rewards={self.avg_rewards}"
        )
