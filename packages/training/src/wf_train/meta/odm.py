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

import torch
import torch.nn.functional as F

from wf_train.meta.config import ODMConfig

logger = logging.getLogger(__name__)


class OnlineDataMixer:
    """EXP3-based online data mixing for dataset weight optimization.

    Each dataset is an "arm" in a multi-armed bandit. We use training
    loss per domain as the reward signal to learn optimal sampling
    probabilities dynamically during training.

    The intuition is that higher loss = more to learn from that domain,
    so we should sample it more often. This naturally balances learning
    across domains without manual tuning.

    The EXP3 algorithm (Exponential-weight algorithm for Exploration and
    Exploitation):
    1. Maintains estimated rewards for each arm (dataset domain)
    2. Samples according to softmax over rewards mixed with uniform exploration
    3. Updates rewards using importance-weighted estimator for unbiased estimation

    The sampling distribution balances exploitation (sample high-reward domains)
    with exploration (sample all domains to gather information):

        π_t(i) = (1 - K·ε_t) × softmax(ε·R̂) + ε_t

    where ε_t decays over time: ε_t = min{1/K, √(ln K / (K·t))}

    Reference:
        Section 3.1 of https://arxiv.org/abs/2312.02406

    Example:
        >>> mixer = OnlineDataMixer(["web", "code", "math"], config)
        >>> # In training loop:
        >>> weights = mixer.get_sampling_weights()  # Use for dataloader
        >>> # ... train on batch from domain ...
        >>> mixer.update({"web": 2.5, "code": 1.8, "math": 3.2})  # Report losses
    """

    def __init__(
        self,
        dataset_names: list[str],
        config: ODMConfig,
    ) -> None:
        """Initialize the online data mixer.

        Args:
            dataset_names: List of dataset/domain names. These should match
                the domain names reported in per-domain losses.
            config: ODM configuration with hyperparameters.
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

        The exploration rate controls the balance between exploitation
        (sampling high-reward domains) and exploration (sampling all
        domains to gather information). It decays over time as we become
        more confident in our reward estimates.

        From the paper (Section 3.1):
            ε_t = min{1/K, √(ln K / (K·t))}

        This ensures:
        - Early training: High exploration (ε ≈ 1/K for all domains)
        - Late training: Low exploration, mostly exploitation

        Returns:
            Current exploration rate in range (0, 1/K].
        """
        if self.step_count == 0:
            return 1.0 / self.K
        return min(
            1.0 / self.K,
            math.sqrt(math.log(self.K) / (self.K * self.step_count))
        )

    def get_sampling_weights(self) -> dict[str, float]:
        """Compute current sampling distribution.

        Computes the EXP3 sampling distribution that mixes exploitation
        (softmax over rewards) with uniform exploration:

            π_t(i) = (1 - K·ε_t) × softmax(ε_{t-1}·R̂) + ε_t

        The previous exploration rate is used as a temperature for the
        softmax, making exploitation sharper as training progresses.

        After computing raw probabilities, min/max constraints are applied
        via iterative projection to ensure no domain is ignored or dominant.

        Returns:
            Dict mapping dataset names to sampling probabilities.
            Probabilities sum to 1 and satisfy min/max weight constraints.
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

        Uses alternating projection to satisfy both box constraints
        [min_weight, max_weight] and simplex constraint (sum to 1).

        The algorithm:
        1. Clamp probabilities to [min_weight, max_weight]
        2. Renormalize to sum to 1
        3. Repeat until both constraints are satisfied

        This typically converges in 2-3 iterations for reasonable
        min/max values. Edge cases (e.g., min_weight * K > 1) may
        not converge, but 10 iterations is a safe upper bound.

        Args:
            probs: Raw probabilities tensor of shape (K,).

        Returns:
            Constrained probabilities that sum to 1 and satisfy
            min_weight <= p_i <= max_weight for all i.
        """
        min_w = self.config.min_weight
        max_w = self.config.max_weight

        # Iterative projection to satisfy both min/max AND sum-to-1
        for _ in range(10):  # Usually converges in 2-3 iterations
            # Project onto box constraint
            probs = probs.clamp(min_w, max_w)
            # Project onto simplex constraint
            probs = probs / probs.sum()
            # Check if both constraints satisfied
            if probs.min() >= min_w - 1e-6 and probs.max() <= max_w + 1e-6:
                break

        return probs

    def update(self, losses_per_domain: dict[str, float]) -> None:
        """Update reward estimates based on observed domain losses.

        Uses importance-weighted rewards for unbiased estimation:
            R̂_i = loss_i / π_i

        where π_i is the current sampling probability. This corrects for
        the fact that we observe high-probability domains more often.

        The reward signal is: higher loss = more to learn = higher reward.
        This encourages sampling domains where the model is struggling.

        Updates are smoothed via exponential moving average:
            R̂_new = α·R̂_old + (1-α)·R̂_observed

        Args:
            losses_per_domain: Dict mapping dataset names to their losses.
                Only observed domains need to be included; missing domains
                are silently skipped.
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

    def state_dict(self) -> dict[str, object]:
        """Get state dict for checkpointing.

        Returns:
            Dict containing reward estimates and step count.
            Can be saved with torch.save() and restored with load_state_dict().
        """
        return {
            "avg_rewards": self.avg_rewards.copy(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Load state from checkpoint.

        Args:
            state: State dict from a previous state_dict() call.
                Missing keys are silently ignored for backwards compatibility.
        """
        if "avg_rewards" in state:
            self.avg_rewards = dict(state["avg_rewards"])  # type: ignore[arg-type]
        if "step_count" in state:
            self.step_count = int(state["step_count"])  # type: ignore[arg-type]
        logger.info(
            f"OnlineDataMixer state restored: step={self.step_count}, "
            f"rewards={self.avg_rewards}"
        )
