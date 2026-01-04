"""LDC-MTL: Loss Discrepancy Control for Multi-Task Learning.

Reference: https://arxiv.org/abs/2502.08585

Optimizes objective weights (CE, DLM, distillation) using a small router
network with penalized single-level updates. Achieves O(1) time and memory
complexity compared to O(K) for traditional bi-level methods.

Key insight from paper: One gradient term in bi-level optimization is
empirically ~100x smaller than the other, allowing safe simplification
to a penalized single-level problem.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from wrinklefree.meta.config import LDCMTLConfig

logger = logging.getLogger(__name__)


class ObjectiveRouter(nn.Module):
    """Small MLP that learns task weights from current losses.

    The router takes the current loss values for each objective and
    outputs softmax-normalized weights. This is trained jointly with
    the main model using the LDC-MTL penalized objective.
    """

    def __init__(self, num_objectives: int, hidden_dim: int = 32):
        """Initialize the router network.

        Args:
            num_objectives: Number of training objectives (e.g., CE, DLM)
            hidden_dim: Hidden layer dimension for the MLP
        """
        super().__init__()
        self.num_objectives = num_objectives
        self.mlp = nn.Sequential(
            nn.Linear(num_objectives, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objectives),
        )

    def forward(self, losses: Tensor) -> Tensor:
        """Compute objective weights from current losses.

        Args:
            losses: Tensor of shape (num_objectives,) with current loss values

        Returns:
            Softmax-normalized weights of shape (num_objectives,)
        """
        logits = self.mlp(losses)
        return F.softmax(logits, dim=-1)


def compute_loss_discrepancy(losses: Tensor, weights: Tensor) -> Tensor:
    """Compute the LDC-MTL upper-level objective (loss discrepancy penalty).

    This penalizes imbalanced weighted losses across objectives, encouraging
    the router to find weights that lead to similar scaled loss values.

    From the paper:
        f(W,x) = Σ_{i=1}^{K-1} |τ_i·l_i(x) - τ_{i+1}·l_{i+1}(x)|

    where τ_i are the weights and l_i are the losses, sorted by weighted value.

    Args:
        losses: Tensor of shape (K,) with current loss values
        weights: Tensor of shape (K,) with current weights

    Returns:
        Scalar tensor with the discrepancy penalty
    """
    weighted = weights * losses
    sorted_weighted, _ = torch.sort(weighted)
    # Sum of absolute differences between consecutive sorted weighted losses
    return (sorted_weighted[1:] - sorted_weighted[:-1]).abs().sum()


class LDCMTLManager:
    """Manages objective weight optimization via LDC-MTL.

    This class wraps the ObjectiveRouter and provides methods to:
    1. Compute weighted loss with the discrepancy penalty
    2. Update the router parameters
    3. Get current weights for logging
    4. Save/load state for checkpointing
    """

    def __init__(
        self,
        objective_names: list[str],
        config: LDCMTLConfig,
        device: torch.device,
    ):
        """Initialize the LDC-MTL manager.

        Args:
            objective_names: List of objective names (e.g., ["ce", "dlm"])
            config: LDC-MTL configuration
            device: Device to place the router on
        """
        self.objective_names = objective_names
        self.config = config
        self.device = device

        self.router = ObjectiveRouter(
            len(objective_names),
            config.hidden_dim,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.router.parameters(),
            lr=config.router_lr,
        )

        # Track current weights for logging
        self._current_weights: dict[str, float] = {
            name: 1.0 / len(objective_names) for name in objective_names
        }

        logger.info(
            f"LDCMTLManager initialized with {len(objective_names)} objectives: "
            f"{objective_names}, lambda={config.lambda_penalty}"
        )

    def compute_weighted_loss(
        self,
        losses: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute weighted loss with LDC-MTL discrepancy penalty.

        The total loss is:
            L_total = Σ_i w_i·l_i + λ·discrepancy(w, l)

        Gradients flow through the router, enabling joint optimization.

        Args:
            losses: Dict mapping objective names to their loss tensors

        Returns:
            Tuple of (total_loss, weight_dict) where weight_dict maps
            objective names to their current weights
        """
        # Stack losses in consistent order
        loss_vec = torch.stack([losses[n] for n in self.objective_names])

        # Get weights from router (detach losses to avoid double backprop)
        weights = self.router(loss_vec.detach())

        # Compute weighted sum of losses
        weighted_loss = (weights * loss_vec).sum()

        # Compute discrepancy penalty
        discrepancy = compute_loss_discrepancy(loss_vec, weights)

        # Total loss with penalty
        total = weighted_loss + self.config.lambda_penalty * discrepancy

        # Update tracked weights for logging
        self._current_weights = {
            name: weights[i].item()
            for i, name in enumerate(self.objective_names)
        }

        return total, self._current_weights.copy()

    def step(self) -> None:
        """Update router parameters after backward pass."""
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_weights(self) -> dict[str, float]:
        """Get current objective weights.

        Returns:
            Dict mapping objective names to their current weights
        """
        return self._current_weights.copy()

    def get_wandb_metrics(self, prefix: str = "meta/ldc_mtl") -> dict[str, float]:
        """Get metrics for WandB logging.

        Args:
            prefix: Metric name prefix

        Returns:
            Dict of metric_name -> value
        """
        metrics = {}
        for name, weight in self._current_weights.items():
            metrics[f"{prefix}/objective_weight_{name}"] = weight
        return metrics

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            "router_state": self.router.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "current_weights": self._current_weights,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        if "router_state" in state:
            self.router.load_state_dict(state["router_state"])
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if "current_weights" in state:
            self._current_weights = state["current_weights"]
        logger.info("LDCMTLManager state restored from checkpoint")
