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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from wf_train.meta.config import LDCMTLConfig

logger = logging.getLogger(__name__)


class ObjectiveRouter(nn.Module):
    """Small MLP that learns task weights from current losses.

    The router takes the current loss values for each objective and
    outputs softmax-normalized weights. This is trained jointly with
    the main model using the LDC-MTL penalized objective.

    Architecture:
        Input (K losses) -> Linear(K, hidden) -> ReLU -> Linear(hidden, K) -> Softmax

    The network is intentionally small (default 32 hidden units) because:
    1. Input dimension is just K (number of objectives, typically 2-4)
    2. The task is to learn a simple loss-to-weight mapping
    3. Smaller network = faster, less memory, less overfitting risk
    """

    def __init__(self, num_objectives: int, hidden_dim: int = 32) -> None:
        """Initialize the router network.

        Args:
            num_objectives: Number of training objectives (e.g., 2 for CE + DLM)
            hidden_dim: Hidden layer dimension for the MLP. Default 32 is
                sufficient for typical multi-task setups.
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
            losses: Tensor of shape (num_objectives,) with current loss values.
                These should be detached from the computation graph to avoid
                double backpropagation through the loss computation.

        Returns:
            Softmax-normalized weights of shape (num_objectives,) that sum to 1.
        """
        logits = self.mlp(losses)
        return F.softmax(logits, dim=-1)


def compute_loss_discrepancy(losses: Tensor, weights: Tensor) -> Tensor:
    """Compute the LDC-MTL loss discrepancy penalty.

    This penalizes imbalanced weighted losses across objectives, encouraging
    the router to find weights that lead to similar scaled loss values.
    The intuition is that if one task has much higher weighted loss than
    others, it's dominating the gradient, which may hurt other tasks.

    From the paper (Section 3.2, Equation 6):
        f(W,x) = Σ_{i=1}^{K-1} |τ_i·l_i(x) - τ_{i+1}·l_{i+1}(x)|

    where τ_i are the weights and l_i are the losses, sorted by weighted value.

    The sorting ensures we're measuring the "spread" of weighted losses,
    not their absolute differences in some fixed order.

    Args:
        losses: Tensor of shape (K,) with current loss values for each objective.
        weights: Tensor of shape (K,) with current weights from the router.

    Returns:
        Scalar tensor with the discrepancy penalty. Lower values indicate
        more balanced weighted losses across objectives.

    Example:
        >>> losses = torch.tensor([2.0, 1.0, 0.5])  # 3 objectives
        >>> weights = torch.tensor([0.3, 0.3, 0.4])  # roughly uniform
        >>> discrepancy = compute_loss_discrepancy(losses, weights)
        >>> # weighted = [0.6, 0.3, 0.2], sorted = [0.2, 0.3, 0.6]
        >>> # discrepancy = |0.3-0.2| + |0.6-0.3| = 0.1 + 0.3 = 0.4
    """
    weighted = weights * losses
    sorted_weighted, _ = torch.sort(weighted)
    # Sum of absolute differences between consecutive sorted weighted losses
    return (sorted_weighted[1:] - sorted_weighted[:-1]).abs().sum()


class LDCMTLManager:
    """Manages objective weight optimization via LDC-MTL.

    This class wraps the ObjectiveRouter and provides a high-level interface for:
    1. Computing weighted loss with the discrepancy penalty
    2. Updating the router parameters via its own optimizer
    3. Getting current weights for logging/debugging
    4. Saving/loading state for checkpointing

    The training loop integration is simple:
        1. Call compute_weighted_loss() to get total loss and current weights
        2. Backpropagate through total loss (gradients flow to both model and router)
        3. Call step() after the main optimizer step to update router weights

    Example:
        >>> manager = LDCMTLManager(["ce", "dlm"], config, device)
        >>> # In training loop:
        >>> losses = {"ce": ce_loss, "dlm": dlm_loss}
        >>> total_loss, weights = manager.compute_weighted_loss(losses)
        >>> total_loss.backward()
        >>> main_optimizer.step()
        >>> manager.step()  # Update router
    """

    def __init__(
        self,
        objective_names: list[str],
        config: LDCMTLConfig,
        device: torch.device,
    ) -> None:
        """Initialize the LDC-MTL manager.

        Args:
            objective_names: List of objective names (e.g., ["ce", "dlm"]).
                Order matters - losses must be provided in the same order.
            config: LDC-MTL configuration with hyperparameters.
            device: Device to place the router network on. Should match
                the device where losses are computed.
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

        The total loss combines the weighted sum of individual losses with
        a discrepancy penalty that encourages balanced contributions:

            L_total = Σ_i w_i·l_i + λ·discrepancy(w, l)

        Gradients flow through:
        - Individual losses (to model parameters)
        - Router weights (to router parameters via the discrepancy term)

        Note: Losses are detached before passing to the router to avoid
        double backpropagation. The router learns from the discrepancy
        penalty, not from the losses directly.

        Args:
            losses: Dict mapping objective names to their scalar loss tensors.
                Must contain all objectives specified in objective_names.

        Returns:
            Tuple of (total_loss, weight_dict):
            - total_loss: Scalar tensor for backpropagation
            - weight_dict: Dict mapping objective names to their current
              weights (floats, for logging purposes)
        """
        # Stack losses in consistent order
        # Handle missing objectives (e.g., SFT has weight 0 during pretrain phases)
        # by using 0 loss for missing objectives
        loss_vec = torch.stack([
            losses.get(n, torch.tensor(0.0, device=next(iter(losses.values())).device))
            for n in self.objective_names
        ])

        # Move router to same device as losses if needed (handles lazy GPU init)
        loss_device = loss_vec.device
        if loss_device != self.device:
            self.router = self.router.to(loss_device)
            self.device = loss_device
            # Must recreate optimizer - old one has references to CPU params
            self.optimizer = torch.optim.Adam(
                self.router.parameters(),
                lr=self.config.router_lr,
            )
            logger.info(f"LDC-MTL router moved to {loss_device}, optimizer recreated")

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
        """Update router parameters after backward pass.

        Call this after the main optimizer step. The router has its own
        optimizer (Adam) with a separate learning rate.
        """
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

    def state_dict(self) -> dict[str, object]:
        """Get state dict for checkpointing.

        Returns:
            Dict containing router weights, optimizer state, and current
            weight values. Can be saved with torch.save() and restored
            with load_state_dict().
        """
        return {
            "router_state": self.router.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "current_weights": self._current_weights,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Load state from checkpoint.

        Args:
            state: State dict from a previous state_dict() call.
                Missing keys are silently ignored for backwards compatibility.
        """
        if "router_state" in state:
            self.router.load_state_dict(state["router_state"])  # type: ignore[arg-type]
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[arg-type]
        if "current_weights" in state:
            self._current_weights = state["current_weights"]  # type: ignore[assignment]
        logger.info("LDCMTLManager state restored from checkpoint")
