"""Meta-parameter manager for outer-loop optimization.

Manages differentiable meta-parameters including:
- Dataset mixture weights
- Objective weights
- Learning rate scales

References:
- ScaleBiO (2024): https://arxiv.org/abs/2406.19976
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from wrinklefree.meta.config import MetaOptimizationConfig

logger = logging.getLogger(__name__)


class MetaParameterManager:
    """Manages meta-parameters for outer-loop optimization.

    Meta-parameters are optimized to improve validation performance.
    Uses first-order approximations (no expensive Hessian computation).

    Supports three types of meta-parameters:
    1. Dataset mixture weights (softmax-normalized)
    2. Objective weights (clipped to range)
    3. Learning rate scales (clipped to range)

    Each is stored as raw logits/values and transformed for use.
    """

    def __init__(
        self,
        config: MetaOptimizationConfig,
        dataset_names: list[str],
        objective_names: list[str],
        optimizer_param_groups: Optional[list[str]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize meta-parameter manager.

        Args:
            config: Meta-optimization configuration
            dataset_names: Names of datasets in the mixture
            objective_names: Names of training objectives
            optimizer_param_groups: Names of optimizer parameter groups (e.g., ["muon", "adamw"])
            device: Device to store parameters on
        """
        self.config = config
        self.device = device
        self.dataset_names = dataset_names
        self.objective_names = objective_names
        self.optimizer_param_groups = optimizer_param_groups or []

        # Initialize meta-parameters as raw values (will be transformed for use)
        self._dataset_logits: Optional[torch.Tensor] = None
        self._objective_weights: Optional[torch.Tensor] = None
        self._lr_scales: Optional[torch.Tensor] = None

        # Momentum buffers for meta-updates
        self._dataset_velocity: Optional[torch.Tensor] = None
        self._objective_velocity: Optional[torch.Tensor] = None
        self._lr_velocity: Optional[torch.Tensor] = None

        # History for logging
        self._update_history: list[dict] = []

        # Initialize based on config
        if config.optimize_dataset_weights and dataset_names:
            self._init_dataset_weights()

        if config.optimize_objective_weights and objective_names:
            self._init_objective_weights()

        if config.optimize_learning_rates and optimizer_param_groups:
            self._init_lr_scales()

    def _init_dataset_weights(self) -> None:
        """Initialize dataset mixture weights to uniform."""
        n = len(self.dataset_names)
        # Start with uniform logits (softmax will give uniform weights)
        self._dataset_logits = torch.zeros(n, device=self.device)
        self._dataset_velocity = torch.zeros(n, device=self.device)
        logger.info(f"Initialized dataset weights for {n} datasets: {self.dataset_names}")

    def _init_objective_weights(self) -> None:
        """Initialize objective weights to 1.0."""
        n = len(self.objective_names)
        self._objective_weights = torch.ones(n, device=self.device)
        self._objective_velocity = torch.zeros(n, device=self.device)
        logger.info(f"Initialized objective weights for {n} objectives: {self.objective_names}")

    def _init_lr_scales(self) -> None:
        """Initialize LR scales to 1.0."""
        n = len(self.optimizer_param_groups)
        self._lr_scales = torch.ones(n, device=self.device)
        self._lr_velocity = torch.zeros(n, device=self.device)
        logger.info(f"Initialized LR scales for {n} param groups: {self.optimizer_param_groups}")

    def get_dataset_weights(self) -> dict[str, float]:
        """Get current dataset mixture weights (normalized via softmax).

        Constraints are applied via projection after update.
        """
        if self._dataset_logits is None:
            return {}

        # Softmax for normalization (sum to 1, all positive)
        weights = F.softmax(self._dataset_logits, dim=0)

        return {
            name: weights[i].item()
            for i, name in enumerate(self.dataset_names)
        }

    def get_objective_weights(self) -> dict[str, float]:
        """Get current objective weights."""
        if self._objective_weights is None:
            return {}

        return {
            name: self._objective_weights[i].item()
            for i, name in enumerate(self.objective_names)
        }

    def get_lr_scales(self) -> dict[str, float]:
        """Get current LR scaling factors."""
        if self._lr_scales is None:
            return {}

        return {
            name: self._lr_scales[i].item()
            for i, name in enumerate(self.optimizer_param_groups)
        }

    def update_from_gradients(
        self,
        dataset_grads: Optional[dict[str, float]] = None,
        objective_grads: Optional[dict[str, float]] = None,
        lr_grads: Optional[dict[str, float]] = None,
    ) -> None:
        """Apply gradient update to meta-parameters with momentum.

        Uses SGD with momentum:
            v = momentum * v + (1 - momentum) * grad
            param = param - lr * v

        Args:
            dataset_grads: Gradients for dataset weights
            objective_grads: Gradients for objective weights
            lr_grads: Gradients for LR scales
        """
        lr = self.config.meta_lr
        momentum = self.config.meta_momentum

        # Update dataset weights
        if dataset_grads is not None and self._dataset_logits is not None:
            grad = torch.tensor(
                [dataset_grads.get(name, 0.0) for name in self.dataset_names],
                device=self.device,
            )
            self._dataset_velocity = momentum * self._dataset_velocity + (1 - momentum) * grad
            self._dataset_logits = self._dataset_logits - lr * self._dataset_velocity

            # Project to constraints by clipping softmax output range
            # (done implicitly through clamped logits)
            self._apply_dataset_constraints()

        # Update objective weights
        if objective_grads is not None and self._objective_weights is not None:
            grad = torch.tensor(
                [objective_grads.get(name, 0.0) for name in self.objective_names],
                device=self.device,
            )
            self._objective_velocity = momentum * self._objective_velocity + (1 - momentum) * grad
            self._objective_weights = self._objective_weights - lr * self._objective_velocity

            # Clamp to constraint range
            min_w, max_w = self.config.constraints.objective_weight_range
            self._objective_weights = self._objective_weights.clamp(min_w, max_w)

        # Update LR scales
        if lr_grads is not None and self._lr_scales is not None:
            grad = torch.tensor(
                [lr_grads.get(name, 0.0) for name in self.optimizer_param_groups],
                device=self.device,
            )
            self._lr_velocity = momentum * self._lr_velocity + (1 - momentum) * grad
            self._lr_scales = self._lr_scales - lr * self._lr_velocity

            # Clamp to constraint range
            min_s, max_s = self.config.constraints.lr_scale_range
            self._lr_scales = self._lr_scales.clamp(min_s, max_s)

        # Record update
        self._update_history.append({
            "dataset_weights": self.get_dataset_weights(),
            "objective_weights": self.get_objective_weights(),
            "lr_scales": self.get_lr_scales(),
        })

    def _apply_dataset_constraints(self) -> None:
        """Apply min/max constraints to dataset weights via logit adjustment.

        Since weights = softmax(logits), we can't directly clamp weights.
        Instead, we iteratively adjust logits to push weights toward bounds.
        """
        min_w, max_w = self.config.constraints.dataset_weight_range
        weights = F.softmax(self._dataset_logits, dim=0)

        # Simple projection: clamp weights and renormalize
        # This is approximate but works well in practice
        clamped = weights.clamp(min_w, max_w)
        clamped = clamped / clamped.sum()  # Renormalize

        # Convert back to logits (inverse softmax is ambiguous, use log)
        # log(softmax(x)) = x - logsumexp(x)
        # We use log of clamped weights as new logits (up to constant)
        self._dataset_logits = torch.log(clamped + 1e-8)

    def apply_to_training(
        self,
        mixed_dataset=None,
        objective_manager=None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Apply current meta-parameters to training components.

        Args:
            mixed_dataset: MixedDataset with set_weights() method
            objective_manager: ObjectiveManager with set_weights() method
            optimizer: Optimizer with param_groups to scale LRs
        """
        # Apply dataset weights
        if mixed_dataset is not None and self._dataset_logits is not None:
            weights = self.get_dataset_weights()
            if hasattr(mixed_dataset, "set_weights"):
                mixed_dataset.set_weights(weights)
                logger.debug(f"Applied dataset weights: {weights}")
            elif hasattr(mixed_dataset, "update_weights"):
                mixed_dataset.update_weights(weights)
                logger.debug(f"Applied dataset weights via update_weights: {weights}")

        # Apply objective weights
        if objective_manager is not None and self._objective_weights is not None:
            weights = self.get_objective_weights()
            if hasattr(objective_manager, "set_weights"):
                objective_manager.set_weights(weights)
                logger.debug(f"Applied objective weights: {weights}")
            else:
                # Fallback: directly update weights dict
                for name, weight in weights.items():
                    if hasattr(objective_manager, "weights") and name in objective_manager.weights:
                        objective_manager.weights[name] = weight

        # Apply LR scales
        if optimizer is not None and self._lr_scales is not None:
            scales = self.get_lr_scales()

            # Handle wrapped optimizers
            actual_optimizer = optimizer
            if hasattr(optimizer, "_optimizer"):
                actual_optimizer = optimizer._optimizer

            # Scale LRs for each param group
            for i, (name, scale) in enumerate(scales.items()):
                if i < len(actual_optimizer.param_groups):
                    pg = actual_optimizer.param_groups[i]
                    # Store original LR on first application
                    if "_original_lr" not in pg:
                        pg["_original_lr"] = pg["lr"]
                    pg["lr"] = pg["_original_lr"] * scale
                    logger.debug(f"Applied LR scale to group {i} ({name}): {scale}")

    def get_update_history(self) -> list[dict]:
        """Get history of meta-parameter updates."""
        return self._update_history

    def get_wandb_metrics(self, prefix: str = "meta") -> dict[str, float]:
        """Get current meta-parameters as WandB-loggable metrics.

        Args:
            prefix: Metric prefix (default: "meta")

        Returns:
            Dict of metric_name -> value
        """
        metrics = {}

        for name, weight in self.get_dataset_weights().items():
            metrics[f"{prefix}/dataset_weight_{name}"] = weight

        for name, weight in self.get_objective_weights().items():
            metrics[f"{prefix}/objective_weight_{name}"] = weight

        for name, scale in self.get_lr_scales().items():
            metrics[f"{prefix}/lr_scale_{name}"] = scale

        return metrics

    def get_hyperparameter_metrics(self) -> dict[str, float]:
        """Get all hyperparameters for dedicated WandB tracking.

        Returns metrics under "hyperparameters/" prefix that track:
        - Dataset mixture weights
        - Task/objective weights
        - Learning rate scales (if optimized)
        - Current learning rates (if optimizer available)

        Returns:
            Dict of hyperparameter_name -> value
        """
        metrics = {}
        prefix = "hyperparameters"

        # Dataset mixture weights
        dataset_weights = self.get_dataset_weights()
        for name, weight in dataset_weights.items():
            metrics[f"{prefix}/data_mix/{name}"] = weight

        # Task/objective weights
        obj_weights = self.get_objective_weights()
        for name, weight in obj_weights.items():
            metrics[f"{prefix}/task_weight/{name}"] = weight

        # LR scales
        lr_scales = self.get_lr_scales()
        for name, scale in lr_scales.items():
            metrics[f"{prefix}/lr_scale/{name}"] = scale

        return metrics

    def get_hyperparameters_with_optimizer(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict[str, float]:
        """Get hyperparameters including actual LR values from optimizer.

        Args:
            optimizer: Optimizer to extract current LRs from

        Returns:
            Dict of hyperparameter_name -> value
        """
        metrics = self.get_hyperparameter_metrics()
        prefix = "hyperparameters"

        # Add actual learning rates from optimizer
        if optimizer is not None:
            actual_optimizer = optimizer
            if hasattr(optimizer, "_optimizer"):
                actual_optimizer = optimizer._optimizer

            for i, pg in enumerate(actual_optimizer.param_groups):
                # Try to get group name
                group_name = pg.get("name", f"group_{i}")
                if i < len(self.optimizer_param_groups):
                    group_name = self.optimizer_param_groups[i]

                metrics[f"{prefix}/lr/{group_name}"] = pg["lr"]

                # Also log weight decay if present
                if "weight_decay" in pg:
                    metrics[f"{prefix}/weight_decay/{group_name}"] = pg["weight_decay"]

        return metrics

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            "dataset_logits": self._dataset_logits,
            "objective_weights": self._objective_weights,
            "lr_scales": self._lr_scales,
            "dataset_velocity": self._dataset_velocity,
            "objective_velocity": self._objective_velocity,
            "lr_velocity": self._lr_velocity,
            "update_history": self._update_history,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict from checkpoint."""
        if "dataset_logits" in state_dict and state_dict["dataset_logits"] is not None:
            self._dataset_logits = state_dict["dataset_logits"].to(self.device)
        if "objective_weights" in state_dict and state_dict["objective_weights"] is not None:
            self._objective_weights = state_dict["objective_weights"].to(self.device)
        if "lr_scales" in state_dict and state_dict["lr_scales"] is not None:
            self._lr_scales = state_dict["lr_scales"].to(self.device)
        if "dataset_velocity" in state_dict and state_dict["dataset_velocity"] is not None:
            self._dataset_velocity = state_dict["dataset_velocity"].to(self.device)
        if "objective_velocity" in state_dict and state_dict["objective_velocity"] is not None:
            self._objective_velocity = state_dict["objective_velocity"].to(self.device)
        if "lr_velocity" in state_dict and state_dict["lr_velocity"] is not None:
            self._lr_velocity = state_dict["lr_velocity"].to(self.device)
        self._update_history = state_dict.get("update_history", [])
