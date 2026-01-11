"""Per-layer learning rate meta-optimization.

Learns per-layer LR multipliers via direct gradient descent. Uses gradient
norms as stability signal - high gradient norms push multipliers down to
stabilize training.

Inspired by LARS (Layer-wise Adaptive Rate Scaling) but learned dynamically
rather than using a fixed formula.

Reference:
    https://arxiv.org/abs/1708.03888 (LARS)
"""

import logging
import math
import re

import torch
import torch.nn as nn
from torch import Tensor

from wf_train.meta.config import LayerLRConfig

logger = logging.getLogger(__name__)


class LayerLRManager:
    """Learns per-layer LR multipliers via direct optimization.

    Uses two penalty terms:
    1. Mean-centering: Keeps geometric mean of multipliers near 1.0
    2. Stability: High gradient norms push multipliers DOWN for that layer

    The stability term is key: if a layer has exploding gradients, its LR
    will automatically decrease to stabilize training.

    Example:
        >>> manager = LayerLRManager(model, config, device)
        >>> # In training loop, after backward:
        >>> manager.collect_grad_norms(model)
        >>> # Before optimizer step:
        >>> multipliers = manager.apply_multipliers(optimizer, step, warmup_steps)
        >>> optimizer.step()
        >>> # After optimizer step:
        >>> manager.step()
        >>> manager.restore_lrs(optimizer)
    """

    def __init__(
        self,
        model: nn.Module,
        config: LayerLRConfig,
        device: torch.device,
    ) -> None:
        """Initialize the layer LR manager.

        Args:
            model: Model to track layers from
            config: LayerLR configuration
            device: Device for learnable parameters
        """
        self.config = config
        self.device = device

        # Build layer -> param_names mapping
        self._layer_param_mapping = self._build_layer_param_mapping(model)
        self.num_layers = len(self._layer_param_mapping)

        if self.num_layers == 0:
            raise ValueError("No transformer layers found in model")

        # Learnable log-multipliers (exp gives multipliers, init to 0 = mult of 1)
        self.log_multipliers = nn.Parameter(
            torch.zeros(self.num_layers, device=device)
        )

        # Optimizer for the multipliers
        self.optimizer = torch.optim.Adam([self.log_multipliers], lr=config.lr)

        # EMA of gradient norms per layer (for stability penalty)
        self._ema_grad_norms = torch.ones(self.num_layers, device=device)

        # Current multipliers (for logging)
        self._current_multipliers: dict[int, float] = {
            i: 1.0 for i in range(self.num_layers)
        }

        # Base LRs storage
        self._base_lrs_stored = False

        logger.info(
            f"LayerLRManager initialized: {self.num_layers} layers, "
            f"lr={config.lr}, bounds=[{config.min_multiplier}, {config.max_multiplier}]"
        )

    def _build_layer_param_mapping(self, model: nn.Module) -> dict[int, list[str]]:
        """Map layer indices to parameter names.

        Looks for HuggingFace-style layer naming: model.layers.{N}.xxx
        """
        layer_params: dict[int, list[str]] = {}
        pattern = re.compile(r"model\.layers\.(\d+)\.")

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            match = pattern.search(name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in layer_params:
                    layer_params[layer_idx] = []
                layer_params[layer_idx].append(name)

        return layer_params

    def get_multipliers(self) -> Tensor:
        """Get current LR multipliers (clamped to bounds)."""
        return self.log_multipliers.exp().clamp(
            self.config.min_multiplier,
            self.config.max_multiplier,
        )

    def _ensure_device(self, target_device: torch.device) -> None:
        """Ensure all tensors are on the target device.

        Called when model moves from CPU to GPU (e.g., during BatchSizeFinder).
        Must recreate optimizer after moving log_multipliers.
        """
        if target_device == self.device:
            return

        logger.info(f"LayerLR moving from {self.device} to {target_device}")

        # Move EMA tensor
        self._ema_grad_norms = self._ema_grad_norms.to(target_device)

        # Move log_multipliers parameter
        self.log_multipliers.data = self.log_multipliers.data.to(target_device)

        # Must recreate optimizer - old one has references to old device params
        self.optimizer = torch.optim.Adam([self.log_multipliers], lr=self.config.lr)

        self.device = target_device

    def collect_grad_norms(self, model: nn.Module) -> None:
        """Compute and store gradient norms per layer.

        Must be called AFTER backward() but BEFORE optimizer.step().
        Updates EMA of gradient norms for stability penalty.
        """
        layer_norms: dict[int, float] = {}
        param_dict = dict(model.named_parameters())

        # Check if we need to move to a different device
        first_param = next(iter(param_dict.values()), None)
        if first_param is not None and first_param.device != self.device:
            self._ensure_device(first_param.device)

        for layer_idx, param_names in self._layer_param_mapping.items():
            grad_squared_sum = 0.0
            for name in param_names:
                param = param_dict.get(name)
                if param is not None and param.grad is not None:
                    grad_squared_sum += param.grad.detach().pow(2).sum().item()
            layer_norms[layer_idx] = math.sqrt(grad_squared_sum)

        # Normalize to [0, 1] range (relative to max)
        max_norm = max(layer_norms.values()) if layer_norms else 1.0
        if max_norm < 1e-8:
            max_norm = 1.0

        # Update EMA
        alpha = self.config.ema_decay
        for idx in range(self.num_layers):
            norm = layer_norms.get(idx, 0.0) / max_norm
            self._ema_grad_norms[idx] = (
                alpha * self._ema_grad_norms[idx] + (1 - alpha) * norm
            )

    def compute_penalty(self) -> Tensor:
        """Compute penalty for multiplier optimization.

        Two components:
        1. Mean-centering: log(multipliers).mean()^2 - keeps geometric mean ~1
        2. Balance: (grad_norms * multipliers - target)^2 - bidirectional!
           - High grad + high mult → penalty → decrease LR
           - Low grad + low mult → penalty → increase LR

        This is BIDIRECTIONAL: if learning is too slow (low grads), it
        increases LR. If learning is unstable (high grads), it decreases LR.

        Returns:
            Scalar penalty tensor for backward()
        """
        multipliers = self.get_multipliers()

        # Mean-centering: geometric mean should stay near 1.0
        mean_penalty = self.log_multipliers.mean() ** 2

        # Bidirectional balance: target grad_norm * multiplier ≈ 0.5
        # - If grad_norm is high (0.8) and mult is high (2.0): product=1.6 → decrease mult
        # - If grad_norm is low (0.2) and mult is low (0.5): product=0.1 → increase mult
        target = 0.5  # Target product of normalized_grad_norm * multiplier
        products = self._ema_grad_norms * multipliers
        balance_penalty = ((products - target) ** 2).mean()

        return (
            self.config.lambda_mean * mean_penalty
            + 0.1 * balance_penalty
        )

    def apply_multipliers(
        self,
        optimizer: torch.optim.Optimizer,
        step: int,
        warmup_steps: int,
    ) -> dict[int, float]:
        """Apply LR multipliers to optimizer param groups.

        Args:
            optimizer: The main optimizer (may be CombinedMuonAdamWOptimizer)
            step: Current training step
            warmup_steps: Number of warmup steps (multipliers=1.0 during warmup)

        Returns:
            Dict of layer_idx -> applied multiplier
        """
        # During warmup, don't modify LRs (grad stats unreliable)
        if step < warmup_steps:
            return {i: 1.0 for i in range(self.num_layers)}

        # Get current multipliers
        multipliers = self.get_multipliers()
        mult_dict = {i: multipliers[i].item() for i in range(self.num_layers)}
        self._current_multipliers = mult_dict

        # Apply to optimizer param groups
        self._apply_to_optimizer(optimizer, mult_dict)

        return mult_dict

    def _apply_to_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        multipliers: dict[int, float],
    ) -> None:
        """Apply multipliers to optimizer param groups."""
        # Handle CombinedMuonAdamWOptimizer wrapper
        if hasattr(optimizer, "muon_opt") and hasattr(optimizer, "adam_opt"):
            self._apply_to_single_optimizer(optimizer.muon_opt, multipliers)
            self._apply_to_single_optimizer(optimizer.adam_opt, multipliers)
        else:
            self._apply_to_single_optimizer(optimizer, multipliers)

        self._base_lrs_stored = True

    def _apply_to_single_optimizer(
        self,
        opt: torch.optim.Optimizer,
        multipliers: dict[int, float],
    ) -> None:
        """Apply multipliers to a single optimizer's param groups.

        For simplicity, applies average multiplier to all groups.
        A more sophisticated version could track exact param->layer mapping.
        """
        avg_mult = sum(multipliers.values()) / len(multipliers) if multipliers else 1.0

        for group in opt.param_groups:
            # Store base LR if not already stored
            if "_base_lr" not in group:
                group["_base_lr"] = group["lr"]
            # Apply average multiplier
            group["lr"] = group["_base_lr"] * avg_mult

    def restore_lrs(self, optimizer: torch.optim.Optimizer) -> None:
        """Restore base LRs after optimizer step."""
        if not self._base_lrs_stored:
            return

        if hasattr(optimizer, "muon_opt") and hasattr(optimizer, "adam_opt"):
            self._restore_single_optimizer(optimizer.muon_opt)
            self._restore_single_optimizer(optimizer.adam_opt)
        else:
            self._restore_single_optimizer(optimizer)

    def _restore_single_optimizer(self, opt: torch.optim.Optimizer) -> None:
        for group in opt.param_groups:
            if "_base_lr" in group:
                group["lr"] = group["_base_lr"]

    def step(self) -> None:
        """Update multiplier parameters via penalty gradient."""
        self.optimizer.zero_grad()
        penalty = self.compute_penalty()
        penalty.backward()
        self.optimizer.step()

    def get_wandb_metrics(self, prefix: str = "meta/layer_lr") -> dict[str, float]:
        """Get metrics for WandB logging."""
        metrics = {}

        # Per-layer multipliers
        for layer_idx, mult in self._current_multipliers.items():
            metrics[f"{prefix}/multiplier_layer_{layer_idx}"] = mult

        # Per-layer gradient norms
        for layer_idx in range(self.num_layers):
            metrics[f"{prefix}/grad_norm_layer_{layer_idx}"] = (
                self._ema_grad_norms[layer_idx].item()
            )

        # Summary statistics
        if self._current_multipliers:
            mults = list(self._current_multipliers.values())
            metrics[f"{prefix}/mean_multiplier"] = sum(mults) / len(mults)
            metrics[f"{prefix}/min_multiplier"] = min(mults)
            metrics[f"{prefix}/max_multiplier"] = max(mults)

        return metrics

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            "log_multipliers": self.log_multipliers.detach().cpu(),
            "optimizer_state": self.optimizer.state_dict(),
            "ema_grad_norms": self._ema_grad_norms.cpu(),
            "current_multipliers": self._current_multipliers.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        if "log_multipliers" in state:
            self.log_multipliers.data.copy_(state["log_multipliers"].to(self.device))
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if "ema_grad_norms" in state:
            self._ema_grad_norms = state["ema_grad_norms"].to(self.device)
        if "current_multipliers" in state:
            self._current_multipliers = dict(state["current_multipliers"])
        logger.info("LayerLRManager state restored from checkpoint")
