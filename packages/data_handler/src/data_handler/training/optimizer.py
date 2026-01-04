"""Optimizer factory and utilities.

Supported optimizers:
- MuonClip (default): Muon + QK-clipping for training stability (used in Kimi K2 1T model)
  Reference: https://github.com/GAD-cell/muon-clip
- Muon: 2x compute efficiency vs AdamW, faster convergence
  Reference: https://arxiv.org/abs/2502.16982
- Adam/AdamW: Standard optimizers
  Reference: MobileLLM-R1 paper (arXiv:2509.24945) uses Adam with β1=0.9, β2=0.95
- APOLLO: Memory-efficient (1/8 of AdamW memory), optional dependency
  Reference: https://arxiv.org/abs/2412.05270
"""

from typing import Optional, Iterator, Any

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, Optimizer

try:
    from muon import Muon
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    Muon = None

try:
    from muon import MuonClip, MuonConfig
    MUONCLIP_AVAILABLE = True
except ImportError:
    MUONCLIP_AVAILABLE = False
    MuonClip = None
    MuonConfig = None

from data_handler.influence import DataInfCalculator, MixtureWeightCalculator
from data_handler.data.mixing import MixedDataset


class InfluenceAwareOptimizer(Optimizer):
    """
    Optimizer wrapper that handles influence-based data mixture updates.

    This wrapper intercepts the step() call to periodically trigger influence
    calculations and update dataset mixture weights.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mixture_calculator: MixtureWeightCalculator,
        mixed_dataset: MixedDataset,
        update_interval: int = 1000,
        learning_rate: float = 0.2,
        rank: int = 0,
    ):
        """
        Initialize InfluenceAwareOptimizer.

        Args:
            optimizer: Base optimizer to wrap
            mixture_calculator: Calculator for optimal mixture weights
            mixed_dataset: Dataset to update weights for
            update_interval: Number of steps between weight updates
            learning_rate: Learning rate for weight updates (how much to move towards optimal)
            rank: Process rank (for logging)
        """
        self.optimizer = optimizer
        self.mixture_calculator = mixture_calculator
        self.mixed_dataset = mixed_dataset
        self.update_interval = update_interval
        self.learning_rate = learning_rate
        self.rank = rank

        self.step_count = 0
        self._last_update_step = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """
        Perform a single optimization step and potentially update mixture weights.
        """
        # Perform standard optimization step
        loss = self.optimizer.step(closure)
        self.step_count += 1

        # Check if we should update weights
        if (
            self.step_count - self._last_update_step >= self.update_interval
            and self.mixture_calculator is not None
        ):
            self._update_mixture_weights()

        return loss

    def _update_mixture_weights(self):
        """Update dataset mixture weights based on influence."""
        import gc
        import time
        try:
            if self.rank == 0:
                print("\n" + "="*80)
                print(f"[INFLUENCE UPDATE] Step {self.step_count}")
                print("="*80)
                print(f"Starting influence-based mixture weight update...")
                next_update = self.step_count + self.update_interval
                print(f"Next update scheduled at step {next_update}")
                print()

            # Get current weights
            old_weights = self.mixed_dataset.get_current_weights()

            if self.rank == 0:
                print(f"Current weights before update:")
                for name, weight in old_weights.items():
                    print(f"  {name}: {weight:.4f}")
                print()

            # Refresh probe cache with current model state
            # Clear CUDA cache and run garbage collection to avoid OOM
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if self.rank == 0:
                print(f"Refreshing probe cache with current model state...")
                start_time = time.time()

            self.mixture_calculator.refresh_probe_cache(show_progress=(self.rank == 0))

            if self.rank == 0:
                elapsed = time.time() - start_time
                print(f"Probe cache refresh completed in {elapsed:.2f}s")
                print()

            # Compute new optimal weights
            if self.rank == 0:
                print(f"Computing optimal mixture weights (lr={self.learning_rate})...")

            new_weights = self.mixture_calculator.get_weight_update(
                old_weights,
                {},  # loaders placeholder
                learning_rate=self.learning_rate,
            )

            # Update dataset
            self.mixed_dataset.update_weights_from_influence(new_weights)
            self._last_update_step = self.step_count

            # Log changes
            if self.rank == 0:
                print()
                print(f"Updated mixture weights:")
                total_change = 0.0
                for name in sorted(new_weights.keys()):
                    old = old_weights.get(name, 0)
                    new = new_weights.get(name, 0)
                    change = new - old
                    total_change += abs(change)
                    sign = "↑" if change > 0 else "↓" if change < 0 else "="
                    print(f"  {sign} {name:15s}: {old:.4f} -> {new:.4f} ({change:+.4f})")
                print(f"\nTotal absolute change: {total_change:.4f}")
                print("="*80)
                print()

            # Clean up GPU memory after influence update
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            if self.rank == 0:
                print(f"\n[ERROR] Failed to update mixture weights at step {self.step_count}: {e}")
                import traceback
                traceback.print_exc()
                print("="*80)
                print()
            # Still clean up memory on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    no_decay_patterns: Optional[list[str]] = None,
) -> list[dict]:
    """Create parameter groups with/without weight decay.

    Typically, biases and layer norm parameters should not have weight decay.

    Args:
        model: Model to create parameter groups for
        weight_decay: Weight decay for parameters that should have it
        no_decay_patterns: Parameter name patterns that should not have decay

    Returns:
        List of parameter group dictionaries
    """
    if no_decay_patterns is None:
        no_decay_patterns = ["bias", "norm", "ln_"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter matches any no-decay pattern
        should_decay = True
        for pattern in no_decay_patterns:
            if pattern in name.lower():
                should_decay = False
                break

        if should_decay:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "muon",
    learning_rate: float = 4e-3,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    momentum: float = 0.95,
    nesterov: bool = True,
    **kwargs,
) -> Optimizer:
    """Create optimizer with proper parameter groups.

    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer:
            - "muonclip" (default): Muon + QK-clipping for training stability (Kimi K2)
            - "muon": 2x compute efficiency vs AdamW, faster convergence
            - "adam": Standard Adam
            - "adamw": AdamW with decoupled weight decay
            - "adamw_8bit": 8-bit AdamW via bitsandbytes (memory efficient)
            - "apollo": Memory-efficient (1/8 of AdamW memory), requires [apollo] extra
            - "apollo_mini": Extreme memory-efficient (1/1024 of AdamW), requires [apollo] extra
            - "sgd": Stochastic gradient descent
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters (β1, β2)
        eps: Adam epsilon
        momentum: Momentum for Muon/SGD (default: 0.95)
        nesterov: Use Nesterov momentum for Muon (default: True)
        **kwargs: Additional optimizer arguments

    Returns:
        Configured optimizer
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "muon":
        if not MUON_AVAILABLE:
            raise ImportError(
                "Muon optimizer not available. Install with: pip install muon-optimizer"
            )
        # Muon: separate hidden 2D weights from embed/head/biases
        # Hidden weights use Muon, others use AdamW at 0.1x LR
        hidden_weights = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Embeddings, heads, biases, norms -> AdamW
            if any(p in name.lower() for p in ["embed", "head", "lm_head", "bias", "norm", "ln_"]):
                other_params.append(param)
            elif param.ndim >= 2:
                hidden_weights.append(param)
            else:
                other_params.append(param)

        return Muon(
            muon_params=hidden_weights,
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            adamw_params=other_params,
            adamw_lr=learning_rate * 0.1,  # 0.1x LR for embed/head per Muon paper
            adamw_betas=betas,
            adamw_wd=weight_decay,
        )

    elif optimizer_type == "muonclip":
        # MuonClip: Muon + QK-clipping for training stability (used in Kimi K2)
        # Reference: https://github.com/GAD-cell/muon-clip
        if not MUONCLIP_AVAILABLE:
            raise ImportError(
                "MuonClip optimizer not available. "
                "Install with: pip install git+https://github.com/GAD-cell/muon-clip.git@main"
            )

        enable_clipping = kwargs.get("enable_clipping", True)
        model_config = kwargs.get("model_config", None)

        # QK-clipping requires model config with attention head info
        if enable_clipping and model_config is None:
            # Try to get config from model if it has one
            if hasattr(model, "config"):
                model_config = model.config
            else:
                import logging
                logging.warning(
                    "QK-clipping requires model_config with attention head info. "
                    "Disabling clipping."
                )
                enable_clipping = False

        # NOTE: muon-clip has a bug where writer is only created if log_dir is empty!
        # See: if not muon_config.log_dir : self.writer = SummaryWriter(...)
        # So we pass empty string to create the writer, otherwise flush_metrics() crashes
        unified_lr = kwargs.get("unified_lr", True)
        lr_adam = kwargs.get("lr_adam", learning_rate * 0.1)  # Default 0.1x for embed/head/norm
        config = MuonConfig(
            unified_lr=unified_lr,
            lr=learning_rate,
            lr_muon=learning_rate,
            lr_adam=lr_adam,
            muon_beta=momentum,
            muon_decay=weight_decay,
            adam_betas=betas,
            adam_eps=eps,
            adam_decay=weight_decay,
            enable_clipping=enable_clipping,
            clipping_threshold=kwargs.get("clipping_threshold", 50.0),
            clipping_alpha=kwargs.get("clipping_alpha", 0.5),
            log_dir="",  # Empty string triggers writer creation (muon-clip bug workaround)
        )
        return MuonClip(model, model_config, config)

    elif optimizer_type == "apollo":
        try:
            from apollo import Apollo
        except ImportError:
            raise ImportError(
                "APOLLO optimizer requires apollo-torch. "
                "Install with: pip install cheapertraining[apollo]"
            )
        param_groups = get_parameter_groups(model, weight_decay=weight_decay)
        return Apollo(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs,
        )

    elif optimizer_type == "apollo_mini":
        try:
            from apollo import ApolloMini
        except ImportError:
            raise ImportError(
                "APOLLO-Mini optimizer requires apollo-torch. "
                "Install with: pip install cheapertraining[apollo]"
            )
        param_groups = get_parameter_groups(model, weight_decay=weight_decay)
        return ApolloMini(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs,
        )

    elif optimizer_type == "adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "AdamW 8-bit optimizer requires bitsandbytes. "
                "Install with: pip install bitsandbytes"
            )
        param_groups = get_parameter_groups(model, weight_decay=weight_decay)
        return bnb.optim.AdamW8bit(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs,
        )

    # Standard optimizers use parameter groups with weight decay separation
    param_groups = get_parameter_groups(model, weight_decay=weight_decay)

    if optimizer_type == "adam":
        return Adam(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs,
        )
    elif optimizer_type == "adamw":
        return AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs,
        )
    elif optimizer_type == "sgd":
        return SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Supported: muonclip, muon, adam, adamw, adamw_8bit, apollo, apollo_mini, sgd"
        )


def get_num_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """Count number of parameters.

    Args:
        model: Model to count parameters for
        only_trainable: Only count trainable parameters

    Returns:
        Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
