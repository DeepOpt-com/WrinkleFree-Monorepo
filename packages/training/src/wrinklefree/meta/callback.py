"""Meta-optimization callback for Lightning trainer.

Implements the outer loop that optimizes:
- Dataset mixture weights
- Objective weights
- Learning rate scales

Using influence-based gradient estimation with Pareto multi-objective optimization.

References:
- LibMOON (NeurIPS 2024): https://arxiv.org/abs/2409.02969
- DataInf (ICLR 2024): https://openreview.net/forum?id=9m02ib92Wz
"""

import logging
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torch.utils.data import DataLoader

from wrinklefree.meta.config import MetaOptimizationConfig
from wrinklefree.meta.manager import MetaParameterManager
from wrinklefree.meta.pareto import ParetoGradientSolver

logger = logging.getLogger(__name__)


class MetaOptimizerCallback(Callback):
    """Lightning callback for meta-optimization outer loop.

    Periodically:
    1. Computes gradients of validation objectives w.r.t. model parameters
    2. Estimates meta-gradients via influence functions
    3. Uses Pareto solver to find optimal update direction
    4. Updates meta-parameters (dataset weights, objective weights, LR scales)
    5. Applies updated meta-parameters to training components

    This extends/replaces InfluenceTrackerCallback when meta_optimization.enabled=True.

    Usage:
        callbacks.append(MetaOptimizerCallback(
            config=MetaOptimizationConfig(...),
            validation_loaders={"c4": c4_loader, "code": code_loader},
        ))
    """

    def __init__(
        self,
        config: MetaOptimizationConfig,
        validation_loaders: Optional[dict[str, DataLoader]] = None,
    ):
        """Initialize callback.

        Args:
            config: Meta-optimization configuration
            validation_loaders: Dict of validation dataloaders for multi-objective optimization.
                If None, will try to get from datamodule.
        """
        super().__init__()
        self.config = config
        self.validation_loaders = validation_loaders or {}

        # Will be initialized in setup()
        self.meta_manager: Optional[MetaParameterManager] = None
        self.pareto_solver: Optional[ParetoGradientSolver] = None
        self.meta_gradient_calc = None

        # Track state
        self._is_enabled = False
        self._last_update_step = -1

        # Cached validation gradients (refreshed periodically)
        self._cached_val_gradients: dict[str, Tensor] = {}
        self._val_cache_step = 0

        # Hyperparameter logging interval (log frequently for visibility)
        self._hyperparam_log_interval = config.log_interval if hasattr(config, "log_interval") else 100

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        """Initialize meta-optimization components."""
        if stage != "fit":
            return

        if not self.config.enabled:
            logger.info("MetaOptimizerCallback: disabled by config")
            return

        # Get datamodule
        datamodule = trainer.datamodule
        if datamodule is None:
            logger.warning("MetaOptimizerCallback: no datamodule, disabling")
            return

        # Get validation loaders if not provided
        if not self.validation_loaders:
            self.validation_loaders = self._get_validation_loaders(datamodule)

        if not self.validation_loaders:
            logger.warning("MetaOptimizerCallback: no validation loaders, disabling")
            return

        # Get component names
        dataset_names = self._get_dataset_names(datamodule)
        objective_names = self._get_objective_names(pl_module)
        optimizer_groups = self._get_optimizer_group_names(pl_module)

        # Initialize components
        device = next(pl_module.model.parameters()).device

        self.meta_manager = MetaParameterManager(
            config=self.config,
            dataset_names=dataset_names,
            objective_names=objective_names,
            optimizer_param_groups=optimizer_groups if self.config.optimize_learning_rates else None,
            device=device,
        )

        self.pareto_solver = ParetoGradientSolver(self.config.pareto)

        # Initialize meta-gradient calculator
        try:
            from math_utils.influence import MetaGradientCalculator, InfluenceConfig

            self.meta_gradient_calc = MetaGradientCalculator(
                model=pl_module.model,
                config=InfluenceConfig(lambda_val=self.config.gradient.lambda_reg),
            )
        except ImportError:
            logger.error("MetaOptimizerCallback: math_utils not available, disabling")
            return

        self._is_enabled = True
        logger.info(
            f"MetaOptimizerCallback: enabled with "
            f"{len(dataset_names)} datasets, "
            f"{len(objective_names)} objectives, "
            f"{len(self.validation_loaders)} validation objectives"
        )

    def _get_validation_loaders(self, datamodule) -> dict[str, DataLoader]:
        """Get validation loaders from datamodule or config."""
        loaders = {}

        # Try to get probe dataloaders (for influence-style validation)
        if hasattr(datamodule, "get_probe_dataloaders"):
            probe_loaders = datamodule.get_probe_dataloaders()
            if probe_loaders:
                loaders.update(probe_loaders)

        # Try to get standard validation loader
        if hasattr(datamodule, "val_dataloader"):
            val_loader = datamodule.val_dataloader()
            if val_loader is not None:
                loaders["validation"] = val_loader

        return loaders

    def _get_dataset_names(self, datamodule) -> list[str]:
        """Get dataset names from datamodule."""
        if hasattr(datamodule, "get_mixed_dataset"):
            mixed = datamodule.get_mixed_dataset()
            if mixed is not None and hasattr(mixed, "get_dataset_names"):
                return mixed.get_dataset_names()
            if mixed is not None and hasattr(mixed, "dataset_names"):
                return mixed.dataset_names

        return []

    def _get_objective_names(self, pl_module: pl.LightningModule) -> list[str]:
        """Get objective names from module."""
        if hasattr(pl_module, "objective_manager"):
            manager = pl_module.objective_manager
            if hasattr(manager, "objectives"):
                return list(manager.objectives.keys())
            if hasattr(manager, "get_objective_names"):
                return manager.get_objective_names()

        return []

    def _get_optimizer_group_names(self, pl_module: pl.LightningModule) -> list[str]:
        """Get optimizer parameter group names."""
        # Common pattern for our codebase: muon + adamw
        return ["muon", "adamw"]

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Cache initial validation gradients."""
        if not self._is_enabled:
            return

        logger.info("MetaOptimizerCallback: caching initial validation gradients")
        self._cache_validation_gradients(pl_module)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Check if meta-update is needed and log hyperparameters periodically."""
        if not self._is_enabled:
            return

        step = trainer.global_step

        # Log hyperparameters periodically for WandB visibility
        if step % self._hyperparam_log_interval == 0:
            self._log_hyperparameters(trainer, pl_module)

        # Check warmup
        if step < self.config.warmup_steps:
            return

        # Check update interval
        if step - self._last_update_step < self.config.update_interval:
            return

        # Perform meta-update
        self._meta_update(trainer, pl_module)
        self._last_update_step = step

    def _cache_validation_gradients(self, pl_module: pl.LightningModule) -> None:
        """Cache validation gradients for each objective."""
        if self.meta_gradient_calc is None:
            return

        self._cached_val_gradients.clear()

        for name, loader in self.validation_loaders.items():
            logger.info(f"Computing validation gradient for {name}...")
            grad = self.meta_gradient_calc.compute_validation_gradient(
                loader,
                max_samples=self.config.gradient.samples_per_source,
                show_progress=False,
            )
            self._cached_val_gradients[name] = grad

        logger.info(f"Cached {len(self._cached_val_gradients)} validation gradients")

    def _meta_update(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Perform one meta-optimization step."""
        logger.info(f"Meta-optimization update at step {trainer.global_step}")

        # CRITICAL: Recompute validation gradients with current model state
        # Using stale gradients from step 0 would give incorrect meta-gradients
        self._cache_validation_gradients(pl_module)

        # Get current components
        datamodule = trainer.datamodule
        objective_manager = getattr(pl_module, "objective_manager", None)
        optimizer = trainer.optimizers[0] if trainer.optimizers else None

        # 1. Collect meta-gradients for each validation objective
        meta_grads_per_val_obj: dict[str, dict[str, dict[str, float]]] = {}

        for val_name, val_grad in self._cached_val_gradients.items():
            val_grad = val_grad.to(next(pl_module.model.parameters()).device)

            meta_grads: dict[str, dict[str, float]] = {}

            # Dataset meta-gradients
            if self.config.optimize_dataset_weights and datamodule is not None:
                dataset_loaders = self._get_dataset_loaders(datamodule)
                if dataset_loaders:
                    meta_grads["dataset"] = self.meta_gradient_calc.compute_dataset_meta_gradients(
                        dataset_loaders=dataset_loaders,
                        validation_gradient=val_grad,
                        current_weights=self.meta_manager.get_dataset_weights(),
                        samples_per_source=self.config.gradient.samples_per_source,
                    )

            # Objective meta-gradients
            if self.config.optimize_objective_weights and objective_manager is not None:
                obj_grads = self._get_objective_gradients(pl_module)
                if obj_grads:
                    meta_grads["objective"] = self.meta_gradient_calc.compute_objective_meta_gradients(
                        objective_gradients=obj_grads,
                        validation_gradient=val_grad,
                    )

            # LR meta-gradients
            if self.config.optimize_learning_rates and optimizer is not None:
                meta_grads["lr"] = self.meta_gradient_calc.compute_lr_meta_gradients(
                    optimizer=optimizer,
                    validation_gradient=val_grad,
                )

            meta_grads_per_val_obj[val_name] = meta_grads

        # 2. Solve Pareto problem
        combined_meta_grads = self._solve_pareto(meta_grads_per_val_obj)

        # 3. Update meta-parameters
        self.meta_manager.update_from_gradients(
            dataset_grads=combined_meta_grads.get("dataset"),
            objective_grads=combined_meta_grads.get("objective"),
            lr_grads=combined_meta_grads.get("lr"),
        )

        # 4. Apply to training
        mixed_dataset = datamodule.get_mixed_dataset() if datamodule else None
        self.meta_manager.apply_to_training(
            mixed_dataset=mixed_dataset,
            objective_manager=objective_manager,
            optimizer=optimizer,
        )

        # 5. Log metrics
        self._log_meta_update(pl_module)

    def _get_dataset_loaders(self, datamodule) -> dict[str, DataLoader]:
        """Get per-dataset loaders for meta-gradient computation."""
        # Try to get source-specific loaders
        if hasattr(datamodule, "get_source_dataloaders"):
            return datamodule.get_source_dataloaders()

        # Fallback: use probe loaders
        if hasattr(datamodule, "get_probe_dataloaders"):
            return datamodule.get_probe_dataloaders() or {}

        return {}

    def _get_objective_gradients(self, pl_module: pl.LightningModule) -> dict[str, Tensor]:
        """Get per-objective gradients from objective manager."""
        if hasattr(pl_module, "objective_manager"):
            manager = pl_module.objective_manager
            if hasattr(manager, "get_objective_gradients"):
                return manager.get_objective_gradients()
            if hasattr(manager, "_cached_objective_gradients"):
                return manager._cached_objective_gradients

        return {}

    def _solve_pareto(
        self,
        meta_grads_per_val_obj: dict[str, dict[str, dict[str, float]]],
    ) -> dict[str, dict[str, float]]:
        """Solve Pareto problem to combine meta-gradients across validation objectives.

        Args:
            meta_grads_per_val_obj: Nested dict:
                validation_objective_name -> meta_param_type -> param_name -> gradient

        Returns:
            Combined meta-gradients: meta_param_type -> param_name -> gradient
        """
        if not meta_grads_per_val_obj:
            return {}

        val_obj_names = list(meta_grads_per_val_obj.keys())
        if len(val_obj_names) == 1:
            # Single validation objective: just return its gradients
            return meta_grads_per_val_obj[val_obj_names[0]]

        # Collect all meta-parameter types
        all_meta_types = set()
        for grads in meta_grads_per_val_obj.values():
            all_meta_types.update(grads.keys())

        combined = {}

        for meta_type in all_meta_types:
            # Collect param names
            all_param_names = set()
            for grads in meta_grads_per_val_obj.values():
                if meta_type in grads:
                    all_param_names.update(grads[meta_type].keys())

            if not all_param_names:
                continue

            # Build gradient vectors for each validation objective
            param_names = sorted(all_param_names)
            grad_vectors = []

            for val_name in val_obj_names:
                if meta_type in meta_grads_per_val_obj[val_name]:
                    grads = meta_grads_per_val_obj[val_name][meta_type]
                    vec = torch.tensor([grads.get(p, 0.0) for p in param_names])
                    grad_vectors.append(vec)
                else:
                    grad_vectors.append(torch.zeros(len(param_names)))

            # Solve Pareto
            pareto_grad = self.pareto_solver.solve(grad_vectors)

            # Convert back to dict
            combined[meta_type] = {
                param_names[i]: pareto_grad[i].item()
                for i in range(len(param_names))
            }

        return combined

    def _log_meta_update(self, pl_module: pl.LightningModule) -> None:
        """Log meta-parameter values to WandB under meta/ prefix."""
        metrics = self.meta_manager.get_wandb_metrics(prefix="meta")
        for name, value in metrics.items():
            pl_module.log(name, value, prog_bar=False)

    def _log_hyperparameters(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log hyperparameters to WandB under dedicated hyperparameters/ section.

        Logs:
        - hyperparameters/data_mix/{name}: Dataset mixture weights
        - hyperparameters/task_weight/{name}: Objective/task weights
        - hyperparameters/lr/{name}: Actual learning rates
        - hyperparameters/lr_scale/{name}: LR scaling factors (if meta-optimized)
        - hyperparameters/weight_decay/{name}: Weight decay values
        """
        if self.meta_manager is None:
            return

        # Get optimizer for LR logging
        optimizer = trainer.optimizers[0] if trainer.optimizers else None

        # Get hyperparameters including optimizer state
        metrics = self.meta_manager.get_hyperparameters_with_optimizer(optimizer)

        # Log each metric
        for name, value in metrics.items():
            pl_module.log(name, value, prog_bar=False)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Save meta-parameter state to checkpoint."""
        if self.meta_manager is not None:
            checkpoint["meta_optimizer_state"] = self.meta_manager.state_dict()

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Load meta-parameter state from checkpoint."""
        if self.meta_manager is not None and "meta_optimizer_state" in checkpoint:
            self.meta_manager.load_state_dict(checkpoint["meta_optimizer_state"])
            logger.info("Restored meta-optimizer state from checkpoint")

    @property
    def is_enabled(self) -> bool:
        """Check if meta-optimization is active."""
        return self._is_enabled

    def get_current_weights(self) -> dict[str, dict[str, float]]:
        """Get all current meta-parameter weights."""
        if self.meta_manager is None:
            return {}

        return {
            "dataset": self.meta_manager.get_dataset_weights(),
            "objective": self.meta_manager.get_objective_weights(),
            "lr_scales": self.meta_manager.get_lr_scales(),
        }
