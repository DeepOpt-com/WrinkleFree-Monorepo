"""Meta-optimization callback for Lightning trainer.

Implements efficient meta-optimization using:
- LDC-MTL for objective weight optimization (O(1) complexity)
- ODM/EXP3 for dataset weight optimization (~0% overhead)

References:
- LDC-MTL: https://arxiv.org/abs/2502.08585
- ODM: https://arxiv.org/abs/2312.02406
"""

import logging
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from wrinklefree.meta.config import MetaOptimizationConfig
from wrinklefree.meta.ldc_mtl import LDCMTLManager
from wrinklefree.meta.odm import OnlineDataMixer

logger = logging.getLogger(__name__)


class MetaOptimizerCallback(Callback):
    """Lightning callback for efficient meta-optimization.

    Integrates LDC-MTL (objective weights) and ODM (dataset weights) into
    the PyTorch Lightning training loop. Both methods are O(1) complexity
    with negligible overhead.

    The callback:
    - Initializes LDC-MTL and/or ODM during setup (if enabled and applicable)
    - Updates weights after each training batch
    - Logs metrics at configurable intervals
    - Saves/loads state with checkpoints

    Requirements:
    - LDC-MTL needs >1 objective (accessed via pl_module.objective_manager)
    - ODM needs >1 dataset (accessed via trainer.datamodule.get_mixed_dataset())

    Example:
        ```python
        config = MetaOptimizationConfig(
            enabled=True,
            ldc_mtl=LDCMTLConfig(enabled=True),
            odm=ODMConfig(enabled=True),
        )
        trainer = pl.Trainer(callbacks=[MetaOptimizerCallback(config)])
        ```

    WandB Metrics:
        - meta/ldc_mtl/objective_weight_{name}: Learned objective weights
        - meta/odm/dataset_weight_{name}: Dataset sampling probabilities
        - meta/odm/exploration_rate: Current EXP3 exploration rate
        - meta/odm/avg_reward_{name}: Per-domain average rewards
    """

    def __init__(self, config: MetaOptimizationConfig) -> None:
        """Initialize callback.

        Args:
            config: Meta-optimization configuration. Components are
                initialized lazily during setup() based on what's available.
        """
        super().__init__()
        self.config = config

        # Initialized lazily in setup()
        self.ldc_mtl: LDCMTLManager | None = None
        self.odm: OnlineDataMixer | None = None

        # Track state
        self._is_enabled = False
        self._total_steps: int | None = None

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        """Initialize meta-optimization components.

        Called by Lightning at the start of fit/validate/test/predict.
        Only initializes components during the "fit" stage.

        Initialization checks:
        - LDC-MTL: Requires pl_module.objective_manager with >1 objective
        - ODM: Requires trainer.datamodule with get_mixed_dataset() and >1 dataset
        """
        if stage != "fit":
            return

        if not self.config.enabled:
            logger.info("MetaOptimizerCallback: disabled by config")
            return

        # Get device
        device = next(pl_module.model.parameters()).device

        # Get total steps for ODM warmup calculation
        self._total_steps = trainer.max_steps if trainer.max_steps else 100000

        # Initialize LDC-MTL for objective weights
        if self.config.ldc_mtl.enabled:
            objective_names = self._get_objective_names(pl_module)
            if objective_names and len(objective_names) > 1:
                self.ldc_mtl = LDCMTLManager(
                    objective_names,
                    self.config.ldc_mtl,
                    device,
                )
                logger.info(
                    f"MetaOptimizerCallback: LDC-MTL enabled for {objective_names}"
                )
            else:
                logger.info(
                    "MetaOptimizerCallback: LDC-MTL disabled (need >1 objective)"
                )

        # Initialize ODM for dataset weights
        if self.config.odm.enabled:
            dataset_names = self._get_dataset_names(trainer.datamodule)
            if dataset_names and len(dataset_names) > 1:
                self.odm = OnlineDataMixer(dataset_names, self.config.odm)
                logger.info(
                    f"MetaOptimizerCallback: ODM enabled for {dataset_names}"
                )
            else:
                logger.info(
                    "MetaOptimizerCallback: ODM disabled (need >1 dataset)"
                )

        self._is_enabled = self.ldc_mtl is not None or self.odm is not None

        if self._is_enabled:
            logger.info("MetaOptimizerCallback: enabled and ready")

    def _get_objective_names(self, pl_module: pl.LightningModule) -> list[str]:
        """Get objective names from the Lightning module.

        Looks for pl_module.objective_manager and extracts objective names
        from its objectives dict or get_objective_names() method.

        Args:
            pl_module: The Lightning module being trained.

        Returns:
            List of objective names, or empty list if not found.
        """
        if hasattr(pl_module, "objective_manager"):
            manager = pl_module.objective_manager
            if hasattr(manager, "objectives"):
                return list(manager.objectives.keys())
            if hasattr(manager, "get_objective_names"):
                return manager.get_objective_names()
        return []

    def _get_dataset_names(self, datamodule: Any) -> list[str]:
        """Get dataset names from the datamodule.

        Looks for datamodule.get_mixed_dataset() and extracts dataset names
        from the mixed dataset's get_dataset_names() or dataset_names attr.

        Args:
            datamodule: The Lightning DataModule (may be None).

        Returns:
            List of dataset names, or empty list if not found.
        """
        if datamodule is None:
            return []

        if hasattr(datamodule, "get_mixed_dataset"):
            mixed = datamodule.get_mixed_dataset()
            if mixed is not None:
                if hasattr(mixed, "get_dataset_names"):
                    return mixed.get_dataset_names()
                if hasattr(mixed, "dataset_names"):
                    return mixed.dataset_names

        return []

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Update meta-parameters after each training batch.

        For ODM: Updates dataset weights based on per-domain losses from outputs.
        For LDC-MTL: Steps the router optimizer (gradients computed during forward).
        """
        if not self._is_enabled:
            return

        step = trainer.global_step

        # Update ODM with per-domain losses (if available)
        if self.odm is not None:
            self._update_odm(trainer, outputs, step)

        # LDC-MTL updates happen during forward pass (gradients flow through router)
        # We just need to step the optimizer periodically
        if self.ldc_mtl is not None:
            self.ldc_mtl.step()

        # Log periodically
        if step % self.config.log_interval == 0:
            self._log_metrics(pl_module, step)

    def _update_odm(
        self,
        trainer: pl.Trainer,
        outputs: Any,
        step: int,
    ) -> None:
        """Update ODM dataset weights based on training outputs.

        Extracts per-domain losses from outputs and updates the bandit.
        Skips during warmup period (uses uniform weights instead).
        """
        # Check warmup
        if self.odm.is_in_warmup(step, self._total_steps):
            return

        # Get per-domain losses from outputs
        losses_per_domain = None

        if isinstance(outputs, dict):
            if "losses_per_domain" in outputs:
                losses_per_domain = outputs["losses_per_domain"]
            elif "domain_losses" in outputs:
                losses_per_domain = outputs["domain_losses"]

        if losses_per_domain is None:
            # Try to get from batch metadata
            if "domain" in outputs.get("batch", {}):
                domain = outputs["batch"]["domain"]
                loss = outputs.get("loss", 0.0)
                if hasattr(loss, "item"):
                    loss = loss.item()
                losses_per_domain = {domain: loss}

        if losses_per_domain:
            self.odm.update(losses_per_domain)

            # Apply new weights to dataset
            weights = self.odm.get_sampling_weights()
            self._apply_dataset_weights(trainer.datamodule, weights)

    def _apply_dataset_weights(
        self,
        datamodule: Any,
        weights: dict[str, float],
    ) -> None:
        """Apply updated sampling weights to the dataset.

        Looks for set_weights() or update_weights() method on the mixed dataset.
        """
        if datamodule is None:
            return

        if hasattr(datamodule, "get_mixed_dataset"):
            mixed = datamodule.get_mixed_dataset()
            if mixed is not None:
                if hasattr(mixed, "set_weights"):
                    mixed.set_weights(weights)
                elif hasattr(mixed, "update_weights"):
                    mixed.update_weights(weights)

    def _log_metrics(
        self,
        pl_module: pl.LightningModule,
        step: int,
    ) -> None:
        """Log meta-optimization metrics."""
        # LDC-MTL metrics
        if self.ldc_mtl is not None:
            metrics = self.ldc_mtl.get_wandb_metrics()
            for name, value in metrics.items():
                pl_module.log(name, value, prog_bar=False)

        # ODM metrics
        if self.odm is not None:
            metrics = self.odm.get_wandb_metrics()
            for name, value in metrics.items():
                pl_module.log(name, value, prog_bar=False)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Save meta-optimization state to checkpoint.

        Stores LDC-MTL router weights and ODM bandit state under
        the key "meta_optimizer_state" in the checkpoint dict.
        """
        state = {}

        if self.ldc_mtl is not None:
            state["ldc_mtl"] = self.ldc_mtl.state_dict()

        if self.odm is not None:
            state["odm"] = self.odm.state_dict()

        if state:
            checkpoint["meta_optimizer_state"] = state

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Load meta-optimization state from checkpoint.

        Restores LDC-MTL router weights and ODM bandit state from
        the "meta_optimizer_state" key if present.
        """
        if "meta_optimizer_state" not in checkpoint:
            return

        state = checkpoint["meta_optimizer_state"]

        if self.ldc_mtl is not None and "ldc_mtl" in state:
            self.ldc_mtl.load_state_dict(state["ldc_mtl"])

        if self.odm is not None and "odm" in state:
            self.odm.load_state_dict(state["odm"])

        logger.info("Restored meta-optimizer state from checkpoint")

    @property
    def is_enabled(self) -> bool:
        """Check if meta-optimization is active.

        Returns:
            True if at least one of LDC-MTL or ODM was successfully initialized.
        """
        return self._is_enabled

    def get_current_weights(self) -> dict[str, dict[str, float]]:
        """Get all current meta-parameter weights.

        Returns:
            Dict with optional "objective" and "dataset" keys, each mapping
            to a dict of name -> weight. Empty dict if nothing is enabled.

        Example:
            >>> callback.get_current_weights()
            {"objective": {"ce": 0.6, "dlm": 0.4}, "dataset": {"web": 0.5, "code": 0.5}}
        """
        result: dict[str, dict[str, float]] = {}

        if self.ldc_mtl is not None:
            result["objective"] = self.ldc_mtl.get_weights()

        if self.odm is not None:
            result["dataset"] = self.odm.get_sampling_weights()

        return result
