"""Meta-optimization callback for Lightning trainer.

Implements efficient meta-optimization using:
- LDC-MTL for objective weight optimization (O(1) complexity)
- ODM/EXP3 for dataset weight optimization (~0% overhead)
- LayerLR for per-layer learning rate optimization

References:
- LDC-MTL: https://arxiv.org/abs/2502.08585
- ODM: https://arxiv.org/abs/2312.02406
- LayerLR: Inspired by LARS (https://arxiv.org/abs/1708.03888)
"""

import logging
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from wf_train.meta.config import MetaOptimizationConfig
from wf_train.meta.ldc_mtl import LDCMTLManager
from wf_train.meta.layer_lr import LayerLRManager
from wf_train.meta.odm import OnlineDataMixer

logger = logging.getLogger(__name__)


class MetaOptimizerCallback(Callback):
    """Lightning callback for efficient meta-optimization.

    Integrates LDC-MTL (objective weights), ODM (dataset weights), and
    LayerLR (per-layer learning rates) into the PyTorch Lightning training
    loop. All methods are O(1) complexity with negligible overhead.

    The callback:
    - Initializes components during setup (if enabled and applicable)
    - Updates weights/multipliers after each training batch
    - Logs metrics at configurable intervals
    - Saves/loads state with checkpoints

    Requirements:
    - LDC-MTL needs >1 objective (accessed via pl_module.objective_manager)
    - ODM needs >1 dataset (accessed via trainer.datamodule.get_mixed_dataset())
    - LayerLR needs >1 transformer layer (detected from model structure)

    Example:
        ```python
        config = MetaOptimizationConfig(
            enabled=True,
            ldc_mtl=LDCMTLConfig(enabled=True),
            odm=ODMConfig(enabled=True),
            layer_lr=LayerLRConfig(enabled=True),
        )
        trainer = pl.Trainer(callbacks=[MetaOptimizerCallback(config)])
        ```

    WandB Metrics:
        - meta/ldc_mtl/objective_weight_{name}: Learned objective weights
        - meta/odm/dataset_weight_{name}: Dataset sampling probabilities
        - meta/odm/exploration_rate: Current EXP3 exploration rate
        - meta/odm/avg_reward_{name}: Per-domain average rewards
        - meta/layer_lr/multiplier_layer_{i}: Per-layer LR multipliers
        - meta/layer_lr/grad_norm_layer_{i}: Per-layer gradient norms
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
        self.layer_lr: LayerLRManager | None = None

        # Track state
        self._is_enabled = False
        self._total_steps: int | None = None
        self._warmup_steps: int = 0

        # LayerLR deferred init - stored for on_train_start
        self._layer_lr_pending_init = False

        # Track active objectives for LDC-MTL reinitialization on curriculum changes
        self._current_active_objectives: list[str] | None = None
        self._ldc_mtl_device: torch.device | None = None

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
        # IMPORTANT: Initialize with ACTIVE objectives only (non-zero curriculum weight)
        # This avoids issues where inactive objectives (e.g., SFT during warmup) get
        # router weight assigned, causing instability when they're padded with 0 loss.
        if self.config.ldc_mtl.enabled:
            # Store device for later reinitialization on curriculum phase changes
            self._ldc_mtl_device = device
            # Get only active objectives for current curriculum phase
            active_objectives = self._get_active_objectives(pl_module)
            all_objectives = self._get_objective_names(pl_module)

            if active_objectives and len(active_objectives) > 1:
                self.ldc_mtl = LDCMTLManager(
                    active_objectives,
                    self.config.ldc_mtl,
                    device,
                )
                self._current_active_objectives = active_objectives
                logger.info(
                    f"MetaOptimizerCallback: LDC-MTL enabled for ACTIVE objectives: "
                    f"{active_objectives} (all objectives: {all_objectives})"
                )
            else:
                logger.info(
                    f"MetaOptimizerCallback: LDC-MTL disabled for current phase "
                    f"(need >1 active objective, have: {active_objectives})"
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

        # LayerLR initialization is DEFERRED to on_train_start
        # This is critical for compatibility with BatchSizeFinder, which runs
        # trial training steps in on_fit_start. If LayerLR is initialized in
        # setup(), its hooks (on_after_backward, on_train_batch_end, etc.) will
        # run during BatchSizeFinder trials, causing hangs due to:
        # 1. penalty.backward() creating computation graphs during memory probing
        # 2. LR modifications interfering with trial step comparisons
        #
        # Similar to MuonClipInitCallback, we defer to on_train_start which runs
        # AFTER BatchSizeFinder completes.
        if self.config.layer_lr.enabled:
            self._layer_lr_pending_init = True
            logger.info(
                "MetaOptimizerCallback: LayerLR init deferred to on_train_start "
                "(for BatchSizeFinder compatibility)"
            )

        self._is_enabled = (
            self.ldc_mtl is not None
            or self.odm is not None
            or self._layer_lr_pending_init  # Will be initialized in on_train_start
        )

        if self._is_enabled:
            logger.info("MetaOptimizerCallback: enabled and ready")
            # Attach to pl_module so training_step can access LDC-MTL
            pl_module._meta_optimizer_callback = self

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

    def _get_active_objectives(self, pl_module: pl.LightningModule) -> list[str]:
        """Get currently active objectives (non-zero curriculum weight).

        Returns only objectives with non-zero curriculum weight in the current
        phase. This is used to initialize LDC-MTL with only relevant objectives,
        avoiding issues where inactive objectives (e.g., SFT during warmup) get
        router weight assigned to them.

        Args:
            pl_module: The Lightning module being trained.

        Returns:
            List of active objective names, sorted for consistent ordering.
        """
        if not hasattr(pl_module, "objective_manager"):
            return []

        manager = pl_module.objective_manager
        if not hasattr(manager, "get_current_weights"):
            # Fallback to all objectives if curriculum weights not available
            return self._get_objective_names(pl_module)

        weights = manager.get_current_weights()
        active = [name for name, weight in weights.items() if weight > 0]
        return sorted(active)  # Sorted for consistent ordering

    def _reinit_ldc_mtl(
        self,
        pl_module: pl.LightningModule,
        new_objectives: list[str],
    ) -> None:
        """Reinitialize LDC-MTL manager with new set of active objectives.

        Called when curriculum phase changes and the set of active objectives
        changes. This ensures the router network dimensions match the number of
        objectives being optimized, preventing issues where inactive objectives
        get weight assigned.

        Args:
            pl_module: The Lightning module being trained.
            new_objectives: New list of active objective names.
        """
        if len(new_objectives) < 2:
            # Single objective - disable LDC-MTL
            logger.info(
                f"LDC-MTL: Disabling for single objective phase "
                f"(only {new_objectives} active)"
            )
            self.ldc_mtl = None
            self._current_active_objectives = new_objectives
            return

        # Get device (use cached or get from model)
        if self._ldc_mtl_device is None:
            self._ldc_mtl_device = next(pl_module.model.parameters()).device

        # Create new LDC-MTL manager
        self.ldc_mtl = LDCMTLManager(
            new_objectives,
            self.config.ldc_mtl,
            self._ldc_mtl_device,
        )
        self._current_active_objectives = new_objectives
        logger.info(
            f"LDC-MTL: Reinitialized for curriculum phase change "
            f"(now optimizing {new_objectives})"
        )

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Initialize LayerLR after BatchSizeFinder completes.

        LayerLR init is deferred to on_train_start (from setup) to avoid
        interfering with BatchSizeFinder's trial training steps. This is
        the same pattern used by MuonClipInitCallback.
        """
        if not self._layer_lr_pending_init:
            return

        # Get device from model (now on GPU after BatchSizeFinder)
        device = next(pl_module.model.parameters()).device

        try:
            self.layer_lr = LayerLRManager(
                pl_module.model,
                self.config.layer_lr,
                device,
            )
            # Compute warmup steps for LayerLR
            self._warmup_steps = int(
                self._total_steps * self.config.layer_lr.warmup_ratio
            )
            logger.info(
                f"MetaOptimizerCallback: LayerLR initialized for "
                f"{self.layer_lr.num_layers} layers, "
                f"warmup={self._warmup_steps} steps "
                "(deferred init after BatchSizeFinder)"
            )
        except ValueError as e:
            logger.info(f"MetaOptimizerCallback: LayerLR disabled ({e})")

        self._layer_lr_pending_init = False

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Collect gradient norms after backward pass (for LayerLR)."""
        if self.layer_lr is not None:
            self.layer_lr.collect_grad_norms(pl_module.model)

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Any,
    ) -> None:
        """Apply LR multipliers before optimizer step (for LayerLR)."""
        if self.layer_lr is not None:
            self.layer_lr.apply_multipliers(
                optimizer,
                trainer.global_step,
                self._warmup_steps,
            )

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
                # MixedDataset exposes names via get_current_weights()
                if hasattr(mixed, "get_current_weights"):
                    return list(mixed.get_current_weights().keys())

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
        Also checks for curriculum phase changes and reinitializes LDC-MTL if the set
        of active objectives changed.
        """
        if not self._is_enabled:
            return

        step = trainer.global_step

        # Check for curriculum phase changes - reinitialize LDC-MTL if needed
        # This ensures the router dimensions match the current active objectives
        if self.config.ldc_mtl.enabled:
            current_active = self._get_active_objectives(pl_module)
            if (
                self._current_active_objectives is not None
                and set(current_active) != set(self._current_active_objectives)
            ):
                logger.info(
                    f"Curriculum phase change detected: objectives changed from "
                    f"{self._current_active_objectives} to {current_active}"
                )
                self._reinit_ldc_mtl(pl_module, current_active)

        # Update ODM with per-domain losses (if available)
        if self.odm is not None:
            self._update_odm(trainer, outputs, batch, step)

        # LDC-MTL updates happen during forward pass (gradients flow through router)
        # We step the optimizer based on step_interval config
        if self.ldc_mtl is not None:
            if step % self.config.ldc_mtl.step_interval == 0:
                self.ldc_mtl.step()

        # LayerLR: step multiplier optimizer and restore base LRs
        if self.layer_lr is not None:
            if step % self.config.layer_lr.step_interval == 0:
                self.layer_lr.step()
            # Restore base LRs for scheduler to work correctly
            optimizer = trainer.optimizers[0]
            if hasattr(optimizer, "_optimizer"):
                optimizer = optimizer._optimizer
            self.layer_lr.restore_lrs(optimizer)

        # Log periodically
        if step % self.config.log_interval == 0:
            self._log_metrics(pl_module, step)
            self._log_batch_dataset_balance(pl_module, batch)

    def _update_odm(
        self,
        trainer: pl.Trainer,
        outputs: Any,
        batch: dict[str, Any],
        step: int,
    ) -> None:
        """Update ODM dataset weights based on training outputs.

        Extracts per-domain losses from outputs and updates the bandit.
        Uses batch["domain"] for ODM paper-style homogeneous batch rewards.
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

        # ODM paper style: use batch["domain"] with overall batch loss
        if losses_per_domain is None and "domain" in batch:
            domain = batch["domain"]
            # Handle batched domain field (take first if it's a list)
            if isinstance(domain, (list, tuple)):
                domain = domain[0] if domain else None
            elif hasattr(domain, "__getitem__") and hasattr(domain, "__len__"):
                # Tensor or array-like
                domain = domain[0].item() if hasattr(domain[0], "item") else domain[0]

            if domain is not None:
                loss = outputs.get("loss", 0.0) if isinstance(outputs, dict) else outputs
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
        # LDC-MTL metrics (weights are already logged in training_step)
        if self.ldc_mtl is not None:
            # Log loss discrepancy for debugging
            pass  # Weights logged in training_step for every step

        # ODM metrics
        if self.odm is not None:
            metrics = self.odm.get_wandb_metrics()
            for name, value in metrics.items():
                pl_module.log(name, value, prog_bar=False)

        # LayerLR metrics
        if self.layer_lr is not None:
            metrics = self.layer_lr.get_wandb_metrics()
            for name, value in metrics.items():
                pl_module.log(name, value, prog_bar=False)

    def _log_batch_dataset_balance(
        self,
        pl_module: pl.LightningModule,
        batch: dict[str, Any],
    ) -> None:
        """Log which datasets contributed to the current batch.

        If batch contains 'source' or 'domain' metadata, logs the distribution.
        This helps visualize actual dataset balance during training.
        """
        # Try different possible field names for dataset source
        source_field = None
        for field in ["source", "domain", "dataset", "source_name"]:
            if field in batch:
                source_field = batch[field]
                break

        if source_field is None:
            return

        # Count occurrences if it's a list/tensor
        if isinstance(source_field, (list, tuple)):
            from collections import Counter
            counts = Counter(source_field)
            total = len(source_field)
            for name, count in counts.items():
                pl_module.log(
                    f"data/batch_fraction_{name}",
                    count / total,
                    prog_bar=False,
                )
        elif isinstance(source_field, str):
            # Single source for whole batch
            pl_module.log(f"data/batch_source_{source_field}", 1.0, prog_bar=False)

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

        if self.layer_lr is not None:
            state["layer_lr"] = self.layer_lr.state_dict()

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

        if self.layer_lr is not None and "layer_lr" in state:
            self.layer_lr.load_state_dict(state["layer_lr"])

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

        if self.layer_lr is not None:
            result["layer_lr"] = self.layer_lr._current_multipliers

        return result
