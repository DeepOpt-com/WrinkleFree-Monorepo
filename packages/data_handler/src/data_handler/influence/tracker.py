"""InfluenceTracker callback for influence-based dataset remixing.

This module provides a training callback that integrates with the existing
influence calculation system (DataInfCalculator, MixtureWeightCalculator)
to dynamically update dataset mixture weights during training.

Usage:
    from data_handler.influence import InfluenceTracker

    # Create tracker (no-ops if influence.enabled=false)
    tracker = InfluenceTracker(
        config=cfg,
        model=model,
        mixed_dataset=mixed_dataset,
        probe_dataloaders=probe_loaders,
    )

    # Training loop integration
    tracker.on_train_begin()
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)
        tracker.on_step_end(step, loss)
    tracker.on_epoch_end(epoch)
    tracker.on_train_end()
"""

import logging
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data_handler.data.mixing import MixedDataset
from data_handler.influence.config import (
    InfluenceConfig,
    MixtureOptimizationConfig,
    InfluenceDistillationConfig,
    JVPEmbeddingConfig,
    LandmarkConfig,
)
from data_handler.influence.mixture_calculator import MixtureWeightCalculator

logger = logging.getLogger(__name__)


class InfluenceTracker:
    """Callback for influence-based dataset remixing during training.

    Integrates with MixtureWeightCalculator to:
    1. Cache probe gradients at training start
    2. Periodically compute influence scores
    3. Update dataset weights based on influence
    4. Log weight changes to wandb

    Self-disables if config.influence.enabled=false - all methods become no-ops.

    Example config:
        ```yaml
        influence:
          enabled: true
          method: datainf
          update_interval: 10000  # Steps between weight updates
          learning_rate: 0.1  # How fast to move toward new weights
          warmup_steps: 1000  # Steps before first weight update
        ```
    """

    def __init__(
        self,
        config: DictConfig | dict,
        model: nn.Module,
        mixed_dataset: MixedDataset | None,
        probe_dataloaders: dict[str, DataLoader] | DataLoader | None = None,
    ):
        """Initialize influence tracker.

        Args:
            config: Training configuration with 'influence' section
            model: Model for gradient computation
            mixed_dataset: MixedDataset to update weights on (None disables tracking)
            probe_dataloaders: Probe DataLoaders for influence calculation
                - dict[str, DataLoader]: Multi-domain mode
                - DataLoader: Single probe mode
                - None: Disables tracking
        """
        # Convert config if needed
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        # Extract influence config
        influence_cfg = config.get("influence", {})
        self.enabled = influence_cfg.get("enabled", False) and mixed_dataset is not None

        if not self.enabled:
            logger.info("InfluenceTracker: disabled (influence.enabled=false or no mixed_dataset)")
            return

        self.model = model
        self.mixed_dataset = mixed_dataset
        self.probe_dataloaders = probe_dataloaders

        # Configuration
        self.update_interval = influence_cfg.get("update_interval", 10000)
        self.learning_rate = influence_cfg.get("learning_rate", 0.1)
        self.warmup_steps = influence_cfg.get("warmup_steps", 1000)
        self.method = influence_cfg.get("method", "datainf")
        self.refresh_probe_cache_on_epoch = influence_cfg.get("refresh_probe_cache", False)

        # Initialize influence config
        influence_config = InfluenceConfig(
            lambda_reg=influence_cfg.get("lambda_reg", 1e-4),
        )

        mixture_config = MixtureOptimizationConfig(
            samples_per_dataset=influence_cfg.get("samples_per_dataset", 1000),
            influence_smoothing=influence_cfg.get("smoothing", 0.1),
        )

        # Determine probe mode and method
        if self.method == "distillation":
            # Use InfluenceDistillation for landmark-based approximation
            self._setup_distillation_calculator(
                influence_cfg, probe_dataloaders
            )
        else:
            # Default: Use DataInf/MixtureWeightCalculator
            self._setup_datainf_calculator(
                influence_config, mixture_config, probe_dataloaders
            )

    def _setup_datainf_calculator(
        self,
        influence_config: InfluenceConfig,
        mixture_config: MixtureOptimizationConfig,
        probe_dataloaders: dict[str, DataLoader] | DataLoader | None,
    ):
        """Set up the DataInf-based calculator."""
        if isinstance(probe_dataloaders, dict):
            # Multi-domain mode
            self.calculator = MixtureWeightCalculator(
                model=self.model,
                domain_probe_loaders=probe_dataloaders,
                config=mixture_config,
                influence_config=influence_config,
            )
            self.multi_domain = True
            logger.info(f"InfluenceTracker: DataInf multi-domain mode with {len(probe_dataloaders)} domains")
        elif isinstance(probe_dataloaders, DataLoader):
            # Single probe mode
            self.calculator = MixtureWeightCalculator(
                model=self.model,
                probe_dataloader=probe_dataloaders,
                config=mixture_config,
                influence_config=influence_config,
            )
            self.multi_domain = False
            logger.info("InfluenceTracker: DataInf single probe mode")
        else:
            raise ValueError("probe_dataloaders must be dict[str, DataLoader] or DataLoader")

    def _setup_distillation_calculator(
        self,
        influence_cfg: dict,
        probe_dataloaders: dict[str, DataLoader] | DataLoader | None,
    ):
        """Set up the InfluenceDistillation-based calculator."""
        from data_handler.influence.distillation import InfluenceDistillation

        # Build config from YAML
        distill_config = InfluenceDistillationConfig(
            jvp=JVPEmbeddingConfig(
                num_jvp_layers=influence_cfg.get("jvp_layers", 4),
                num_tangent_vectors=influence_cfg.get("jvp_vectors", 2),
                projection_dim=influence_cfg.get("projection_dim", 131072),
            ),
            landmark=LandmarkConfig(
                num_landmarks=influence_cfg.get("num_landmarks", 4096),
                selection_strategy=influence_cfg.get("landmark_strategy", "kmeans_pp"),
            ),
            samples_per_dataset=influence_cfg.get("samples_per_dataset", 1000),
        )

        self.distillation = InfluenceDistillation(self.model, distill_config)
        self.probe_loader = (
            list(probe_dataloaders.values())[0]
            if isinstance(probe_dataloaders, dict)
            else probe_dataloaders
        )
        self.multi_domain = isinstance(probe_dataloaders, dict)
        self.calculator = None  # Not used in distillation mode

        logger.info(
            f"InfluenceTracker: InfluenceDistillation mode "
            f"(jvp_layers={distill_config.jvp.num_jvp_layers}, "
            f"landmarks={distill_config.landmark.num_landmarks})"
        )

        # Tracking state (for both methods)
        self._initialized = False
        self._weight_history: list[dict[str, float]] = []
        self._last_update_step = 0

        logger.info(
            f"InfluenceTracker: enabled (method={self.method}, "
            f"update_interval={self.update_interval}, "
            f"warmup={self.warmup_steps}, lr={self.learning_rate})"
        )

    def on_train_begin(self):
        """Called once at the beginning of training.

        Caches probe gradients for influence calculation.
        """
        import sys
        print(f"[DEBUG] InfluenceTracker.on_train_begin() called, enabled={self.enabled}", flush=True)
        sys.stdout.flush()

        if not self.enabled:
            print(f"[DEBUG] InfluenceTracker disabled, returning", flush=True)
            return

        logger.info("InfluenceTracker: caching probe gradients...")
        print(f"[DEBUG] Setting model to eval mode...", flush=True)
        sys.stdout.flush()
        self.model.eval()
        print(f"[DEBUG] Model set to eval mode", flush=True)
        sys.stdout.flush()

        try:
            if self.method == "distillation":
                # InfluenceDistillation: full setup for landmark-based KRR
                # 1. Cache probe gradients
                logger.info("InfluenceTracker: step 1 - caching probe gradients...")
                self.distillation.cache_probe_gradients(self.probe_loader, show_progress=True)

                # 2. Set up landmarks using a source loader
                # Use first dataset loader if available, else use probe loader
                source_loader = None
                has_dataset_loaders = hasattr(self, '_dataset_loaders') and self._dataset_loaders
                logger.info(f"InfluenceTracker: _dataset_loaders available: {has_dataset_loaders}")

                if has_dataset_loaders:
                    source_loader = next(iter(self._dataset_loaders.values()))
                    loader_name = next(iter(self._dataset_loaders.keys()))
                    logger.info(f"InfluenceTracker: using source loader '{loader_name}'")
                else:
                    # Fallback to probe loader - but warn that this reuses exhausted iterator
                    logger.warning(
                        "InfluenceTracker: no _dataset_loaders available, falling back to probe_loader. "
                        "This may fail for streaming datasets."
                    )
                    source_loader = self.probe_loader

                # Cache source embeddings and compute landmarks/KRR
                logger.info("InfluenceTracker: step 2 - caching source embeddings...")
                self.distillation.cache_source_embeddings(source_loader, show_progress=True)
                logger.info("InfluenceTracker: step 3 - caching landmarks...")
                self.distillation.cache_landmarks(source_loader, show_progress=True)
            elif self.multi_domain:
                self.calculator.cache_all_domain_probes(show_progress=True)
            else:
                self.calculator.cache_probe_gradients(show_progress=True)

            self._initialized = True
            logger.info("InfluenceTracker: probe gradients cached successfully")
        except Exception as e:
            import traceback
            logger.error(f"InfluenceTracker: failed to cache probe gradients: {e}")
            logger.error(f"InfluenceTracker: traceback:\n{traceback.format_exc()}")
            self.enabled = False
        finally:
            self.model.train()

    def on_step_end(self, step: int, loss: Tensor | None = None):
        """Called after each optimizer step.

        Triggers weight update if interval reached and past warmup.

        Args:
            step: Current global step
            loss: Current loss (optional, for logging)
        """
        if not self.enabled or not self._initialized:
            return

        # Skip during warmup
        if step < self.warmup_steps:
            return

        # Check if it's time to update
        if step > 0 and step % self.update_interval == 0:
            self._update_weights(step)

    def on_epoch_end(self, epoch: int):
        """Called at the end of each epoch.

        Optionally refreshes probe cache with updated model.

        Args:
            epoch: Current epoch number
        """
        if not self.enabled or not self._initialized:
            return

        if self.refresh_probe_cache_on_epoch:
            logger.info(f"InfluenceTracker: refreshing probe cache at epoch {epoch}")
            self.model.eval()
            try:
                self.calculator.refresh_probe_cache(show_progress=False)
            finally:
                self.model.train()

    def on_train_end(self):
        """Called once at the end of training.

        Logs final weight history summary.
        """
        if not self.enabled:
            return

        if self._weight_history:
            logger.info(
                f"InfluenceTracker: training complete. "
                f"Made {len(self._weight_history)} weight updates."
            )

            # Log final weights
            final_weights = self.get_current_weights()
            logger.info(f"InfluenceTracker: final weights: {final_weights}")

    def _update_weights(self, step: int):
        """Compute and apply weight update using InfluenceDistillation.

        For distillation method, uses cached landmark-based KRR to compute
        per-dataset influence scores and update mixture weights.

        Args:
            step: Current global step
        """
        logger.info(f"InfluenceTracker: computing weight update at step {step}")
        self.model.eval()

        try:
            current_weights = self.mixed_dataset.get_current_weights()
            logger.info(f"InfluenceTracker: current weights: {current_weights}")

            # Compute new weights using InfluenceDistillation if available
            if self.method == "distillation" and hasattr(self, 'distillation'):
                if hasattr(self, '_dataset_loaders') and self._dataset_loaders:
                    # Use InfluenceDistillation's compute_mixture_weights
                    new_weights = self.distillation.compute_mixture_weights(
                        self._dataset_loaders, show_progress=False
                    )
                    # Apply with learning rate smoothing
                    smoothed = {
                        k: (1 - self.learning_rate) * current_weights.get(k, 0) + self.learning_rate * v
                        for k, v in new_weights.items()
                    }
                    self.mixed_dataset.update_weights_from_influence(smoothed)
                    current_weights = smoothed
                    logger.info(f"InfluenceTracker: updated weights via distillation: {current_weights}")
                else:
                    logger.warning("InfluenceTracker: distillation mode but no _dataset_loaders set")

            # Store in history
            self._weight_history.append({
                "step": step,
                **current_weights,
            })

            # Log to wandb if available
            self._log_weights_to_wandb(step, current_weights)

            self._last_update_step = step

        except Exception as e:
            logger.error(f"InfluenceTracker: weight update failed: {e}")
        finally:
            self.model.train()

    def update_weights_manual(
        self,
        dataset_loaders: dict[str, DataLoader],
        step: int,
    ):
        """Manually trigger weight update with explicit dataset loaders.

        Use this when you have access to the individual dataset loaders
        for computing per-dataset influence.

        Args:
            dataset_loaders: Dictionary mapping dataset names to DataLoaders
            step: Current global step
        """
        if not self.enabled or not self._initialized:
            return

        logger.info(f"InfluenceTracker: manual weight update at step {step}")
        self.model.eval()

        try:
            current_weights = self.mixed_dataset.get_current_weights()

            # Compute optimal weights
            new_weights = self.calculator.get_weight_update(
                current_weights=current_weights,
                dataset_loaders=dataset_loaders,
                learning_rate=self.learning_rate,
            )

            # Apply to mixed dataset
            self.mixed_dataset.update_weights_from_influence(new_weights)

            logger.info(f"InfluenceTracker: updated weights: {new_weights}")

            # Store in history
            self._weight_history.append({
                "step": step,
                **new_weights,
            })

            self._log_weights_to_wandb(step, new_weights)

        except Exception as e:
            logger.error(f"InfluenceTracker: manual weight update failed: {e}")
        finally:
            self.model.train()

    def _log_weights_to_wandb(self, step: int, weights: dict[str, float]):
        """Log weights and relative ratios to wandb if available.

        Logs both absolute weights and relative ratios (each weight / max weight)
        for easier visualization of dataset balance.
        """
        try:
            import wandb
            if wandb.run is None:
                return

            metrics = {}

            # Absolute weights
            for k, v in weights.items():
                metrics[f"influence/weight_{k}"] = v

            # Relative ratios - each dataset relative to max weight
            if weights:
                max_weight = max(weights.values())
                if max_weight > 0:
                    for name, weight in weights.items():
                        metrics[f"influence/relative_{name}"] = weight / max_weight

            wandb.log(metrics, step=step)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"InfluenceTracker: wandb logging failed: {e}")

    def get_current_weights(self) -> dict[str, float]:
        """Get current mixture weights.

        Returns:
            Dictionary mapping dataset names to current weights.
            Empty dict if tracking is disabled.
        """
        if not self.enabled or self.mixed_dataset is None:
            return {}
        return self.mixed_dataset.get_current_weights()

    def get_weight_history(self) -> list[dict[str, Any]]:
        """Get history of weight updates.

        Returns:
            List of weight update records with step numbers.
        """
        return self._weight_history.copy()

    @property
    def is_enabled(self) -> bool:
        """Check if influence tracking is enabled."""
        return self.enabled

    @property
    def is_initialized(self) -> bool:
        """Check if probe gradients have been cached."""
        return self._initialized


def create_influence_tracker(
    config: DictConfig | dict,
    model: nn.Module,
    mixed_dataset: MixedDataset | None,
    probe_dataloaders: dict[str, DataLoader] | DataLoader | None = None,
) -> InfluenceTracker:
    """Factory function to create an InfluenceTracker.

    Args:
        config: Training configuration
        model: Model for gradient computation
        mixed_dataset: MixedDataset to track
        probe_dataloaders: Probe DataLoaders

    Returns:
        Configured InfluenceTracker
    """
    return InfluenceTracker(
        config=config,
        model=model,
        mixed_dataset=mixed_dataset,
        probe_dataloaders=probe_dataloaders,
    )
