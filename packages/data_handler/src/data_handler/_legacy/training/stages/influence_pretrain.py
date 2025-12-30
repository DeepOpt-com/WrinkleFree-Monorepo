"""Influence-based pretraining stage.

Extends PretrainStage with dynamic mixture weight adjustment
based on influence function calculations.

Reference: MobileLLM-R1 paper (arXiv:2509.24945) Phase II
"""

from typing import Any, Iterator, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from data_handler._legacy.training.stages.base import StageConfig, TrainingMetrics
from data_handler._legacy.training.stages.pretrain import PretrainStage


class InfluencePretrainStage(PretrainStage):
    """Pretraining stage with dynamic influence-based mixture adjustment.

    This stage periodically recomputes the optimal mixture weights
    based on influence function analysis and updates the dataset
    sampling probabilities accordingly.

    Features:
    - Automatic mixture weight optimization during training
    - Logging of weight changes and influence scores
    - Support for both Phase II (mixture) and Phase III (self-boosting)
    """

    def __init__(
        self,
        config: StageConfig,
        model: Any,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        dataloader: DataLoader,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        influence_filter: Optional[Any] = None,
        mixture_calculator: Optional[Any] = None,
        mixed_dataset: Optional[Any] = None,
        weight_update_interval: int = 10000,
    ):
        """Initialize influence-based pretraining stage.

        Args:
            config: Stage configuration
            model: Model to train
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            dataloader: Training data loader
            device: Device to train on
            rank: Process rank for distributed training
            world_size: Total number of processes
            influence_filter: Optional SelfBoostingFilter for Phase III
            mixture_calculator: Optional MixtureWeightCalculator for Phase II
            mixed_dataset: MixedDataset instance for weight updates
            weight_update_interval: Steps between weight updates
        """
        super().__init__(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
            device=device,
            rank=rank,
            world_size=world_size,
            influence_filter=influence_filter,
        )

        self.mixture_calculator = mixture_calculator
        self.mixed_dataset = mixed_dataset
        self.weight_update_interval = weight_update_interval

        self._last_weight_update_step = 0
        self._weight_history: list = []

    def _maybe_update_weights(self) -> Optional[dict]:
        """Update mixture weights if interval has elapsed.

        Returns:
            New weights dictionary if updated, None otherwise
        """
        if self.mixture_calculator is None or self.mixed_dataset is None:
            return None

        if self.global_step - self._last_weight_update_step < self.weight_update_interval:
            return None

        # Get current weights for logging
        old_weights = self.mixed_dataset.get_current_weights()

        # Compute new optimal weights
        # Note: This is expensive, so we do it infrequently
        if self.rank == 0:
            print(f"\nStep {self.global_step}: Recomputing mixture weights...")

        # Refresh probe cache with current model state
        self.mixture_calculator.refresh_probe_cache(show_progress=(self.rank == 0))

        # Get dataset loaders for influence calculation
        # This requires access to the original dataset sources
        # For simplicity, we use incremental updates based on current weights
        new_weights = self.mixture_calculator.get_weight_update(
            old_weights,
            self._get_dataset_loaders_for_mixture(),
            learning_rate=0.2,  # Move 20% toward optimal
        )

        # Update dataset
        self.mixed_dataset.update_weights_from_influence(new_weights)
        self._last_weight_update_step = self.global_step

        # Log changes
        if self.rank == 0:
            print(f"Updated mixture weights:")
            for name in new_weights:
                old = old_weights.get(name, 0)
                new = new_weights.get(name, 0)
                change = new - old
                print(f"  {name}: {old:.4f} -> {new:.4f} ({change:+.4f})")

        # Track history
        self._weight_history.append({
            "step": self.global_step,
            "weights": new_weights.copy(),
        })

        return new_weights

    def _get_dataset_loaders_for_mixture(self) -> dict:
        """Get dataset loaders for mixture calculation.

        This is a simplified implementation that returns empty dict.
        In a full implementation, you would maintain separate loaders
        for each dataset in the mixture.

        Returns:
            Dictionary mapping dataset names to DataLoaders
        """
        # Simplified: return empty dict, mixture calculator will use cached estimates
        return {}

    def train_step(self, batch: dict[str, Tensor]) -> Optional[TrainingMetrics]:
        """Execute training step with optional weight updates.

        Args:
            batch: Input batch dictionary

        Returns:
            TrainingMetrics if step was executed, else None
        """
        # Check for weight updates before training step
        new_weights = self._maybe_update_weights()

        # Run standard training step
        metrics = super().train_step(batch)

        # Add weight info to metrics if updated
        if metrics is not None and new_weights is not None:
            metrics.extra["mixture_weights_updated"] = True
            metrics.extra["mixture_weights"] = new_weights

        # Add influence filter stats if available
        if metrics is not None and self.influence_filter is not None:
            rejection_rate = self.influence_filter.get_rejection_rate()
            metrics.extra["influence_rejection_rate"] = rejection_rate

        return metrics

    def run(
        self,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        log_interval: int = 100,
        checkpoint_interval: int = 1000,
        checkpoint_callback: Optional[callable] = None,
        progress_callback: Optional[callable] = None,
    ) -> Iterator[TrainingMetrics]:
        """Run training with influence-based optimizations.

        Args:
            max_steps: Maximum steps (overrides config)
            max_epochs: Maximum epochs (overrides config)
            log_interval: How often to yield metrics
            checkpoint_interval: How often to call checkpoint callback
            checkpoint_callback: Function to call for checkpointing
            progress_callback: Function to call with progress updates

        Yields:
            TrainingMetrics for each logged step
        """
        # Initialize influence components if available
        if self.mixture_calculator is not None:
            if self.rank == 0:
                print("Initializing mixture calculator probe cache...")
            self.mixture_calculator.cache_probe_gradients(show_progress=(self.rank == 0))

        if self.influence_filter is not None:
            if self.rank == 0:
                print("Initializing influence filter probe cache...")
            self.influence_filter.initialize_probe_set(show_progress=(self.rank == 0))

        # Run standard training loop
        yield from super().run(
            max_steps=max_steps,
            max_epochs=max_epochs,
            log_interval=log_interval,
            checkpoint_interval=checkpoint_interval,
            checkpoint_callback=checkpoint_callback,
            progress_callback=progress_callback,
        )

        # Log final statistics
        if self.rank == 0:
            self._log_final_stats()

    def _log_final_stats(self):
        """Log final training statistics."""
        print("\n" + "=" * 50)
        print("Influence-Based Pretraining Statistics")
        print("=" * 50)

        if self._weight_history:
            print(f"\nWeight updates: {len(self._weight_history)}")
            print("\nFinal mixture weights:")
            final_weights = self._weight_history[-1]["weights"]
            for name, weight in sorted(final_weights.items(), key=lambda x: -x[1]):
                print(f"  {name}: {weight:.4f}")

        if self.influence_filter is not None:
            rejection_rate = self.influence_filter.get_rejection_rate()
            print(f"\nInfluence filter rejection rate: {rejection_rate:.2%}")

        print("=" * 50 + "\n")

    def state_dict(self) -> dict:
        """Get state dictionary including influence state.

        Returns:
            State dictionary for checkpointing
        """
        state = super().state_dict()
        state["influence_state"] = {
            "last_weight_update_step": self._last_weight_update_step,
            "weight_history": self._weight_history,
        }
        return state

    def load_state_dict(self, state_dict: dict):
        """Load state including influence state.

        Args:
            state_dict: Checkpoint dictionary
        """
        super().load_state_dict(state_dict)
        if "influence_state" in state_dict:
            influence_state = state_dict["influence_state"]
            self._last_weight_update_step = influence_state.get("last_weight_update_step", 0)
            self._weight_history = influence_state.get("weight_history", [])
