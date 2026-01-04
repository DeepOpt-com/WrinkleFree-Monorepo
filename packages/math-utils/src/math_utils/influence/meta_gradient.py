"""Meta-gradient computation for outer-loop optimization.

Computes gradients of validation loss w.r.t. meta-parameters:
- Dataset mixture weights
- Objective weights
- Learning rate scales

Uses first-order influence approximation (DataInf) to avoid expensive Hessian.

References:
- DataInf (ICLR 2024): https://openreview.net/forum?id=9m02ib92Wz
- AutoMixer (ACL 2025): Discriminative layer selection
- ScaleBiO (2024): https://arxiv.org/abs/2406.19976
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from math_utils.influence.config import InfluenceConfig
from math_utils.influence.gradient import DiscriminativeGradientExtractor

logger = logging.getLogger(__name__)


class MetaGradientCalculator:
    """Computes gradients of meta-parameters w.r.t. validation objectives.

    Uses first-order influence approximations to avoid expensive Hessian.

    Key insight: The influence of training data on validation can be
    computed as the dot product of training gradients with validation gradients,
    scaled by a regularization term (DataInf formula).

    Meta-gradient computation for different parameter types:
    1. Dataset weights: Sum of per-sample influences from each dataset
    2. Objective weights: Alignment between objective gradients and validation gradient
    3. LR scales: Effect of LR changes on validation (via gradient norm)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InfluenceConfig] = None,
    ):
        """Initialize calculator.

        Args:
            model: Model to compute gradients for
            config: Influence configuration
        """
        self.model = model
        self.config = config or InfluenceConfig()
        self.gradient_extractor = DiscriminativeGradientExtractor(model, config)

        # Cache for validation gradients
        self._cached_val_gradients: dict[str, Tensor] = {}

    def compute_validation_gradient(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
    ) -> Tensor:
        """Compute aggregated gradient from validation data.

        Args:
            dataloader: Validation dataloader
            max_samples: Maximum samples to use (None = all)
            show_progress: Show progress bar

        Returns:
            Aggregated gradient tensor [D]
        """
        device = next(self.model.parameters()).device
        all_grads = []
        n_samples = 0

        iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc="Computing validation gradient")

        for batch in iterator:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            # Compute gradient for batch (aggregated)
            grad = self.gradient_extractor.compute_aggregated_gradient(batch)
            all_grads.append(grad.cpu())

            n_samples += batch["input_ids"].shape[0]
            if max_samples is not None and n_samples >= max_samples:
                break

        # Average across batches
        if not all_grads:
            raise ValueError("Empty dataloader")

        return torch.stack(all_grads).mean(dim=0)

    def cache_validation_gradients(
        self,
        validation_loaders: dict[str, DataLoader],
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> None:
        """Pre-compute and cache validation gradients for each objective.

        Args:
            validation_loaders: Dict mapping objective name to validation dataloader
            max_samples: Maximum samples per objective
            show_progress: Show progress bar
        """
        logger.info(f"Caching validation gradients for {len(validation_loaders)} objectives")

        for name, loader in validation_loaders.items():
            logger.info(f"Computing gradient for {name}...")
            self._cached_val_gradients[name] = self.compute_validation_gradient(
                loader,
                max_samples=max_samples,
                show_progress=show_progress,
            )

        logger.info("Validation gradient caching complete")

    def get_cached_validation_gradients(self) -> dict[str, Tensor]:
        """Get cached validation gradients."""
        return self._cached_val_gradients

    def compute_dataset_meta_gradients(
        self,
        dataset_loaders: dict[str, DataLoader],
        validation_gradient: Tensor,
        current_weights: dict[str, float],
        samples_per_source: int = 256,
        show_progress: bool = False,
    ) -> dict[str, float]:
        """Compute gradient of validation loss w.r.t. dataset weights.

        Uses DataInf first-order approximation:
            d(L_val)/d(w_i) ≈ Σ_j influence(sample_j from dataset_i)

        where influence is computed as:
            influence_j = <grad_j, grad_val> / (λ + ||grad_j||²)

        Args:
            dataset_loaders: Dict mapping dataset name to dataloader
            validation_gradient: Aggregated validation gradient [D]
            current_weights: Current dataset weights (for context)
            samples_per_source: Samples to use per dataset
            show_progress: Show progress bar

        Returns:
            Dict mapping dataset name to meta-gradient
        """
        device = next(self.model.parameters()).device
        val_grad = validation_gradient.to(device)
        lambda_reg = self.config.lambda_val

        meta_grads = {}

        for name, loader in dataset_loaders.items():
            influences = []
            n_samples = 0

            iterator = loader
            if show_progress:
                iterator = tqdm(loader, desc=f"Dataset {name}")

            for batch in iterator:
                batch = {
                    k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }

                # Compute gradient for this batch
                grad = self.gradient_extractor.compute_aggregated_gradient(batch)

                # DataInf influence formula
                grad_norm_sq = (grad ** 2).sum()
                alignment = (val_grad * grad).sum()
                influence = alignment / (lambda_reg + grad_norm_sq)

                influences.append(influence.item())

                n_samples += batch["input_ids"].shape[0]
                if n_samples >= samples_per_source:
                    break

            # Average influence for this dataset
            avg_influence = sum(influences) / len(influences) if influences else 0.0

            # Meta-gradient is negative influence (higher influence = should increase weight)
            meta_grads[name] = -avg_influence

        return meta_grads

    def compute_objective_meta_gradients(
        self,
        objective_gradients: dict[str, Tensor],
        validation_gradient: Tensor,
    ) -> dict[str, float]:
        """Compute gradient of validation loss w.r.t. objective weights.

        Uses alignment between objective gradients and validation gradient:
            d(L_val)/d(w_obj) ≈ <objective_grad, val_grad>

        Intuition: If an objective's gradient points in a similar direction
        to the validation gradient, increasing its weight should help.

        Args:
            objective_gradients: Dict mapping objective name to gradient
            validation_gradient: Aggregated validation gradient [D]

        Returns:
            Dict mapping objective name to meta-gradient
        """
        device = validation_gradient.device
        val_grad = validation_gradient

        meta_grads = {}

        for name, obj_grad in objective_gradients.items():
            obj_grad = obj_grad.to(device)

            # Normalize for scale-invariance
            val_norm = val_grad.norm()
            obj_norm = obj_grad.norm()

            if val_norm < 1e-8 or obj_norm < 1e-8:
                # Degenerate case
                meta_grads[name] = 0.0
                continue

            # Cosine similarity
            alignment = (val_grad * obj_grad).sum() / (val_norm * obj_norm)

            # Meta-gradient: negative alignment means increasing weight helps
            # (because we want to descend on validation loss)
            meta_grads[name] = -alignment.item()

        return meta_grads

    def compute_lr_meta_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        validation_gradient: Tensor,
        training_gradient: Optional[Tensor] = None,
    ) -> dict[str, float]:
        """Compute gradient of validation loss w.r.t. LR scales.

        Approximation: Higher LR amplifies gradient updates.
        If training and validation gradients are aligned, higher LR helps.
        If they're misaligned, lower LR is safer.

        Args:
            optimizer: Optimizer with param_groups
            validation_gradient: Aggregated validation gradient [D]
            training_gradient: Optional training gradient (uses param grads if None)

        Returns:
            Dict mapping param group index to meta-gradient
        """
        device = validation_gradient.device
        val_grad = validation_gradient

        meta_grads = {}

        # Handle wrapped optimizers
        actual_optimizer = optimizer
        if hasattr(optimizer, "_optimizer"):
            actual_optimizer = optimizer._optimizer

        for i, pg in enumerate(actual_optimizer.param_groups):
            # Collect gradients from this param group
            pg_grads = []
            for p in pg["params"]:
                if p.grad is not None:
                    pg_grads.append(p.grad.flatten())

            if not pg_grads:
                meta_grads[f"group_{i}"] = 0.0
                continue

            # Aggregate param group gradient
            pg_grad = torch.cat(pg_grads).to(device)

            # Truncate or pad to match validation gradient size
            # (this is a simplification - in practice you'd want to match dimensions properly)
            min_len = min(len(pg_grad), len(val_grad))
            pg_grad = pg_grad[:min_len]
            val_grad_truncated = val_grad[:min_len]

            # Alignment
            alignment = (pg_grad * val_grad_truncated).sum()
            pg_norm = pg_grad.norm()
            val_norm = val_grad_truncated.norm()

            if pg_norm < 1e-8 or val_norm < 1e-8:
                meta_grads[f"group_{i}"] = 0.0
                continue

            normalized_alignment = alignment / (pg_norm * val_norm)

            # Negative alignment: if aligned, we want higher LR
            # So gradient should be negative (to increase LR when aligned)
            meta_grads[f"group_{i}"] = -normalized_alignment.item()

        return meta_grads

    def compute_all_meta_gradients(
        self,
        validation_gradient: Tensor,
        dataset_loaders: Optional[dict[str, DataLoader]] = None,
        objective_gradients: Optional[dict[str, Tensor]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        current_weights: Optional[dict[str, float]] = None,
        samples_per_source: int = 256,
    ) -> dict[str, dict[str, float]]:
        """Compute all meta-gradients in one call.

        Args:
            validation_gradient: Aggregated validation gradient
            dataset_loaders: Dataset loaders (for dataset weight meta-gradients)
            objective_gradients: Objective gradients (for objective weight meta-gradients)
            optimizer: Optimizer (for LR scale meta-gradients)
            current_weights: Current dataset weights
            samples_per_source: Samples per dataset

        Returns:
            Dict with keys "dataset", "objective", "lr" mapping to meta-gradients
        """
        result = {}

        if dataset_loaders is not None:
            result["dataset"] = self.compute_dataset_meta_gradients(
                dataset_loaders=dataset_loaders,
                validation_gradient=validation_gradient,
                current_weights=current_weights or {},
                samples_per_source=samples_per_source,
            )

        if objective_gradients is not None:
            result["objective"] = self.compute_objective_meta_gradients(
                objective_gradients=objective_gradients,
                validation_gradient=validation_gradient,
            )

        if optimizer is not None:
            result["lr"] = self.compute_lr_meta_gradients(
                optimizer=optimizer,
                validation_gradient=validation_gradient,
            )

        return result
