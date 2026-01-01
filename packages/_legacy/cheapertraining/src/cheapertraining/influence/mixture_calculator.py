"""Mixture weight calculator for Phase II pre-training optimization.

Implements influence-based dataset weight calculation.
Uses the current model (simplified from domain specialists) to determine
how much each dataset contributes to probe set performance.

Supports multi-domain probes (Code, Math, Knowledge) per MobileLLM-R1 methodology.

Reference: MobileLLM-R1 paper (arXiv:2509.24945) Phase II
"""

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from cheapertraining.influence.config import InfluenceConfig, MixtureOptimizationConfig
from cheapertraining.influence.datainf import DataInfCalculator, create_influence_calculator


# Default domain weights per MobileLLM-R1 (equal weighting)
DEFAULT_DOMAIN_WEIGHTS = {
    "code": 0.33,
    "math": 0.33,
    "knowledge": 0.34,
}


class MixtureWeightCalculator:
    """Calculates optimal pre-training mixture weights using influence functions.

    This is a simplified version of MobileLLM-R1 Phase II that uses the
    current model for all influence calculations instead of domain specialists.

    The key idea: datasets whose samples have higher positive influence on
    the probe set should receive higher sampling weights.

    Usage:
        calculator = MixtureWeightCalculator(model, probe_dataloader)
        weights = calculator.compute_mixture_weights(dataset_loaders)
    """

    def __init__(
        self,
        model: Any,
        probe_dataloader: Optional[DataLoader] = None,
        config: Optional[MixtureOptimizationConfig] = None,
        influence_config: Optional[InfluenceConfig] = None,
        domain_probe_loaders: Optional[Dict[str, DataLoader]] = None,
        domain_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize mixture weight calculator.

        Args:
            model: MobileLLM model for influence calculations
            probe_dataloader: DataLoader for the probe set (single-domain mode)
            config: Mixture optimization configuration
            influence_config: Influence calculation configuration
            domain_probe_loaders: Dict of domain -> DataLoader (multi-domain mode)
            domain_weights: Dict of domain -> weight for joint influence
        """
        self.model = model
        self.probe_dataloader = probe_dataloader
        self.config = config or MixtureOptimizationConfig()
        self.influence_config = influence_config or InfluenceConfig()

        # Multi-domain probe support (MobileLLM-R1 style)
        self.domain_probe_loaders = domain_probe_loaders or {}
        self.domain_weights = domain_weights or DEFAULT_DOMAIN_WEIGHTS
        self.multi_domain_mode = bool(domain_probe_loaders)

        # Create influence calculator
        self.influence_calculator = create_influence_calculator(
            model, self.influence_config
        )

        # Cache probe gradients
        self._probe_cached = False
        self._domain_probe_cached: Dict[str, bool] = {}

        # Running influence estimates (for EMA smoothing)
        self._influence_estimates: Dict[str, float] = {}
        self._domain_influence_estimates: Dict[str, Dict[str, float]] = {}

    def cache_probe_gradients(self, show_progress: bool = True):
        """Cache probe set gradients for influence calculation.

        This should be called once before computing mixture weights,
        and can be re-called when the model is updated.
        """
        self.influence_calculator.cache_probe_gradients(
            self.probe_dataloader,
            show_progress=show_progress,
        )
        self._probe_cached = True

    def refresh_probe_cache(self, show_progress: bool = False):
        """Refresh probe cache with current model state.

        Call this periodically during training to keep influence
        estimates aligned with model capabilities.
        """
        self.influence_calculator.clear_cache()
        if self.multi_domain_mode:
            self.cache_all_domain_probes(show_progress=show_progress)
        else:
            self.cache_probe_gradients(show_progress=show_progress)

    def cache_domain_probe_gradients(
        self,
        domain: str,
        show_progress: bool = True,
    ):
        """Cache probe gradients for a specific domain.

        Args:
            domain: Domain name (e.g., 'code', 'math', 'knowledge')
            show_progress: Whether to show progress bar
        """
        if domain not in self.domain_probe_loaders:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(self.domain_probe_loaders.keys())}")

        loader = self.domain_probe_loaders[domain]
        self.influence_calculator.cache_probe_gradients(
            loader,
            show_progress=show_progress,
        )
        self._domain_probe_cached[domain] = True

    def cache_all_domain_probes(self, show_progress: bool = True):
        """Cache probe gradients for all domains.

        Args:
            show_progress: Whether to show progress bar
        """
        for domain in self.domain_probe_loaders:
            print(f"Caching probe gradients for domain: {domain}")
            self.cache_domain_probe_gradients(domain, show_progress=show_progress)

    def compute_domain_influence(
        self,
        domain: str,
        dataset_loader: DataLoader,
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> float:
        """Calculate influence of a dataset on a specific domain probe.

        Args:
            domain: Domain name
            dataset_loader: DataLoader for the dataset
            max_samples: Maximum samples to evaluate
            show_progress: Whether to show progress bar

        Returns:
            Average influence score for this domain
        """
        if not self._domain_probe_cached.get(domain, False):
            self.cache_domain_probe_gradients(domain, show_progress=False)

        return self.compute_dataset_influence(
            dataset_loader,
            max_samples=max_samples,
            show_progress=show_progress,
        )

    def compute_joint_influence(
        self,
        dataset_loader: DataLoader,
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> float:
        """Calculate joint influence across all domains.

        Combines per-domain influences using domain weights.

        Args:
            dataset_loader: DataLoader for the dataset
            max_samples: Maximum samples to evaluate
            show_progress: Whether to show progress bar

        Returns:
            Weighted average influence across domains
        """
        domain_influences = {}

        for domain in self.domain_probe_loaders:
            # Cache this domain's probe if needed
            if not self._domain_probe_cached.get(domain, False):
                self.cache_domain_probe_gradients(domain, show_progress=False)

            # Compute influence for this domain
            influence = self.compute_domain_influence(
                domain,
                dataset_loader,
                max_samples=max_samples,
                show_progress=show_progress,
            )
            domain_influences[domain] = influence

            if show_progress:
                print(f"    {domain}: {influence:.4f}")

        # Compute weighted average
        total_weight = sum(self.domain_weights.get(d, 1.0) for d in domain_influences)
        joint_influence = sum(
            self.domain_weights.get(d, 1.0) * inf
            for d, inf in domain_influences.items()
        ) / max(total_weight, 1e-6)

        return joint_influence

    def compute_dataset_influence(
        self,
        dataset_loader: DataLoader,
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> float:
        """Calculate average influence of a dataset on the probe set.

        Args:
            dataset_loader: DataLoader for the dataset
            max_samples: Maximum samples to evaluate (default: from config)
            show_progress: Whether to show progress bar

        Returns:
            Average influence score (can be negative)
        """
        if not self._probe_cached:
            self.cache_probe_gradients()

        max_samples = max_samples or self.config.samples_per_dataset
        device = next(self.model.parameters()).device

        total_influence = 0.0
        num_samples = 0

        iterator = tqdm(
            dataset_loader,
            desc="Computing dataset influence",
            disable=not show_progress,
        )

        for batch in iterator:
            if max_samples and num_samples >= max_samples:
                break

            # Move to device
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

            # Compute aggregated influence for batch
            influences = self.influence_calculator.compute_batch_influence_aggregated(batch)
            total_influence += influences.sum().item()
            num_samples += influences.size(0)

            # Update progress bar
            if show_progress:
                avg = total_influence / max(num_samples, 1)
                iterator.set_postfix({"avg_influence": f"{avg:.4f}", "samples": num_samples})

        return total_influence / max(num_samples, 1)

    def compute_mixture_weights(
        self,
        dataset_loaders: Dict[str, DataLoader],
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """Compute optimal mixture weights for all datasets.

        Datasets with higher positive influence receive higher weights.
        In multi-domain mode, uses joint influence across all domains.

        Args:
            dataset_loaders: Dictionary mapping dataset names to DataLoaders
            show_progress: Whether to show progress

        Returns:
            Dictionary of dataset names to normalized weights (sum to 1)
        """
        # Cache appropriate probes
        if self.multi_domain_mode:
            self.cache_all_domain_probes(show_progress=show_progress)
        elif not self._probe_cached:
            self.cache_probe_gradients()

        raw_influences = {}

        for name, loader in dataset_loaders.items():
            print(f"Computing influence for dataset: {name}")

            if self.multi_domain_mode:
                # Use joint influence across domains
                influence = self.compute_joint_influence(
                    loader,
                    show_progress=show_progress,
                )
            else:
                # Single probe mode
                influence = self.compute_dataset_influence(
                    loader,
                    show_progress=show_progress,
                )

            raw_influences[name] = influence
            print(f"  {name}: joint influence = {influence:.4f}")

        # Apply EMA smoothing if we have previous estimates
        if self.config.influence_smoothing > 0 and self._influence_estimates:
            alpha = self.config.influence_smoothing
            for name in raw_influences:
                if name in self._influence_estimates:
                    raw_influences[name] = (
                        alpha * raw_influences[name] +
                        (1 - alpha) * self._influence_estimates[name]
                    )

        # Update running estimates
        self._influence_estimates = raw_influences.copy()

        # Convert to weights
        weights = self._influences_to_weights(raw_influences)

        return weights

    def _influences_to_weights(
        self,
        influences: Dict[str, float],
    ) -> Dict[str, float]:
        """Convert influence scores to normalized weights.

        Higher positive influence -> higher weight.
        Negative influences are clamped to 0.

        Args:
            influences: Dictionary of dataset influences

        Returns:
            Normalized weights (sum to 1)
        """
        # Clamp negative influences to small positive value
        # (we still want some samples from every dataset)
        clamped = {}
        for name, inf in influences.items():
            clamped[name] = max(inf, 0.0) + 1e-6  # Add epsilon to avoid division by zero

        if self.config.normalize_weights:
            # Normalize to sum to 1
            total = sum(clamped.values())
            if total > 0:
                weights = {k: v / total for k, v in clamped.items()}
            else:
                # Fallback to uniform
                n = len(clamped)
                weights = {k: 1.0 / n for k in clamped}

            # Apply min/max constraints
            for k in weights:
                weights[k] = max(weights[k], self.config.min_weight)
                weights[k] = min(weights[k], self.config.max_weight)

            # Re-normalize after constraints
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = clamped

        return weights

    def get_weight_update(
        self,
        current_weights: Dict[str, float],
        dataset_loaders: Dict[str, DataLoader],
        learning_rate: float = 0.1,
    ) -> Dict[str, float]:
        """Get incremental weight update for online learning.

        Instead of fully recomputing weights, this method provides
        an incremental update that can be applied during training.

        Args:
            current_weights: Current mixture weights
            dataset_loaders: Dataset loaders
            learning_rate: How much to move toward new weights

        Returns:
            Updated weights (interpolated between current and optimal)
        """
        optimal_weights = self.compute_mixture_weights(
            dataset_loaders,
            show_progress=False,
        )

        # Interpolate
        updated = {}
        for name in current_weights:
            if name in optimal_weights:
                updated[name] = (
                    (1 - learning_rate) * current_weights[name] +
                    learning_rate * optimal_weights[name]
                )
            else:
                updated[name] = current_weights[name]

        # Normalize
        total = sum(updated.values())
        if total > 0:
            updated = {k: v / total for k, v in updated.items()}

        return updated


def create_mixture_calculator(
    model: Any,
    probe_dataloader: DataLoader,
    config: Optional[MixtureOptimizationConfig] = None,
) -> MixtureWeightCalculator:
    """Factory function to create a mixture weight calculator.

    Args:
        model: MobileLLM model
        probe_dataloader: Probe set DataLoader
        config: Configuration

    Returns:
        Configured MixtureWeightCalculator
    """
    return MixtureWeightCalculator(model, probe_dataloader, config)
