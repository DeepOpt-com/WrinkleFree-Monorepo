"""DataInf algorithm for tractable influence function computation.

Implements efficient influence calculation without Hessian inversion.

Reference:
- AutoMixer (ACL 2025) - Efficient Influence Approximation
- DataInf (Kwon et al.)
"""

from typing import Optional, Tuple, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_handler.influence.config import InfluenceConfig
from data_handler.influence.gradient import DiscriminativeGradientExtractor


class DataInfCalculator:
    """DataInf algorithm for tractable influence function computation.

    Implements the key formula from AutoMixer:
        influence(z_train, z_probe) = <grad_train, grad_probe> / (lambda + ||grad_train||^2)

    This avoids Hessian inversion by using a diagonal approximation.
    The regularization term lambda stabilizes the computation.

    Reference: AutoMixer paper - "Efficient Influence Approximation"
    """

    def __init__(
        self,
        gradient_extractor: DiscriminativeGradientExtractor,
        config: Optional[InfluenceConfig] = None,
    ):
        """Initialize DataInf calculator.

        Args:
            gradient_extractor: Gradient extraction utility
            config: Influence configuration
        """
        self.gradient_extractor = gradient_extractor
        self.config = config or InfluenceConfig()

        # Cache for probe set gradients
        self._probe_gradients: Optional[Tensor] = None
        self._probe_grad_norms_sq: Optional[Tensor] = None
        self._avg_probe_gradient: Optional[Tensor] = None
        self._num_probe_samples: int = 0

    @property
    def is_cached(self) -> bool:
        """Check if probe gradients are cached."""
        return self._probe_gradients is not None

    def cache_probe_gradients(
        self,
        probe_dataloader: DataLoader,
        show_progress: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Pre-compute and cache gradients for the probe set.

        This should be called once before computing influences.
        The cached gradients are used for all subsequent influence calculations.

        Note: Gradients are stored on CPU to save GPU memory. They are moved
        to GPU only when computing influence scores.

        Args:
            probe_dataloader: DataLoader for probe set samples
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (probe_gradients [N, D], probe_grad_norms_sq [N])
        """
        all_grads = []
        device = next(self.gradient_extractor.model.parameters()).device

        iterator = tqdm(probe_dataloader, desc="Caching probe gradients", disable=not show_progress)
        for batch in iterator:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

            # Compute gradients for each sample in batch
            batch_grads = self.gradient_extractor.compute_batch_gradients(batch)
            # Move to CPU immediately to save GPU memory
            all_grads.append(batch_grads.cpu())

        # Concatenate all gradients (on CPU to save GPU memory)
        self._probe_gradients = torch.cat(all_grads, dim=0)  # Now on CPU
        self._probe_grad_norms_sq = torch.sum(self._probe_gradients ** 2, dim=1)
        self._avg_probe_gradient = self._probe_gradients.mean(dim=0)
        self._num_probe_samples = self._probe_gradients.size(0)

        return self._probe_gradients, self._probe_grad_norms_sq

    def clear_cache(self):
        """Clear cached probe gradients."""
        self._probe_gradients = None
        self._probe_grad_norms_sq = None
        self._avg_probe_gradient = None
        self._num_probe_samples = 0

    def compute_influence(
        self,
        train_gradient: Tensor,
    ) -> Tensor:
        """Compute influence of a training sample on all probe samples.

        Following DataInf formula:
            I(z_train, z_probe) = <grad_train, grad_probe> / (lambda + ||grad_train||^2)

        Args:
            train_gradient: Flattened gradient for training sample [D]

        Returns:
            Influence scores on each probe sample [N_probe]
        """
        if self._probe_gradients is None:
            raise RuntimeError(
                "Probe gradients not cached. Call cache_probe_gradients first."
            )

        # Move probe gradients to same device as train_gradient for computation
        # (probe gradients are stored on CPU to save GPU memory)
        device = train_gradient.device
        probe_grads = self._probe_gradients.to(device)

        # Compute gradient norm squared
        train_grad_norm_sq = torch.sum(train_gradient ** 2)

        # Compute dot products with all probe gradients
        # probe_gradients: [N_probe, D], train_gradient: [D]
        grad_alignment = torch.mv(probe_grads, train_gradient)

        # DataInf influence formula
        # I = <g_train, g_probe> / (lambda + ||g_train||^2)
        influence = grad_alignment / (self.config.lambda_reg + train_grad_norm_sq)

        return influence

    def compute_influence_aggregated(
        self,
        train_gradient: Tensor,
    ) -> float:
        """Compute aggregated influence using average probe gradient.

        This is more efficient when we only need the mean influence.

        Args:
            train_gradient: Flattened gradient for training sample [D]

        Returns:
            Aggregated influence score (scalar)
        """
        if self._avg_probe_gradient is None:
            raise RuntimeError(
                "Probe gradients not cached. Call cache_probe_gradients first."
            )

        # Move avg probe gradient to same device as train_gradient
        # (stored on CPU to save GPU memory)
        device = train_gradient.device
        avg_probe_grad = self._avg_probe_gradient.to(device)

        train_grad_norm_sq = torch.sum(train_gradient ** 2)
        grad_alignment = torch.dot(avg_probe_grad, train_gradient)
        influence = grad_alignment / (self.config.lambda_reg + train_grad_norm_sq)

        return influence.item()

    def compute_batch_influence(
        self,
        batch: dict,
    ) -> Tensor:
        """Compute influence for a batch of training samples.

        Args:
            batch: Batch of training samples

        Returns:
            Influence matrix [batch_size, N_probe]
        """
        # Compute gradients for all samples in batch
        device = next(self.gradient_extractor.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

        train_gradients = self.gradient_extractor.compute_batch_gradients(batch)

        # Compute influence for each sample
        influences = []
        for grad in train_gradients:
            inf = self.compute_influence(grad)
            influences.append(inf)

        return torch.stack(influences)

    def compute_batch_influence_aggregated(
        self,
        batch: dict,
    ) -> Tensor:
        """Compute aggregated influence for each sample in a batch.

        More efficient than compute_batch_influence when we only need
        the mean influence per sample.

        Args:
            batch: Batch of training samples

        Returns:
            Aggregated influence scores [batch_size]
        """
        device = next(self.gradient_extractor.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

        train_gradients = self.gradient_extractor.compute_batch_gradients(batch)

        # Compute gradient norms
        train_grad_norms_sq = torch.sum(train_gradients ** 2, dim=1)

        # Move avg probe gradient to GPU for computation (stored on CPU)
        avg_probe_grad = self._avg_probe_gradient.to(device)

        # Compute alignment with average probe gradient
        # train_gradients: [B, D], avg_probe_gradient: [D]
        grad_alignments = torch.mv(train_gradients, avg_probe_grad)

        # DataInf formula
        influences = grad_alignments / (self.config.lambda_reg + train_grad_norms_sq)

        return influences

    def aggregate_influence(
        self,
        influence_matrix: Tensor,
        method: Literal["mean", "sum", "max", "min"] = "mean",
    ) -> Tensor:
        """Aggregate influence across probe set.

        Args:
            influence_matrix: [batch_size, N_probe]
            method: Aggregation method

        Returns:
            Aggregated influence [batch_size]
        """
        if method == "mean":
            return influence_matrix.mean(dim=1)
        elif method == "sum":
            return influence_matrix.sum(dim=1)
        elif method == "max":
            return influence_matrix.max(dim=1).values
        elif method == "min":
            return influence_matrix.min(dim=1).values
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


def create_influence_calculator(
    model,
    config: Optional[InfluenceConfig] = None,
) -> DataInfCalculator:
    """Factory function to create an influence calculator.

    Args:
        model: MobileLLM model
        config: Influence configuration

    Returns:
        Configured DataInfCalculator
    """
    config = config or InfluenceConfig()
    extractor = DiscriminativeGradientExtractor(model, config)
    return DataInfCalculator(extractor, config)
