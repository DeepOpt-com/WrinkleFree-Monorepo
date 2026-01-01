"""Abstract base classes for influence calculation.

Defines common interfaces for embedding extraction and influence calculation,
enabling both DataInf and InfluenceDistillation methods to share a common API.

Reference:
- DataInf (ICLR 2024): Efficient influence without Hessian inversion
- InfluenceDistillation (arXiv:2505.19051): Landmark-based influence approximation
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from torch import Tensor
from torch.utils.data import DataLoader


@runtime_checkable
class EmbeddingExtractor(Protocol):
    """Protocol for extracting sample embeddings.

    Both gradient-based (DiscriminativeGradientExtractor) and JVP-based
    (JVPEmbeddingExtractor) extractors implement this protocol.
    """

    def compute_embeddings(
        self,
        dataloader: DataLoader,
        max_samples: int | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """Extract embeddings for samples in a dataloader.

        Args:
            dataloader: DataLoader providing batches
            max_samples: Maximum samples to process (None = all)
            show_progress: Whether to show progress bar

        Returns:
            Tensor of shape [N, D] where N is samples, D is embedding dim
        """
        ...

    def compute_batch_embeddings(self, batch: dict) -> Tensor:
        """Compute embeddings for a single batch.

        Args:
            batch: Dict with input_ids, attention_mask, etc.

        Returns:
            Tensor of shape [batch_size, D]
        """
        ...

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension D."""
        ...


class InfluenceCalculator(ABC):
    """Abstract base class for influence calculation methods.

    Both DataInfCalculator and InfluenceDistillation inherit from this.
    """

    @abstractmethod
    def cache_probe_gradients(
        self,
        probe_dataloader: DataLoader,
        show_progress: bool = True,
    ) -> None:
        """Pre-compute and cache probe/target gradients.

        Args:
            probe_dataloader: DataLoader for probe set samples
            show_progress: Whether to show progress bar
        """
        pass

    @abstractmethod
    def compute_influence_scores(
        self,
        source_loader: DataLoader,
        max_samples: int | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """Compute influence scores for source samples.

        Args:
            source_loader: DataLoader for source samples
            max_samples: Maximum samples to process
            show_progress: Whether to show progress bar

        Returns:
            Tensor of shape [N_source] with influence scores
        """
        pass

    @abstractmethod
    def compute_batch_influence_aggregated(
        self,
        batch: dict,
    ) -> Tensor:
        """Compute aggregated influence for samples in a batch.

        This is the primary method for continuous rebalancing during training.

        Args:
            batch: Dict with input_ids, attention_mask, etc.

        Returns:
            Tensor of shape [batch_size] with aggregated influence scores
        """
        pass

    @property
    @abstractmethod
    def is_cached(self) -> bool:
        """Check if probe gradients are cached."""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear cached probe gradients."""
        pass


class DataSelector(ABC):
    """Abstract base class for data selection strategies.

    Used for one-time selection of top-k samples before training.
    """

    @abstractmethod
    def select(
        self,
        source_loader: DataLoader,
        budget_k: int,
        target_loader: DataLoader | None = None,
        show_progress: bool = True,
    ) -> list[int]:
        """Select top-k samples from source data.

        Args:
            source_loader: DataLoader for candidate samples
            budget_k: Number of samples to select
            target_loader: Optional target/probe data (if not already cached)
            show_progress: Whether to show progress bar

        Returns:
            List of selected sample indices
        """
        pass
