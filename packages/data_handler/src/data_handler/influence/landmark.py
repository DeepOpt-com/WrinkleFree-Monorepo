"""Landmark selection strategies for influence distillation.

Landmarks are a representative subset of source samples for which we compute
accurate gradients. Other samples' influences are approximated via KRR.

Reference: Section 4.2 of arXiv:2505.19051
"""

import logging
from typing import Literal

import torch
from torch import Tensor
from tqdm import tqdm

from data_handler.influence.config import LandmarkConfig

logger = logging.getLogger(__name__)


class LandmarkSelector:
    """Select representative landmark samples from embeddings.

    Supports multiple selection strategies:
    - random: Uniform random selection (fastest)
    - kmeans_pp: K-means++ initialization (diverse landmarks)
    - farthest_point: Farthest point sampling (maximum coverage)

    Example:
        >>> selector = LandmarkSelector(LandmarkConfig(num_landmarks=4096))
        >>> embeddings = torch.randn(100000, 256)  # 100k samples, 256-dim
        >>> landmark_indices = selector.select(embeddings)  # [4096]
    """

    def __init__(self, config: LandmarkConfig | None = None):
        """Initialize the selector.

        Args:
            config: Landmark selection configuration
        """
        self.config = config or LandmarkConfig()

    def select(
        self,
        embeddings: Tensor,
        k: int | None = None,
    ) -> Tensor:
        """Select landmarks using the configured strategy.

        Args:
            embeddings: Sample embeddings [N, D]
            k: Number of landmarks (uses config.num_landmarks if None)

        Returns:
            Tensor of selected indices [k]
        """
        k = k or self.config.num_landmarks
        strategy = self.config.selection_strategy

        if k >= embeddings.size(0):
            logger.warning(
                f"Requested {k} landmarks but only {embeddings.size(0)} samples. "
                "Returning all indices."
            )
            return torch.arange(embeddings.size(0), device=embeddings.device)

        if strategy == "random":
            return self.select_random(embeddings, k)
        elif strategy == "kmeans_pp":
            return self.select_kmeans_pp(embeddings, k)
        elif strategy == "farthest_point":
            return self.select_farthest_point(embeddings, k)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

    def select_random(
        self,
        embeddings: Tensor,
        k: int,
    ) -> Tensor:
        """Random landmark selection.

        Args:
            embeddings: Sample embeddings [N, D]
            k: Number of landmarks to select

        Returns:
            Tensor of selected indices [k]
        """
        n = embeddings.size(0)
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(self.config.seed)

        perm = torch.randperm(n, generator=generator, device=embeddings.device)
        return perm[:k]

    def select_kmeans_pp(
        self,
        embeddings: Tensor,
        k: int,
        show_progress: bool = False,
    ) -> Tensor:
        """K-means++ initialization for diverse landmarks.

        Algorithm:
        1. Pick first center uniformly at random
        2. For each remaining center:
           - Compute distance to nearest existing center for all points
           - Sample next center with probability proportional to squared distance
        3. Repeat until k centers selected

        This produces a diverse set of landmarks that cover the embedding space.

        Args:
            embeddings: Sample embeddings [N, D]
            k: Number of landmarks to select
            show_progress: Whether to show progress bar

        Returns:
            Tensor of selected indices [k]
        """
        n = embeddings.size(0)
        device = embeddings.device
        dtype = embeddings.dtype

        # Seed random generator
        generator = torch.Generator(device=device)
        generator.manual_seed(self.config.seed)

        # First center: random
        first_idx = torch.randint(n, (1,), generator=generator, device=device).item()
        indices = [first_idx]

        # Track minimum squared distance to any center for each point
        # Initialize with distance to first center
        first_center = embeddings[first_idx:first_idx + 1]  # [1, D]
        min_dists_sq = torch.sum((embeddings - first_center) ** 2, dim=1)  # [N]

        iterator = range(k - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="Selecting landmarks (kmeans++)")

        for _ in iterator:
            # Sample next center with probability proportional to min_dists_sq
            probs = min_dists_sq.clone()
            probs[indices] = 0  # Don't re-select existing centers
            probs = probs / probs.sum()

            # Handle edge case where all probs are 0 (shouldn't happen)
            if probs.sum() == 0:
                remaining = torch.ones(n, device=device)
                remaining[indices] = 0
                probs = remaining / remaining.sum()

            new_idx = torch.multinomial(probs, 1, generator=generator).item()
            indices.append(new_idx)

            # Update minimum distances
            new_center = embeddings[new_idx:new_idx + 1]  # [1, D]
            new_dists_sq = torch.sum((embeddings - new_center) ** 2, dim=1)
            min_dists_sq = torch.minimum(min_dists_sq, new_dists_sq)

        return torch.tensor(indices, device=device, dtype=torch.long)

    def select_farthest_point(
        self,
        embeddings: Tensor,
        k: int,
        show_progress: bool = False,
    ) -> Tensor:
        """Farthest point sampling for maximum coverage.

        Algorithm:
        1. Pick first point uniformly at random
        2. For each remaining point:
           - Find the point with maximum distance to nearest selected point
           - Add it to the selected set
        3. Repeat until k points selected

        This maximizes the minimum distance between selected points (coverage).

        Args:
            embeddings: Sample embeddings [N, D]
            k: Number of landmarks to select
            show_progress: Whether to show progress bar

        Returns:
            Tensor of selected indices [k]
        """
        n = embeddings.size(0)
        device = embeddings.device

        # Seed random generator
        generator = torch.Generator(device=device)
        generator.manual_seed(self.config.seed)

        # First point: random
        first_idx = torch.randint(n, (1,), generator=generator, device=device).item()
        indices = [first_idx]

        # Track minimum distance to any selected point
        first_point = embeddings[first_idx:first_idx + 1]
        min_dists = torch.sqrt(torch.sum((embeddings - first_point) ** 2, dim=1))

        iterator = range(k - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="Selecting landmarks (farthest)")

        for _ in iterator:
            # Find point with maximum minimum distance
            # Mask already selected points
            masked_dists = min_dists.clone()
            masked_dists[indices] = -float('inf')

            new_idx = masked_dists.argmax().item()
            indices.append(new_idx)

            # Update minimum distances
            new_point = embeddings[new_idx:new_idx + 1]
            new_dists = torch.sqrt(torch.sum((embeddings - new_point) ** 2, dim=1))
            min_dists = torch.minimum(min_dists, new_dists)

        return torch.tensor(indices, device=device, dtype=torch.long)

    def select_with_chunking(
        self,
        embeddings: Tensor,
        k: int,
        chunk_size: int = 10000,
    ) -> Tensor:
        """Memory-efficient selection for very large datasets.

        For datasets too large to fit in memory, this method:
        1. Samples a larger subset first (e.g., 10x the final count)
        2. Applies the full selection algorithm on this subset
        3. Maps back to original indices

        Args:
            embeddings: Sample embeddings [N, D]
            k: Number of landmarks to select
            chunk_size: Number of samples to pre-sample

        Returns:
            Tensor of selected indices [k]
        """
        n = embeddings.size(0)

        if n <= chunk_size:
            return self.select(embeddings, k)

        # Pre-sample a subset
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(self.config.seed)
        subset_indices = torch.randperm(n, generator=generator)[:chunk_size]
        subset_embeddings = embeddings[subset_indices]

        # Select from subset
        local_indices = self.select(subset_embeddings, k)

        # Map back to original indices
        return subset_indices[local_indices]


def select_landmarks(
    embeddings: Tensor,
    num_landmarks: int,
    strategy: Literal["random", "kmeans_pp", "farthest_point"] = "kmeans_pp",
    seed: int = 42,
) -> Tensor:
    """Convenience function for landmark selection.

    Args:
        embeddings: Sample embeddings [N, D]
        num_landmarks: Number of landmarks to select
        strategy: Selection strategy
        seed: Random seed

    Returns:
        Tensor of selected indices [num_landmarks]
    """
    config = LandmarkConfig(
        num_landmarks=num_landmarks,
        selection_strategy=strategy,
        seed=seed,
    )
    selector = LandmarkSelector(config)
    return selector.select(embeddings)
