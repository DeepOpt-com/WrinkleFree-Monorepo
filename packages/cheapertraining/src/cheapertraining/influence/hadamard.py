"""Randomized Hadamard Transform for fast gradient projection.

Provides O(D log D) dimensionality reduction that approximately preserves
pairwise distances (Johnson-Lindenstrauss property).

Reference:
- Fast JL Transform: Ailon & Chazelle, "Approximate Nearest Neighbors and
  the Fast Johnson-Lindenstrauss Transform" (2006)
- HadaCore PyTorch implementation: https://pytorch.org/blog/hadacore/
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _fast_walsh_hadamard_transform(x: Tensor) -> Tensor:
    """Compute the Walsh-Hadamard transform in O(n log n) time.

    Uses the recursive structure of the Hadamard matrix:
    H_2n = [[H_n, H_n], [H_n, -H_n]]

    Args:
        x: Input tensor of shape [..., n] where n is a power of 2

    Returns:
        Transformed tensor of same shape
    """
    n = x.size(-1)

    # Verify n is power of 2
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"

    # In-place butterfly operations
    h = 1
    while h < n:
        # Split into pairs and apply Hadamard butterfly
        x = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        # x[..., 0, :] = a, x[..., 1, :] = b
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.view(*x.shape[:-3], n)
        h *= 2

    return x


class RandomizedHadamardTransform(nn.Module):
    """Fast randomized projection using Hadamard transform.

    Projects D-dimensional vectors to K dimensions in O(D log D) time
    using the randomized Hadamard transform:

        y = PHDx

    Where:
    - D is a random diagonal matrix of {+1, -1}
    - H is the normalized Hadamard matrix
    - P is a random subset selection of K coordinates

    This approximately preserves pairwise distances with high probability
    (Johnson-Lindenstrauss lemma).

    Example:
        >>> transform = RandomizedHadamardTransform(input_dim=1024, output_dim=256)
        >>> x = torch.randn(batch_size, 1024)
        >>> y = transform(x)  # [batch_size, 256]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int = 42,
        normalize: bool = True,
    ):
        """Initialize the transform.

        Args:
            input_dim: Input dimension (will be padded to power of 2)
            output_dim: Output projection dimension
            seed: Random seed for reproducibility
            normalize: Whether to normalize by sqrt(padded_dim)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize

        # Pad to power of 2
        self.padded_dim = _next_power_of_2(input_dim)

        # Generate random sign flips (diagonal D matrix)
        generator = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (self.padded_dim,), generator=generator) * 2 - 1
        self.register_buffer("signs", signs.float())

        # Generate random coordinate selection (matrix P)
        # Select output_dim coordinates uniformly at random
        perm = torch.randperm(self.padded_dim, generator=generator)[:output_dim]
        self.register_buffer("selected_indices", perm)

        # Normalization constant
        if normalize:
            self.scale = 1.0 / math.sqrt(self.padded_dim)
        else:
            self.scale = 1.0

    def forward(self, x: Tensor) -> Tensor:
        """Project input vectors to lower dimension.

        Args:
            x: Input tensor of shape [..., input_dim]

        Returns:
            Projected tensor of shape [..., output_dim]
        """
        # Pad to power of 2 if needed
        if x.size(-1) < self.padded_dim:
            padding = torch.zeros(
                *x.shape[:-1], self.padded_dim - x.size(-1),
                device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, padding], dim=-1)
        elif x.size(-1) > self.padded_dim:
            raise ValueError(
                f"Input dim {x.size(-1)} > padded dim {self.padded_dim}"
            )

        # Apply random sign flips: Dx
        x = x * self.signs.to(x.device)

        # Apply Hadamard transform: HDx
        x = _fast_walsh_hadamard_transform(x)

        # Normalize
        x = x * self.scale

        # Select output coordinates: PHDx
        x = x.index_select(-1, self.selected_indices.to(x.device))

        return x

    def project(self, x: Tensor) -> Tensor:
        """Alias for forward() for API consistency."""
        return self.forward(x)

    def project_batch(
        self,
        x: Tensor,
        chunk_size: int = 1000,
    ) -> Tensor:
        """Project in batches for memory efficiency.

        Args:
            x: Input tensor of shape [N, input_dim]
            chunk_size: Number of samples per chunk

        Returns:
            Projected tensor of shape [N, output_dim]
        """
        if x.size(0) <= chunk_size:
            return self.forward(x)

        results = []
        for i in range(0, x.size(0), chunk_size):
            chunk = x[i:i + chunk_size]
            results.append(self.forward(chunk))

        return torch.cat(results, dim=0)


class IdentityTransform(nn.Module):
    """Identity transform (no projection) for baseline comparison."""

    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.output_dim < self.input_dim:
            return x[..., :self.output_dim]
        return x

    def project(self, x: Tensor) -> Tensor:
        return self.forward(x)


def create_projection(
    input_dim: int,
    output_dim: int,
    method: str = "hadamard",
    seed: int = 42,
) -> nn.Module:
    """Factory function to create a projection transform.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        method: "hadamard" for RHT, "identity" for no projection
        seed: Random seed

    Returns:
        Projection module
    """
    if method == "hadamard":
        return RandomizedHadamardTransform(input_dim, output_dim, seed)
    elif method == "identity":
        return IdentityTransform(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown projection method: {method}")
