"""Reproducibility utilities for training.

Provides functions for setting random seeds across all libraries used in training.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed to use across all libraries.

    Example:
        >>> from wf_train.utils import set_seed
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
