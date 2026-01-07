"""
Fake MoE conversion utilities (for testing dense-to-MoE equivalence).

TODO: Implement MoE support - https://github.com/DeepOpt-com/WrinkleFree-Monorepo/issues/36
"""

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

MOE_NOT_IMPLEMENTED_MSG = (
    "MoE (Mixture of Experts) is not yet implemented. "
    "See: https://github.com/DeepOpt-com/WrinkleFree-Monorepo/issues/36"
)


@dataclass
class FakeMoEConfig:
    """Configuration for fake MoE conversion."""

    num_experts: int = 8
    top_k: int = 2
    share_expert_weights: bool = True
    use_identity_router: bool = True


class FakeMoEConverter:
    """Converter for creating fake MoE from dense models."""

    def __init__(self, config: Optional[FakeMoEConfig] = None):
        self.config = config or FakeMoEConfig()
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)

    def convert(self, module: nn.Module) -> nn.Module:
        """Convert dense module to fake MoE."""
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)


def create_fake_moe_from_dense(
    module: nn.Module,
    num_experts: int = 8,
    top_k: int = 2,
    **kwargs,
) -> nn.Module:
    """Convert a dense model to fake MoE with identity routing.

    This creates an MoE model that behaves identically to the dense model
    when using identity routing (all tokens go to expert 0).

    Args:
        module: Dense model to convert
        num_experts: Number of experts to create
        top_k: Number of experts to route to

    Returns:
        Converted MoE model
    """
    raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)


def verify_moe_matches_dense(
    dense_module: nn.Module,
    moe_module: nn.Module,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> bool:
    """Verify that MoE with identity routing matches dense model outputs.

    Args:
        dense_module: Original dense model
        moe_module: Converted MoE model
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if outputs match within tolerance
    """
    raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)
