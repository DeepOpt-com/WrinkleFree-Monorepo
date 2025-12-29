"""MoE (Mixture of Experts) module for WrinkleFree BitNet training.

This module provides MoE components for training and converting MoE models to GGUF:
- Routers: TopKRouter, IdentityRouter
- Experts: BitNetExpertFFN, BitNetMoEFFN
- Conversion: FakeMoEConverter for testing
"""

from wrinklefree.moe.router import (
    MoERouter,
    TopKRouter,
    IdentityRouter,
    compute_load_balancing_loss,
)
from wrinklefree.moe.expert import (
    BitLinear,
    BitNetExpertFFN,
    BitNetMoEFFN,
    BitNetMoELayer,
)
from wrinklefree.moe.fake_moe import (
    FakeMoEConfig,
    FakeMoEConverter,
    create_fake_moe_from_dense,
    verify_moe_matches_dense,
)

__all__ = [
    # Routers
    "MoERouter",
    "TopKRouter",
    "IdentityRouter",
    "compute_load_balancing_loss",
    # Experts
    "BitLinear",
    "BitNetExpertFFN",
    "BitNetMoEFFN",
    "BitNetMoELayer",
    # Conversion
    "FakeMoEConfig",
    "FakeMoEConverter",
    "create_fake_moe_from_dense",
    "verify_moe_matches_dense",
]
