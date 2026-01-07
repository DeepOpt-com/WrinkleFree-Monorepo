"""
MoE (Mixture of Experts) module stubs.

TODO: Implement MoE support - https://github.com/DeepOpt-com/WrinkleFree-Monorepo/issues/36

This module provides stub implementations that raise NotImplementedError.
Tests that import from this module will fail with clear messages about
the missing implementation.
"""

from wf_infer.moe.router import TopKRouter, IdentityRouter, compute_load_balancing_loss
from wf_infer.moe.expert import BitLinear, BitNetExpertFFN, BitNetMoEFFN, BitNetMoELayer
from wf_infer.moe.fake_moe import (
    FakeMoEConfig,
    FakeMoEConverter,
    create_fake_moe_from_dense,
    verify_moe_matches_dense,
)

__all__ = [
    # Router
    "TopKRouter",
    "IdentityRouter",
    "compute_load_balancing_loss",
    # Expert
    "BitLinear",
    "BitNetExpertFFN",
    "BitNetMoEFFN",
    "BitNetMoELayer",
    # Fake MoE
    "FakeMoEConfig",
    "FakeMoEConverter",
    "create_fake_moe_from_dense",
    "verify_moe_matches_dense",
]
