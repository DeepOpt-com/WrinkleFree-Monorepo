"""Mixture of Experts (MoE) support for BitNet models.

This module provides MoE layer implementations compatible with BitNet's
1.58-bit quantization, including routers, expert networks, and utilities
for converting dense models to MoE architectures.

Components:
    MoERouter: Base router class for expert selection
    TopKRouter: Top-K routing strategy
    BitNetMoEFFN: MoE feed-forward network with BitNet experts
    BitNetMoELayer: Complete MoE layer with routing and experts
    FakeMoEConverter: Convert dense models to "fake" MoE for testing
    create_fake_moe_from_dense: Factory function for fake MoE conversion
"""

from wrinklefree_inference.moe.expert import BitNetMoEFFN, BitNetMoELayer
from wrinklefree_inference.moe.fake_moe import FakeMoEConverter, create_fake_moe_from_dense
from wrinklefree_inference.moe.router import MoERouter, TopKRouter

__all__ = [
    "MoERouter",
    "TopKRouter",
    "BitNetMoEFFN",
    "BitNetMoELayer",
    "FakeMoEConverter",
    "create_fake_moe_from_dense",
]
