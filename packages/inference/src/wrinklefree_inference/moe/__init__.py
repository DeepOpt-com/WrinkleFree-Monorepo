"""Mixture of Experts (MoE) support for BitNet models."""

from wrinklefree_inference.moe.router import MoERouter, TopKRouter
from wrinklefree_inference.moe.expert import BitNetMoEFFN, BitNetMoELayer
from wrinklefree_inference.moe.fake_moe import FakeMoEConverter, create_fake_moe_from_dense

__all__ = [
    "MoERouter",
    "TopKRouter",
    "BitNetMoEFFN",
    "BitNetMoELayer",
    "FakeMoEConverter",
    "create_fake_moe_from_dense",
]
