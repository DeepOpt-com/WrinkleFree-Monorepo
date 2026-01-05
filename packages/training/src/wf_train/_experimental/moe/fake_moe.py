"""Fake MoE converter for testing.

This module converts a dense BitNet model into a "fake" MoE model where:
1. The original FFN becomes expert 0
2. Experts 1-N are copies of expert 0 (same weights)
3. An IdentityRouter routes ALL tokens to expert 0

This means the MoE model should produce IDENTICAL outputs to the original.
Useful for:
- Testing MoE infrastructure
- Verifying serving pipeline works
- Benchmarking MoE overhead
"""

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from wf_train._experimental.moe.expert import BitNetExpertFFN, BitNetMoEFFN


@dataclass
class FakeMoEConfig:
    """Configuration for fake MoE conversion."""

    num_experts: int = 8
    """Total number of experts (N)"""

    top_k: int = 2
    """Number of active experts per token (K)"""

    share_expert_weights: bool = True
    """If True, all experts share weights with original FFN.
    If False, weights are copied (independent)."""

    use_identity_router: bool = True
    """If True, use IdentityRouter (for testing identical outputs).
    If False, use TopKRouter (for actual MoE training)."""


class FakeMoEConverter:
    """
    Convert a dense BitNet model to a fake MoE model.

    The conversion:
    1. Finds all FFN/MLP layers in the model
    2. Replaces each with a BitNetMoEFFN
    3. Expert 0 gets the original FFN weights
    4. Experts 1-N get copies (or shared references)
    5. Router is set to always select expert 0 (for identity testing)

    Example:
        >>> model = load_bitnet_model(...)
        >>> converter = FakeMoEConverter(FakeMoEConfig(num_experts=8, top_k=2))
        >>> moe_model = converter.convert(model)
        >>> # moe_model outputs should be identical to original model
    """

    def __init__(self, config: FakeMoEConfig):
        self.config = config

    def convert(self, model: nn.Module) -> nn.Module:
        """
        Convert dense model to fake MoE.

        Args:
            model: Dense BitNet model

        Returns:
            Model with MoE FFN layers
        """
        model = copy.deepcopy(model)
        self._convert_ffn_layers(model)
        return model

    def _convert_ffn_layers(self, module: nn.Module, prefix: str = "") -> None:
        """Recursively convert FFN layers to MoE."""
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this is an FFN layer to convert
            if self._is_ffn_layer(child):
                moe_ffn = self._create_moe_from_ffn(child)
                setattr(module, name, moe_ffn)
            else:
                # Recurse into child modules
                self._convert_ffn_layers(child, full_name)

    def _is_ffn_layer(self, module: nn.Module) -> bool:
        """Check if module is an FFN layer to convert."""
        # Match common FFN naming patterns
        module_name = module.__class__.__name__.lower()

        # Check for typical FFN patterns
        is_ffn = any(pattern in module_name for pattern in ["ffn", "mlp", "feedforward"])

        # Also check for gate/up/down projection structure
        has_ffn_structure = (
            hasattr(module, "gate_proj") or
            hasattr(module, "up_proj") or
            (hasattr(module, "fc1") and hasattr(module, "fc2"))
        )

        return is_ffn or has_ffn_structure

    def _create_moe_from_ffn(self, ffn: nn.Module) -> BitNetMoEFFN:
        """Create MoE FFN from dense FFN, copying weights to expert 0."""
        # Detect dimensions from FFN
        hidden_size, intermediate_size = self._detect_ffn_dims(ffn)

        # Create MoE FFN
        router_type = "identity" if self.config.use_identity_router else "topk"
        moe_ffn = BitNetMoEFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            router_type=router_type,
        )

        # Copy weights to expert 0
        self._copy_ffn_weights(ffn, moe_ffn.experts[0])

        # Copy to other experts (if not sharing)
        for i in range(1, self.config.num_experts):
            if self.config.share_expert_weights:
                # Share weight tensors (memory efficient, truly identical)
                self._share_weights(moe_ffn.experts[0], moe_ffn.experts[i])
            else:
                # Copy weights (independent copies)
                self._copy_ffn_weights(ffn, moe_ffn.experts[i])

        return moe_ffn

    def _detect_ffn_dims(self, ffn: nn.Module) -> tuple[int, int]:
        """Detect hidden_size and intermediate_size from FFN."""
        # Try common attribute patterns
        if hasattr(ffn, "gate_proj"):
            return ffn.gate_proj.in_features, ffn.gate_proj.out_features
        elif hasattr(ffn, "up_proj"):
            return ffn.up_proj.in_features, ffn.up_proj.out_features
        elif hasattr(ffn, "fc1"):
            return ffn.fc1.in_features, ffn.fc1.out_features
        elif hasattr(ffn, "w1"):
            return ffn.w1.in_features, ffn.w1.out_features
        else:
            # Search for linear layers
            for child in ffn.children():
                if isinstance(child, nn.Linear):
                    return child.in_features, child.out_features
            raise ValueError(f"Cannot detect FFN dimensions from {type(ffn)}")

    def _copy_ffn_weights(self, src_ffn: nn.Module, dst_expert: BitNetExpertFFN) -> None:
        """Copy weights from source FFN to destination expert."""
        # Map common naming conventions
        weight_mappings = [
            # (src_attr, dst_attr)
            ("gate_proj", "gate_proj"),
            ("up_proj", "up_proj"),
            ("down_proj", "down_proj"),
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
            ("fc1", "up_proj"),
            ("fc2", "down_proj"),
        ]

        for src_attr, dst_attr in weight_mappings:
            if hasattr(src_ffn, src_attr) and hasattr(dst_expert, dst_attr):
                src_layer = getattr(src_ffn, src_attr)
                dst_layer = getattr(dst_expert, dst_attr)

                if isinstance(src_layer, nn.Linear) and isinstance(dst_layer, nn.Linear):
                    dst_layer.weight.data.copy_(src_layer.weight.data)
                    if src_layer.bias is not None and dst_layer.bias is not None:
                        dst_layer.bias.data.copy_(src_layer.bias.data)

    def _share_weights(self, src_expert: BitNetExpertFFN, dst_expert: BitNetExpertFFN) -> None:
        """Make dst_expert share weight tensors with src_expert."""
        dst_expert.gate_proj.weight = src_expert.gate_proj.weight
        dst_expert.up_proj.weight = src_expert.up_proj.weight
        dst_expert.down_proj.weight = src_expert.down_proj.weight

        if src_expert.gate_proj.bias is not None:
            dst_expert.gate_proj.bias = src_expert.gate_proj.bias
        if src_expert.up_proj.bias is not None:
            dst_expert.up_proj.bias = src_expert.up_proj.bias
        if src_expert.down_proj.bias is not None:
            dst_expert.down_proj.bias = src_expert.down_proj.bias


def create_fake_moe_from_dense(
    model: nn.Module,
    num_experts: int = 8,
    top_k: int = 2,
    use_identity_router: bool = True,
) -> nn.Module:
    """
    Convenience function to create fake MoE from dense model.

    Args:
        model: Dense BitNet model
        num_experts: Total number of experts (N)
        top_k: Number of active experts (K)
        use_identity_router: Use IdentityRouter for testing

    Returns:
        Model with MoE FFN layers

    Example:
        >>> dense_model = load_model("microsoft/BitNet-b1.58-2B-4T")
        >>> moe_model = create_fake_moe_from_dense(dense_model, num_experts=8, top_k=2)
        >>> # Verify outputs match
        >>> dense_out = dense_model(input_ids)
        >>> moe_out = moe_model(input_ids)
        >>> assert torch.allclose(dense_out, moe_out)
    """
    config = FakeMoEConfig(
        num_experts=num_experts,
        top_k=top_k,
        share_expert_weights=True,
        use_identity_router=use_identity_router,
    )
    converter = FakeMoEConverter(config)
    return converter.convert(model)


def verify_moe_matches_dense(
    dense_model: nn.Module,
    moe_model: nn.Module,
    input_ids: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[bool, Optional[str]]:
    """
    Verify that MoE model produces identical outputs to dense model.

    Args:
        dense_model: Original dense model
        moe_model: Converted MoE model (with IdentityRouter)
        input_ids: Test input tokens
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        (matches, error_message) where matches is True if outputs match
    """
    dense_model.eval()
    moe_model.eval()

    with torch.no_grad():
        dense_out = dense_model(input_ids)
        moe_out = moe_model(input_ids)

    # Handle different output formats
    if isinstance(dense_out, dict):
        dense_logits = dense_out.get("logits", dense_out.get("last_hidden_state"))
        moe_logits = moe_out.get("logits", moe_out.get("last_hidden_state"))
    elif isinstance(dense_out, tuple):
        dense_logits = dense_out[0]
        moe_logits = moe_out[0]
    else:
        dense_logits = dense_out
        moe_logits = moe_out

    if dense_logits is None or moe_logits is None:
        return False, "Could not extract outputs from models"

    if not torch.allclose(dense_logits, moe_logits, atol=atol, rtol=rtol):
        max_diff = (dense_logits - moe_logits).abs().max().item()
        return False, f"Output mismatch: max diff = {max_diff}"

    return True, None
