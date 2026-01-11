"""
MoE Expert stubs.

TODO: Implement MoE support - https://github.com/DeepOpt-com/WrinkleFree-Monorepo/issues/36
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

MOE_NOT_IMPLEMENTED_MSG = (
    "MoE (Mixture of Experts) is not yet implemented. "
    "See: https://github.com/DeepOpt-com/WrinkleFree-Monorepo/issues/36"
)


class BitLinear(nn.Module):
    """BitLinear layer with ternary weights and INT8 activations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)

    def weight_quant(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize weights to ternary values."""
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)

    def activation_quant(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activations to INT8."""
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)


class BitNetExpertFFN(nn.Module):
    """Single expert FFN with BitLinear layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)


class BitNetMoEFFN(nn.Module):
    """MoE FFN with multiple BitNet experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        router_type: str = "topk",
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_type = router_type
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)

    def forward(
        self,
        x: torch.Tensor,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through MoE FFN.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            output_router_logits: Whether to return router logits

        Returns:
            output: Output tensor (batch, seq_len, hidden_size)
            router_logits: Optional router logits if requested
        """
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)


class BitNetMoELayer(nn.Module):
    """Full MoE transformer layer with attention and MoE FFN."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        num_experts: int,
        top_k: int,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.num_experts = num_experts
        self.top_k = top_k
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)
