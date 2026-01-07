"""
MoE Router stubs.

TODO: Implement MoE support - https://github.com/DeepOpt-com/WrinkleFree-Monorepo/issues/36
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

MOE_NOT_IMPLEMENTED_MSG = (
    "MoE (Mixture of Experts) is not yet implemented. "
    "See: https://github.com/DeepOpt-com/WrinkleFree-Monorepo/issues/36"
)


class TopKRouter(nn.Module):
    """Top-K router for MoE layers."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            weights: Routing weights (batch, seq_len, top_k)
            experts: Selected expert indices (batch, seq_len, top_k)
            logits: Router logits (batch, seq_len, num_experts)
        """
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)


class IdentityRouter(nn.Module):
    """Identity router that always routes to specified experts (for testing)."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        target_experts: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.target_experts = target_experts or [0]
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route all tokens to target experts."""
        raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)


def compute_load_balancing_loss(
    logits: torch.Tensor,
    experts: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Compute load balancing auxiliary loss.

    Args:
        logits: Router logits (batch, seq_len, num_experts)
        experts: Selected expert indices (batch, seq_len, top_k)
        num_experts: Total number of experts
        top_k: Number of experts selected per token

    Returns:
        Scalar loss tensor
    """
    raise NotImplementedError(MOE_NOT_IMPLEMENTED_MSG)
