"""MoE Router implementations for BitNet.

Routers determine which experts process each token. Key variants:
- TopKRouter: Standard top-k routing (e.g., Mixtral uses top-2)
- IdentityRouter: For testing - routes all tokens to specific expert(s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MoERouter(nn.Module):
    """
    Base class for MoE routers.

    Routes input tokens to a subset of experts based on routing scores.

    Args:
        hidden_size: Input hidden dimension
        num_experts: Total number of experts (N)
        top_k: Number of experts to route each token to (K)
        router_jitter: Add noise during training for load balancing
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        router_jitter: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter = router_jitter

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Tuple of:
                - routing_weights: Weights for each token-expert pair (batch, seq_len, top_k)
                - selected_experts: Expert indices for each token (batch, seq_len, top_k)
                - router_logits: Raw router scores (batch, seq_len, num_experts)
        """
        raise NotImplementedError


class TopKRouter(MoERouter):
    """
    Standard Top-K router used in Mixtral, Switch Transformer, etc.

    Uses a learned linear projection to compute routing scores,
    then selects top-K experts for each token.

    The routing weights are softmax-normalized over selected experts.

    Args:
        hidden_size: Input hidden dimension
        num_experts: Total number of experts (N)
        top_k: Number of experts to route each token to (K)
        router_jitter: Noise scale during training (0 = disabled)
        normalize_expert_weights: Whether to normalize weights to sum to 1
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        router_jitter: float = 0.0,
        normalize_expert_weights: bool = True,
    ):
        super().__init__(hidden_size, num_experts, top_k, router_jitter)
        self.normalize_expert_weights = normalize_expert_weights

        # Router gate: projects hidden states to expert scores
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-K experts.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            routing_weights: (batch, seq_len, top_k)
            selected_experts: (batch, seq_len, top_k)
            router_logits: (batch, seq_len, num_experts)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Compute router logits
        router_logits = self.gate(hidden_states)  # (batch, seq_len, num_experts)

        # Add jitter during training for load balancing
        if self.training and self.router_jitter > 0:
            noise = torch.randn_like(router_logits) * self.router_jitter
            router_logits = router_logits + noise

        # Select top-K experts
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # Both: (batch, seq_len, top_k)

        # Normalize weights (softmax over selected experts)
        if self.normalize_expert_weights:
            routing_weights = F.softmax(routing_weights, dim=-1)
        else:
            # Use softmax over all experts but only keep top-k values
            all_weights = F.softmax(router_logits, dim=-1)
            routing_weights = torch.gather(all_weights, -1, selected_experts)

        return routing_weights, selected_experts, router_logits


class IdentityRouter(MoERouter):
    """
    Identity router for testing - routes all tokens to specific expert(s).

    This is useful for creating "fake" MoE models where we want to
    verify that outputs match a dense model.

    When target_expert=0 and top_k=1, all tokens go to expert 0,
    which should produce identical outputs to the original dense FFN.

    Args:
        hidden_size: Input hidden dimension
        num_experts: Total number of experts (N)
        top_k: Number of experts per token (K)
        target_experts: Which expert(s) to always route to (default: [0])
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 1,
        target_experts: Optional[list[int]] = None,
    ):
        super().__init__(hidden_size, num_experts, top_k)
        self.target_experts = target_experts or list(range(top_k))

        if len(self.target_experts) != top_k:
            raise ValueError(f"target_experts must have {top_k} elements")

        # Register as buffer for device handling
        self.register_buffer(
            "expert_indices",
            torch.tensor(self.target_experts, dtype=torch.long),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route all tokens to the same expert(s).

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            routing_weights: All 1/top_k (batch, seq_len, top_k)
            selected_experts: All target_experts (batch, seq_len, top_k)
            router_logits: Zeros (batch, seq_len, num_experts)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Equal weights for all selected experts
        routing_weights = torch.ones(
            batch_size, seq_len, self.top_k,
            device=device, dtype=hidden_states.dtype
        ) / self.top_k

        # All tokens go to same experts
        selected_experts = self.expert_indices.expand(batch_size, seq_len, -1)

        # Dummy router logits
        router_logits = torch.zeros(
            batch_size, seq_len, self.num_experts,
            device=device, dtype=hidden_states.dtype
        )

        return routing_weights, selected_experts, router_logits


def compute_load_balancing_loss(
    router_logits: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """
    Compute auxiliary load balancing loss to encourage even expert usage.

    This is the standard aux loss from Switch Transformer / Mixtral.

    Args:
        router_logits: Raw router scores (batch, seq_len, num_experts)
        selected_experts: Selected expert indices (batch, seq_len, top_k)
        num_experts: Total number of experts
        top_k: Number of experts per token

    Returns:
        Scalar load balancing loss
    """
    # Compute expert probabilities
    router_probs = F.softmax(router_logits, dim=-1)  # (batch, seq_len, num_experts)

    # Compute fraction of tokens routed to each expert
    expert_mask = F.one_hot(selected_experts, num_experts)  # (batch, seq_len, top_k, num_experts)
    expert_mask = expert_mask.sum(dim=2)  # (batch, seq_len, num_experts)

    # Average across batch and sequence
    tokens_per_expert = expert_mask.float().mean(dim=[0, 1])  # (num_experts,)
    router_prob_per_expert = router_probs.mean(dim=[0, 1])  # (num_experts,)

    # Loss encourages uniform distribution
    # High loss = some experts get many tokens, others get few
    loss = num_experts * (tokens_per_expert * router_prob_per_expert).sum()

    return loss
