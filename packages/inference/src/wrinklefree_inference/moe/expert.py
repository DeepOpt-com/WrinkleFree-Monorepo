"""MoE Expert implementations for BitNet.

This module provides MoE-aware FFN layers that:
1. Use BitLinear (1.58-bit weights, INT8 activations)
2. Support configurable K-of-N expert routing
3. Can be created from existing dense models for testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from wrinklefree_inference.moe.router import MoERouter, TopKRouter, IdentityRouter


class BitLinear(nn.Linear):
    """
    BitLinear with INT8 activation quantization.

    1.58-bit weight quantization: {-1, 0, 1} * scale
    INT8 activation quantization: per-token absmax to [-128, 127]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps

    def weight_quant(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize weights to ternary {-1, 0, 1}."""
        scale = 1.0 / w.abs().mean().clamp(min=self.eps)
        return (w * scale).round().clamp(-1, 1) / scale

    def activation_quant(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activations to INT8 per-token."""
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=self.eps)
        return (x * scale).round().clamp(-128, 127) / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with quantized weights and INT8 activations."""
        w = self.weight.to(x.dtype)
        w_quant = w + (self.weight_quant(w) - w).detach()
        x_quant = x + (self.activation_quant(x) - x).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x_quant, w_quant, bias)


class BitNetExpertFFN(nn.Module):
    """
    Single BitNet expert FFN (SwiGLU-style with ReLU²).

    Architecture: FFN(x) = down(up(x) * relu²(gate(x)))

    Uses INT8 activation quantization throughout.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through expert FFN."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # ReLU² activation
        activated = up * F.relu(gate).pow(2)
        return self.down_proj(activated)


class BitNetMoEFFN(nn.Module):
    """
    Mixture of Experts FFN for BitNet with configurable K-of-N routing.

    Each expert is a full BitNetExpertFFN. The router selects top-K experts
    for each token, and outputs are weighted sum of expert outputs.

    For testing, use IdentityRouter to make all tokens go to expert 0,
    which should produce identical outputs to a dense model.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension per expert
        num_experts: Total number of experts (N)
        top_k: Number of active experts per token (K)
        router_type: "topk" for learned routing, "identity" for testing
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        router_type: str = "topk",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Create router
        if router_type == "topk":
            self.router = TopKRouter(hidden_size, num_experts, top_k)
        elif router_type == "identity":
            self.router = IdentityRouter(hidden_size, num_experts, top_k)
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

        # Create experts
        self.experts = nn.ModuleList([
            BitNetExpertFFN(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward through MoE FFN.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            output_router_logits: Whether to return router logits for aux loss

        Returns:
            output: (batch, seq_len, hidden_size)
            router_logits: (batch, seq_len, num_experts) if output_router_logits
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get routing decisions
        routing_weights, selected_experts, router_logits = self.router(hidden_states)
        # routing_weights: (batch, seq_len, top_k)
        # selected_experts: (batch, seq_len, top_k)

        # Compute expert outputs
        # For efficiency, we process all tokens through all selected experts
        # then combine with routing weights

        # Flatten for easier indexing
        hidden_flat = hidden_states.view(-1, hidden_size)  # (batch*seq, hidden)
        routing_flat = routing_weights.view(-1, self.top_k)  # (batch*seq, top_k)
        experts_flat = selected_experts.view(-1, self.top_k)  # (batch*seq, top_k)

        # Initialize output
        output_flat = torch.zeros_like(hidden_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert (any of their top-k choices)
            expert_mask = (experts_flat == expert_idx).any(dim=-1)  # (batch*seq,)

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_input = hidden_flat[expert_mask]  # (num_tokens, hidden)

            # Run through expert
            expert_output = self.experts[expert_idx](expert_input)  # (num_tokens, hidden)

            # Get weights for this expert
            # For each token, find which top-k slot has this expert
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            expert_weights = torch.zeros(token_indices.shape[0], device=hidden_flat.device, dtype=hidden_flat.dtype)

            for k in range(self.top_k):
                k_mask = experts_flat[token_indices, k] == expert_idx
                expert_weights[k_mask] = routing_flat[token_indices[k_mask], k]

            # Weighted addition to output
            output_flat[expert_mask] += expert_output * expert_weights.unsqueeze(-1)

        # Reshape output
        output = output_flat.view(batch_size, seq_len, hidden_size)

        if output_router_logits:
            return output, router_logits
        return output, None


class BitNetMoELayer(nn.Module):
    """
    Full transformer layer with MoE FFN for BitNet.

    Combines:
    - Self-attention (standard BitNet attention)
    - MoE FFN with K-of-N routing

    This is a drop-in replacement for standard BitNet transformer layers.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension per expert
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        num_experts: Total number of experts (N)
        top_k: Number of active experts per token (K)
        head_dim: Dimension per attention head
        router_type: "topk" or "identity"
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        head_dim: Optional[int] = None,
        router_type: str = "topk",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Layer norms
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-5)

        # Attention (simplified - in real impl would use BitNet attention)
        head_dim = head_dim or hidden_size // num_attention_heads
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = BitLinear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = BitLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = BitLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = BitLinear(num_attention_heads * head_dim, hidden_size, bias=False)

        # MoE FFN
        self.mlp = BitNetMoEFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            router_type=router_type,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward through MoE transformer layer.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            output_router_logits: Return router logits for aux loss

        Returns:
            hidden_states: (batch, seq_len, hidden_size)
            router_logits: Optional (batch, seq_len, num_experts)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Simplified attention (would use RoPE in full impl)
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA: repeat KV heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        hidden_states = residual + attn_output

        # MoE FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states, output_router_logits)
        hidden_states = residual + hidden_states

        return hidden_states, router_logits
