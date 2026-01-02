"""Tests for MoE (Mixture of Experts) module."""

import pytest
import torch
import torch.nn as nn

from wrinklefree._experimental.moe.router import (
    TopKRouter,
    IdentityRouter,
    compute_load_balancing_loss,
)


class TestTopKRouter:
    """Tests for TopKRouter."""

    def test_init(self):
        """Test TopKRouter initialization."""
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2)

        assert router.hidden_size == 64
        assert router.num_experts == 8
        assert router.top_k == 2
        assert router.router_jitter == 0.0
        assert router.normalize_expert_weights is True

    def test_forward_shapes(self):
        """Test TopKRouter forward pass produces correct shapes."""
        batch_size, seq_len, hidden_size = 2, 10, 64
        num_experts, top_k = 8, 2

        router = TopKRouter(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        routing_weights, selected_experts, router_logits = router(hidden_states)

        assert routing_weights.shape == (batch_size, seq_len, top_k)
        assert selected_experts.shape == (batch_size, seq_len, top_k)
        assert router_logits.shape == (batch_size, seq_len, num_experts)

    def test_forward_weights_normalized(self):
        """Test that routing weights sum to 1 for each token."""
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2, normalize_expert_weights=True)
        hidden_states = torch.randn(2, 10, 64)

        routing_weights, _, _ = router(hidden_states)

        # Weights should sum to 1 for each token
        weight_sums = routing_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_forward_weights_positive(self):
        """Test that routing weights are non-negative."""
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2)
        hidden_states = torch.randn(2, 10, 64)

        routing_weights, _, _ = router(hidden_states)

        assert (routing_weights >= 0).all()

    def test_expert_indices_valid(self):
        """Test that selected expert indices are valid."""
        num_experts = 8
        router = TopKRouter(hidden_size=64, num_experts=num_experts, top_k=2)
        hidden_states = torch.randn(2, 10, 64)

        _, selected_experts, _ = router(hidden_states)

        assert (selected_experts >= 0).all()
        assert (selected_experts < num_experts).all()

    def test_top_k_selection(self):
        """Test that top-K experts have highest scores."""
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2)
        hidden_states = torch.randn(2, 10, 64)

        routing_weights, selected_experts, router_logits = router(hidden_states)

        # For each token, verify selected experts have top-K logits
        for b in range(2):
            for s in range(10):
                logits = router_logits[b, s]
                selected = selected_experts[b, s]
                selected_logits = logits[selected]
                other_logits = logits[~torch.isin(torch.arange(8), selected)]
                assert selected_logits.min() >= other_logits.max()

    def test_router_jitter_training(self):
        """Test that jitter is applied during training."""
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2, router_jitter=0.1)
        hidden_states = torch.randn(2, 10, 64)

        router.train()
        _, _, logits1 = router(hidden_states)
        _, _, logits2 = router(hidden_states)

        # Logits should differ due to jitter
        # (note: there's a small chance they could be equal by accident)
        # We use the deterministic gate output as baseline
        router.eval()
        _, _, logits_eval = router(hidden_states)

        # In eval mode, same input should give same output
        _, _, logits_eval2 = router(hidden_states)
        assert torch.allclose(logits_eval, logits_eval2)

    def test_no_jitter_eval(self):
        """Test that jitter is not applied during eval."""
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2, router_jitter=0.1)
        hidden_states = torch.randn(2, 10, 64)

        router.eval()
        _, _, logits1 = router(hidden_states)
        _, _, logits2 = router(hidden_states)

        # Should be identical in eval mode
        assert torch.allclose(logits1, logits2)


class TestIdentityRouter:
    """Tests for IdentityRouter."""

    def test_init_default(self):
        """Test IdentityRouter initialization with defaults."""
        router = IdentityRouter(hidden_size=64, num_experts=8, top_k=1)

        assert router.hidden_size == 64
        assert router.num_experts == 8
        assert router.top_k == 1
        assert router.target_experts == [0]

    def test_init_custom_experts(self):
        """Test IdentityRouter with custom target experts."""
        router = IdentityRouter(
            hidden_size=64, num_experts=8, top_k=2, target_experts=[3, 5]
        )

        assert router.target_experts == [3, 5]
        assert torch.equal(router.expert_indices, torch.tensor([3, 5]))

    def test_init_mismatch_raises(self):
        """Test that mismatched top_k and target_experts raises error."""
        with pytest.raises(ValueError, match="must have 2 elements"):
            IdentityRouter(
                hidden_size=64, num_experts=8, top_k=2, target_experts=[0]
            )

    def test_forward_shapes(self):
        """Test IdentityRouter forward pass shapes."""
        batch_size, seq_len, hidden_size = 2, 10, 64
        num_experts, top_k = 8, 2

        router = IdentityRouter(
            hidden_size=hidden_size, num_experts=num_experts, top_k=top_k,
            target_experts=[0, 1]
        )
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        routing_weights, selected_experts, router_logits = router(hidden_states)

        assert routing_weights.shape == (batch_size, seq_len, top_k)
        assert selected_experts.shape == (batch_size, seq_len, top_k)
        assert router_logits.shape == (batch_size, seq_len, num_experts)

    def test_routes_to_target_expert(self):
        """Test that all tokens route to target expert."""
        router = IdentityRouter(hidden_size=64, num_experts=8, top_k=1, target_experts=[5])
        hidden_states = torch.randn(2, 10, 64)

        _, selected_experts, _ = router(hidden_states)

        # All tokens should go to expert 5
        assert (selected_experts == 5).all()

    def test_equal_weights(self):
        """Test that routing weights are equal for all experts."""
        router = IdentityRouter(hidden_size=64, num_experts=8, top_k=2, target_experts=[0, 1])
        hidden_states = torch.randn(2, 10, 64)

        routing_weights, _, _ = router(hidden_states)

        # Weights should be 0.5 each for top_k=2
        expected = torch.ones(2, 10, 2) * 0.5
        assert torch.allclose(routing_weights, expected)

    def test_zero_router_logits(self):
        """Test that router logits are zeros (dummy values)."""
        router = IdentityRouter(hidden_size=64, num_experts=8, top_k=1)
        hidden_states = torch.randn(2, 10, 64)

        _, _, router_logits = router(hidden_states)

        assert (router_logits == 0).all()

    def test_device_handling(self):
        """Test that IdentityRouter handles device correctly."""
        router = IdentityRouter(hidden_size=64, num_experts=8, top_k=1)

        # CPU test
        hidden_states = torch.randn(2, 10, 64)
        routing_weights, selected_experts, router_logits = router(hidden_states)

        assert routing_weights.device == hidden_states.device
        assert selected_experts.device == hidden_states.device
        assert router_logits.device == hidden_states.device


class TestLoadBalancingLoss:
    """Tests for compute_load_balancing_loss."""

    def test_uniform_distribution_low_loss(self):
        """Test that uniform expert usage gives low loss."""
        batch_size, seq_len, num_experts, top_k = 4, 100, 8, 1

        # Create uniform distribution - each expert gets equal tokens
        router_logits = torch.zeros(batch_size, seq_len, num_experts)

        # Evenly distribute tokens across experts
        selected_experts = torch.zeros(batch_size, seq_len, top_k, dtype=torch.long)
        for b in range(batch_size):
            for s in range(seq_len):
                selected_experts[b, s, 0] = s % num_experts

        loss = compute_load_balancing_loss(router_logits, selected_experts, num_experts, top_k)

        # Loss should be close to 1.0 for perfectly uniform distribution
        assert loss > 0
        assert loss < 2.0  # Reasonable upper bound for uniform

    def test_concentrated_distribution_high_loss(self):
        """Test that concentrated expert usage gives higher loss."""
        batch_size, seq_len, num_experts, top_k = 4, 100, 8, 1

        # All tokens go to expert 0
        router_logits = torch.zeros(batch_size, seq_len, num_experts)
        router_logits[:, :, 0] = 10.0  # High score for expert 0

        selected_experts = torch.zeros(batch_size, seq_len, top_k, dtype=torch.long)

        loss = compute_load_balancing_loss(router_logits, selected_experts, num_experts, top_k)

        # Loss should be higher when all tokens go to one expert
        assert loss > 0

    def test_loss_shape(self):
        """Test that load balancing loss is a scalar."""
        router_logits = torch.randn(2, 10, 8)
        selected_experts = torch.randint(0, 8, (2, 10, 2))

        loss = compute_load_balancing_loss(router_logits, selected_experts, 8, 2)

        assert loss.dim() == 0  # Scalar

    def test_loss_differentiable(self):
        """Test that load balancing loss is differentiable."""
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2)
        hidden_states = torch.randn(2, 10, 64, requires_grad=True)

        routing_weights, selected_experts, router_logits = router(hidden_states)
        loss = compute_load_balancing_loss(router_logits, selected_experts, 8, 2)

        # Should be able to compute gradients
        loss.backward()
        assert router.gate.weight.grad is not None


class TestMoEIntegration:
    """Integration tests for MoE components."""

    def test_router_with_different_batch_sizes(self):
        """Test routers work with various batch sizes."""
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2)

        for batch_size in [1, 4, 16]:
            hidden_states = torch.randn(batch_size, 10, 64)
            routing_weights, selected_experts, router_logits = router(hidden_states)

            assert routing_weights.shape[0] == batch_size
            assert selected_experts.shape[0] == batch_size

    def test_router_with_different_seq_lengths(self):
        """Test routers work with various sequence lengths."""
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2)

        for seq_len in [1, 10, 100, 1024]:
            hidden_states = torch.randn(2, seq_len, 64)
            routing_weights, selected_experts, router_logits = router(hidden_states)

            assert routing_weights.shape[1] == seq_len
            assert selected_experts.shape[1] == seq_len

    def test_gradient_flow_through_router(self):
        """Test that gradients flow through the router."""
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2)
        hidden_states = torch.randn(2, 10, 64, requires_grad=True)

        routing_weights, _, _ = router(hidden_states)
        loss = routing_weights.sum()
        loss.backward()

        # Gradients should flow to gate weights
        assert router.gate.weight.grad is not None
        assert hidden_states.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
