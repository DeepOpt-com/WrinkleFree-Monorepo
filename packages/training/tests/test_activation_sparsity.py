"""Tests for Q-Sparse activation sparsity.

Reference: arxiv.org/abs/2407.10969
"""

import pytest
import torch

from wrinklefree.quantization.activation_sparse import (
    TopKSparsity,
    block_sparsify_nm,
    detach_sparsify,
    topk_sparsify,
)
from wrinklefree.quantization.sparsity_warmup import (
    SparsityWarmup,
    get_current_sparsity,
    get_global_sparsity_warmup,
    set_global_sparsity_warmup,
)


class TestTopKSparsity:
    """Tests for top-k activation sparsification."""

    def test_sparsity_ratio_50_percent(self):
        """Test that 50% sparsity zeros out roughly half the elements."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, 64)
        x_sparse = topk_sparsify(x, sparsity_ratio=0.5)

        # ~50% should be zero (per token)
        zero_ratio = (x_sparse == 0).float().mean().item()
        assert 0.45 < zero_ratio < 0.55, f"Expected ~50% zeros, got {zero_ratio:.2%}"

    def test_sparsity_ratio_61_percent(self):
        """Test the paper's recommended 61% sparsity for 1-bit models."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, 64)
        x_sparse = topk_sparsify(x, sparsity_ratio=0.61)

        # ~61% should be zero
        zero_ratio = (x_sparse == 0).float().mean().item()
        assert 0.55 < zero_ratio < 0.67, f"Expected ~61% zeros, got {zero_ratio:.2%}"

    def test_gradient_flow_ste(self):
        """Test that STE allows gradients to flow through."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, 64, requires_grad=True)
        x_sparse = topk_sparsify(x, sparsity_ratio=0.5)
        loss = x_sparse.sum()
        loss.backward()

        # Gradient should flow to ALL elements (STE)
        assert x.grad is not None, "Gradient should not be None"
        # All gradients should be non-zero (STE passes through)
        assert (x.grad != 0).all(), "STE should allow gradients to all elements"

    def test_zero_sparsity_is_identity(self):
        """Test that zero sparsity returns input unchanged."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, 64)
        x_sparse = topk_sparsify(x, sparsity_ratio=0.0)
        assert torch.allclose(x, x_sparse), "Zero sparsity should return input unchanged"

    def test_full_sparsity_is_zeros(self):
        """Test that 100% sparsity returns all zeros."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, 64)
        x_sparse = topk_sparsify(x, sparsity_ratio=1.0)
        assert (x_sparse == 0).all(), "100% sparsity should zero everything"

    def test_top_k_keeps_largest_magnitudes(self):
        """Test that top-k keeps elements with largest magnitudes."""
        # Create tensor where we know what top-k should be
        x = torch.tensor([[1.0, -5.0, 2.0, -3.0, 0.5]])  # (1, 5)
        # With 40% sparsity (keep 60% = 3 elements), should keep -5, -3, 2
        x_sparse = topk_sparsify(x, sparsity_ratio=0.4)

        # Check that the largest magnitude elements are kept
        assert x_sparse[0, 1].item() == -5.0  # Largest magnitude
        assert x_sparse[0, 3].item() == -3.0  # Second largest
        assert x_sparse[0, 2].item() == 2.0  # Third largest

    def test_per_token_sparsity(self):
        """Test that sparsity is applied per-token."""
        torch.manual_seed(42)
        x = torch.randn(2, 3, 10)  # 2 batches, 3 tokens, 10 features

        # With 50% sparsity, each token should have ~5 non-zeros
        x_sparse = topk_sparsify(x, sparsity_ratio=0.5, per_token=True)

        for b in range(2):
            for t in range(3):
                nonzero_count = (x_sparse[b, t] != 0).sum().item()
                assert nonzero_count == 5, f"Expected 5 non-zeros per token, got {nonzero_count}"


class TestBlockSparsity:
    """Tests for N:M structured block sparsity."""

    def test_nm_2_4_pattern(self):
        """Test 2:4 structured sparsity pattern."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, 64)  # D=64 divisible by M=4
        x_sparse = block_sparsify_nm(x, n=2, m=4)

        # Check each block of 4 has exactly 2 non-zeros
        x_blocked = x_sparse.view(4, 16, 16, 4)  # Reshape to blocks
        nonzero_per_block = (x_blocked != 0).float().sum(dim=-1)
        assert (nonzero_per_block == 2).all(), "Each block of 4 should have exactly 2 non-zeros"

    def test_nm_1_2_pattern(self):
        """Test 1:2 structured sparsity (50% sparse)."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 16)
        x_sparse = block_sparsify_nm(x, n=1, m=2)

        # Check each block of 2 has exactly 1 non-zero
        x_blocked = x_sparse.view(2, 8, 8, 2)
        nonzero_per_block = (x_blocked != 0).float().sum(dim=-1)
        assert (nonzero_per_block == 1).all(), "Each block of 2 should have exactly 1 non-zero"

    def test_handles_non_divisible_dimensions(self):
        """Test that block sparsity handles non-divisible dimensions."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 10)  # D=10 not divisible by M=4
        x_sparse = block_sparsify_nm(x, n=2, m=4)

        # Should still work and preserve shape
        assert x_sparse.shape == x.shape, "Shape should be preserved"

    def test_n_equals_m_is_identity(self):
        """Test that N=M means no sparsification."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 8)
        x_sparse = block_sparsify_nm(x, n=4, m=4)
        assert torch.allclose(x, x_sparse), "N=M should return input unchanged"


class TestDetachSparsify:
    """Tests for the detach trick STE variant."""

    def test_detach_sparsify_same_forward_as_topk(self):
        """Test that detach_sparsify produces same forward output as topk_sparsify."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, 64)
        x_topk = topk_sparsify(x, sparsity_ratio=0.5)
        x_detach = detach_sparsify(x, sparsity_ratio=0.5)

        # Forward should be identical
        assert torch.allclose(x_topk, x_detach), "Forward outputs should match"

    def test_detach_sparsify_gradient_flow(self):
        """Test gradient flow with detach trick."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, 64, requires_grad=True)
        x_sparse = detach_sparsify(x, sparsity_ratio=0.5)
        loss = x_sparse.sum()
        loss.backward()

        # Gradients should flow to ALL elements
        assert x.grad is not None
        assert (x.grad != 0).all(), "Detach trick should allow gradients to all elements"


class TestSparsityWarmup:
    """Tests for sparsity warmup schedule."""

    def test_linear_schedule(self):
        """Test linear warmup schedule."""
        warmup = SparsityWarmup(
            warmup_steps=100,
            schedule="linear",
            initial_sparsity=0.0,
            target_sparsity=0.5,
        )

        assert warmup.sparsity == 0.0, "Initial sparsity should be 0"

        for _ in range(50):
            warmup.step()

        # At halfway, should be ~0.25
        assert 0.24 < warmup.sparsity < 0.26, f"Expected ~0.25 at halfway, got {warmup.sparsity}"

        for _ in range(50):
            warmup.step()

        # After warmup, should be at target
        assert warmup.sparsity == 0.5, f"Expected 0.5 after warmup, got {warmup.sparsity}"

    def test_cosine_schedule(self):
        """Test cosine warmup schedule."""
        warmup = SparsityWarmup(
            warmup_steps=100,
            schedule="cosine",
            initial_sparsity=0.0,
            target_sparsity=0.6,
        )

        assert warmup.sparsity == 0.0

        for _ in range(50):
            warmup.step()

        # Cosine at halfway should be exactly 0.3 (midpoint)
        assert 0.29 < warmup.sparsity < 0.31, f"Expected ~0.3 at halfway, got {warmup.sparsity}"

        for _ in range(50):
            warmup.step()

        assert warmup.sparsity == 0.6

    def test_global_sparsity_accessor(self):
        """Test global sparsity accessor functions."""
        # Initially should be None and return 0.0
        set_global_sparsity_warmup(None)
        assert get_current_sparsity() == 0.0, "No warmup should return 0.0"

        warmup = SparsityWarmup(
            warmup_steps=10,
            target_sparsity=0.6,
        )
        set_global_sparsity_warmup(warmup)

        assert get_current_sparsity() == 0.0  # Initial

        for _ in range(10):
            warmup.step()

        assert get_current_sparsity() == 0.6  # After warmup

        # Cleanup
        set_global_sparsity_warmup(None)

    def test_state_dict_save_load(self):
        """Test checkpointing with state_dict."""
        warmup = SparsityWarmup(
            warmup_steps=100,
            schedule="cosine",
            target_sparsity=0.61,
        )

        for _ in range(50):
            warmup.step()

        # Save state
        state = warmup.state_dict()

        # Create new warmup and load state
        warmup2 = SparsityWarmup()
        warmup2.load_state_dict(state)

        assert warmup2.current_step == warmup.current_step
        assert warmup2.sparsity == warmup.sparsity
        assert warmup2.target_sparsity == warmup.target_sparsity

    def test_warmup_complete_flag(self):
        """Test is_warmup_complete flag."""
        warmup = SparsityWarmup(warmup_steps=5, target_sparsity=0.5)

        assert not warmup.is_warmup_complete()

        for _ in range(5):
            warmup.step()

        assert warmup.is_warmup_complete()


class TestBitLinearIntegration:
    """Integration tests for BitLinear with sparsity."""

    def test_bitlinear_with_sparsity(self):
        """Test BitLinear forward pass with sparsity enabled."""
        from wrinklefree.models.bitlinear import BitLinear

        # Enable sparsity
        warmup = SparsityWarmup(warmup_steps=0, initial_sparsity=0.5, target_sparsity=0.5)
        set_global_sparsity_warmup(warmup)

        layer = BitLinear(64, 32)
        x = torch.randn(2, 4, 64)

        # Forward should work
        y = layer(x)
        assert y.shape == (2, 4, 32), f"Expected shape (2, 4, 32), got {y.shape}"

        # Cleanup
        set_global_sparsity_warmup(None)

    def test_bitlinear_without_sparsity(self):
        """Test BitLinear forward pass with sparsity disabled."""
        from wrinklefree.models.bitlinear import BitLinear

        # Ensure sparsity is disabled
        set_global_sparsity_warmup(None)

        layer = BitLinear(64, 32)
        x = torch.randn(2, 4, 64)

        y = layer(x)
        assert y.shape == (2, 4, 32)

    def test_gradient_flow_through_bitlinear_with_sparsity(self):
        """Test that gradients flow correctly through BitLinear with sparsity."""
        from wrinklefree.models.bitlinear import BitLinear

        warmup = SparsityWarmup(warmup_steps=0, initial_sparsity=0.5, target_sparsity=0.5)
        set_global_sparsity_warmup(warmup)

        layer = BitLinear(64, 32)
        x = torch.randn(2, 4, 64, requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Gradients should flow to input
        assert x.grad is not None, "Input gradient should not be None"
        assert (x.grad != 0).any(), "Input should have non-zero gradients"

        # Gradients should flow to weights
        assert layer.weight.grad is not None, "Weight gradient should not be None"
        assert (layer.weight.grad != 0).any(), "Weights should have non-zero gradients"

        # Cleanup
        set_global_sparsity_warmup(None)


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Ensure global state is cleaned up after each test."""
    yield
    set_global_sparsity_warmup(None)
