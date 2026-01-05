"""Tests for SubLN normalization module."""

import pytest
import torch

from wf_train.models import SubLN, RMSNorm


class TestSubLN:
    """Tests for Sub-Layer Normalization."""

    def test_forward_shape(self):
        """Test that forward pass preserves shape."""
        norm = SubLN(hidden_size=64)
        x = torch.randn(2, 10, 64)

        output = norm(x)

        assert output.shape == x.shape

    def test_normalization(self):
        """Test that output has approximately unit variance."""
        norm = SubLN(hidden_size=64)
        x = torch.randn(2, 10, 64) * 10  # Large variance input

        output = norm(x)

        # RMSNorm: output = x / sqrt(mean(x^2))
        # So variance should be approximately 1
        variance = output.pow(2).mean(dim=-1)
        assert torch.allclose(variance, torch.ones_like(variance), atol=0.2)

    def test_elementwise_affine(self):
        """Test that elementwise affine parameter works."""
        norm_affine = SubLN(hidden_size=64, elementwise_affine=True)
        norm_no_affine = SubLN(hidden_size=64, elementwise_affine=False)

        x = torch.randn(2, 10, 64)

        output_affine = norm_affine(x)
        output_no_affine = norm_no_affine(x)

        # With default initialization (ones), outputs should be the same
        assert torch.allclose(output_affine, output_no_affine, atol=1e-6)

        # Change weight
        norm_affine.weight.data.fill_(2.0)
        output_scaled = norm_affine(x)

        # Output should be scaled
        assert torch.allclose(output_scaled, 2 * output_no_affine, atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow through normalization."""
        norm = SubLN(hidden_size=64)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None


class TestRMSNorm:
    """Tests for RMS normalization."""

    def test_forward_shape(self):
        """Test that forward pass preserves shape."""
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64)

        output = norm(x)

        assert output.shape == x.shape

    def test_weight_parameter(self):
        """Test that weight parameter exists."""
        norm = RMSNorm(hidden_size=64)

        assert norm.weight is not None
        assert norm.weight.shape == (64,)
