"""Tests for SubLN layer."""

import pytest
import torch

from bitnet_arch.layers.subln import SubLN, RMSNorm


class TestSubLN:
    """Test SubLN (Sub-Layer Normalization) functionality."""

    def test_init(self):
        """Test SubLN initialization."""
        subln = SubLN(256)
        assert subln.hidden_size == 256
        assert subln.weight.shape == (256,)
        assert subln.elementwise_affine is True

    def test_init_no_affine(self):
        """Test SubLN without elementwise affine."""
        subln = SubLN(256, elementwise_affine=False)
        assert subln.weight is None

    def test_rmsnorm_computation(self):
        """Test RMSNorm computation is correct."""
        subln = SubLN(64, elementwise_affine=False)
        x = torch.randn(2, 8, 64)

        output = subln(x)

        # Manually compute RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        expected = x * torch.rsqrt(variance + subln.eps)

        assert torch.allclose(output, expected, atol=1e-5)

    def test_forward_shape(self):
        """Test output shape is correct."""
        subln = SubLN(128)
        x = torch.randn(4, 32, 128)
        output = subln(x)
        assert output.shape == x.shape

    def test_normalization_effect(self):
        """Test that normalization reduces variance differences."""
        subln = SubLN(64, elementwise_affine=False)

        # Create input with varying magnitudes
        x = torch.randn(2, 8, 64) * torch.tensor([0.1, 10.0]).view(2, 1, 1)

        output = subln(x)

        # After RMSNorm, the RMS of each vector should be ~1
        rms = output.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_gradient_flow(self):
        """Test that gradients flow through SubLN."""
        subln = SubLN(64)
        x = torch.randn(2, 8, 64, requires_grad=True)

        output = subln(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert subln.weight.grad is not None


class TestRMSNorm:
    """Test RMSNorm functionality."""

    def test_init(self):
        """Test RMSNorm initialization."""
        norm = RMSNorm(256)
        assert norm.hidden_size == 256
        assert norm.weight.shape == (256,)

    def test_forward(self):
        """Test RMSNorm forward pass."""
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        output = norm(x)
        assert output.shape == x.shape

    def test_extra_repr(self):
        """Test string representation."""
        norm = RMSNorm(256, eps=1e-5)
        repr_str = norm.extra_repr()
        assert "256" in repr_str
        assert "1e-05" in repr_str
