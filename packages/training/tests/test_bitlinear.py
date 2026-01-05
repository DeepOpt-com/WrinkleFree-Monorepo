"""Tests for BitLinear layer."""

import pytest
import torch

from wf_train.models import BitLinear


class TestBitLinear:
    """Tests for BitLinear quantized linear layer."""

    def test_forward_shape(self):
        """Test that forward pass produces correct shape."""
        layer = BitLinear(in_features=64, out_features=128)
        x = torch.randn(2, 10, 64)

        output = layer(x)

        assert output.shape == (2, 10, 128)

    def test_weight_quantization(self):
        """Test that weights are quantized to ternary values."""
        layer = BitLinear(in_features=64, out_features=128)
        w = layer.weight

        w_quant = layer.weight_quant(w)

        # Check that quantized weights are scaled versions of {-1, 0, 1}
        scale = 1.0 / w.abs().mean()
        w_scaled = (w_quant * scale).round()

        assert torch.all(w_scaled.abs() <= 1.0)

    def test_activation_quantization(self):
        """Test that activations are quantized to 8-bit."""
        layer = BitLinear(in_features=64, out_features=128)
        x = torch.randn(2, 10, 64)

        x_quant = layer.activation_quant(x)

        # Check that quantized activations are within expected range
        # After unscaling, should be close to original
        assert x_quant.shape == x.shape
        assert torch.allclose(x_quant, x, atol=0.1)

    def test_gradient_flow(self):
        """Test that gradients flow through STE."""
        layer = BitLinear(in_features=64, out_features=128)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert x.grad is not None

    def test_no_bias_by_default(self):
        """Test that bias is disabled by default."""
        layer = BitLinear(in_features=64, out_features=128)

        assert layer.bias is None

    def test_with_bias(self):
        """Test that bias can be enabled."""
        layer = BitLinear(in_features=64, out_features=128, bias=True)

        assert layer.bias is not None
        assert layer.bias.shape == (128,)
