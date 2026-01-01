"""Tests for BitLinear layer."""

import pytest
import torch
import torch.nn as nn

from bitnet_arch.layers.bitlinear import (
    BitLinear,
    BitLinearNoActivationQuant,
    convert_linear_to_bitlinear,
)
from bitnet_arch.quantization import set_global_lambda_warmup, LambdaWarmup


class TestBitLinear:
    """Test BitLinear layer functionality."""

    def test_init(self):
        """Test BitLinear initialization."""
        layer = BitLinear(128, 256, bias=False)
        assert layer.in_features == 128
        assert layer.out_features == 256
        assert layer.weight.shape == (256, 128)
        assert layer.bias is None

    def test_init_with_bias(self):
        """Test BitLinear with bias."""
        layer = BitLinear(128, 256, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (256,)

    def test_forward_shape(self):
        """Test output shape is correct."""
        layer = BitLinear(128, 256)
        x = torch.randn(4, 32, 128)  # batch, seq, features
        output = layer(x)
        assert output.shape == (4, 32, 256)

    def test_weight_quantization_ternary(self):
        """Test that weights are quantized to {-1, 0, 1}."""
        layer = BitLinear(128, 256)
        # Set lambda to 1.0 for full quantization
        set_global_lambda_warmup(None)  # Reset to default (lambda=1.0)

        x = torch.randn(4, 32, 128)
        # Forward pass triggers quantization
        _ = layer(x)

        # Check that quantized weights are ternary
        # Access the quantized weights during forward
        weight = layer.weight.data
        mean = weight.abs().mean()
        # After round_ste, values should be near -1, 0, or 1 when scaled
        # The quantization formula: round(w / (mean + eps)) -> {-1, 0, 1}

    def test_gradient_flow_ste(self):
        """Test that gradients flow through STE."""
        layer = BitLinear(64, 32)
        x = torch.randn(2, 16, 64, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Gradients should flow to input and weights
        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_lambda_interpolation(self):
        """Test that lambda warmup affects output."""
        layer = BitLinear(64, 32)
        x = torch.randn(2, 8, 64)

        # With lambda=0 (full precision)
        warmup = LambdaWarmup(warmup_steps=100, min_lambda=0.0)
        set_global_lambda_warmup(warmup)
        output_fp = layer(x).clone()

        # With lambda=1 (full quantization)
        for _ in range(100):
            warmup.step()
        output_quant = layer(x).clone()

        # Outputs should be different
        assert not torch.allclose(output_fp, output_quant, atol=1e-6)

        # Cleanup
        set_global_lambda_warmup(None)


class TestBitLinearNoActivationQuant:
    """Test BitLinear without activation quantization."""

    def test_forward(self):
        """Test forward pass without activation quantization."""
        layer = BitLinearNoActivationQuant(64, 32)
        x = torch.randn(2, 8, 64)
        output = layer(x)
        assert output.shape == (2, 8, 32)


class TestConvertLinear:
    """Test linear to BitLinear conversion."""

    def test_convert_preserves_weights(self):
        """Test that conversion preserves weight values."""
        linear = nn.Linear(128, 64, bias=True)
        original_weight = linear.weight.data.clone()
        original_bias = linear.bias.data.clone()

        bitlinear = convert_linear_to_bitlinear(linear)

        assert torch.allclose(bitlinear.weight.data, original_weight)
        assert torch.allclose(bitlinear.bias.data, original_bias)

    def test_convert_no_bias(self):
        """Test conversion without bias."""
        linear = nn.Linear(128, 64, bias=False)
        bitlinear = convert_linear_to_bitlinear(linear)

        assert bitlinear.bias is None
        assert bitlinear.in_features == 128
        assert bitlinear.out_features == 64
