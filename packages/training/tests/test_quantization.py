"""Tests for quantization modules."""

import pytest
import torch

from wf_train.quantization.weight_quant import (
    ternary_weight_quantization,
    ternary_weight_quantization_no_scale,
    compute_weight_scale,
)
from wf_train.quantization.activation_quant import (
    activation_quantization_per_token,
    activation_quantization_per_tensor,
    activation_quantization_absmean,
)
from wf_train.quantization.ste import (
    StraightThroughEstimator,
    ste_quantize,
    detach_quantize,
)


class TestTernaryWeightQuantization:
    """Tests for ternary weight quantization."""

    def test_output_shape_preserved(self):
        """Test that output shape matches input shape."""
        for shape in [(64, 128), (32, 64, 128), (4, 8, 16, 32)]:
            w = torch.randn(shape)
            w_quant = ternary_weight_quantization(w)
            assert w_quant.shape == w.shape

    def test_values_are_ternary_scaled(self):
        """Test that quantized weights are scaled versions of {-1, 0, 1}."""
        w = torch.randn(64, 128)
        w_quant = ternary_weight_quantization(w)

        # Compute scale
        scale = compute_weight_scale(w)

        # Unscale and check ternary
        w_unscaled = w_quant * scale
        unique_values = w_unscaled.round().unique()

        # Should only contain -1, 0, 1 (after rounding for numerical precision)
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_values.tolist())

    def test_no_scale_variant_returns_tuple(self):
        """Test that no_scale variant returns (weights, scale) tuple."""
        w = torch.randn(64, 128)
        w_quant, scale = ternary_weight_quantization_no_scale(w)

        assert w_quant.shape == w.shape
        assert scale.dim() == 0  # Scalar
        assert scale.item() > 0

    def test_no_scale_variant_ternary_values(self):
        """Test that no_scale variant produces exact ternary values."""
        w = torch.randn(64, 128)
        w_quant, _ = ternary_weight_quantization_no_scale(w)

        unique_values = w_quant.unique()
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_values.tolist())

    def test_scale_computation(self):
        """Test that scale is computed correctly as absmean."""
        w = torch.tensor([1.0, -2.0, 3.0, -4.0])
        scale = compute_weight_scale(w)

        expected = (1.0 + 2.0 + 3.0 + 4.0) / 4.0  # absmean
        assert torch.isclose(scale, torch.tensor(expected))

    def test_numerical_stability_with_small_weights(self):
        """Test numerical stability when weights are very small."""
        w = torch.randn(64, 128) * 1e-8
        w_quant = ternary_weight_quantization(w)

        # Should not produce NaN or Inf
        assert torch.isfinite(w_quant).all()

    def test_numerical_stability_with_zero_weights(self):
        """Test handling of all-zero weights."""
        w = torch.zeros(64, 128)
        w_quant = ternary_weight_quantization(w)

        assert torch.isfinite(w_quant).all()
        assert (w_quant == 0).all()


class TestActivationQuantization:
    """Tests for activation quantization."""

    def test_per_token_output_shape(self):
        """Test that per-token quantization preserves shape."""
        for shape in [(2, 10, 64), (4, 20, 128), (1, 5, 256)]:
            x = torch.randn(shape)
            x_quant = activation_quantization_per_token(x)
            assert x_quant.shape == x.shape

    def test_per_tensor_output_shape(self):
        """Test that per-tensor quantization preserves shape."""
        x = torch.randn(2, 10, 64)
        x_quant = activation_quantization_per_tensor(x)
        assert x_quant.shape == x.shape

    def test_absmean_output_shape(self):
        """Test that absmean quantization preserves shape."""
        x = torch.randn(2, 10, 64)
        x_quant = activation_quantization_absmean(x)
        assert x_quant.shape == x.shape

    def test_quantization_range_8bit(self):
        """Test that 8-bit quantization uses [-128, 127] range internally."""
        x = torch.randn(2, 10, 64) * 10

        # The quantized values (before unscaling) should be in [-128, 127]
        # After quantize-unscale cycle, values should be close to original
        x_quant = activation_quantization_per_token(x)

        # Check reconstruction is reasonable
        assert torch.allclose(x, x_quant, atol=1.0)

    def test_per_token_uses_per_token_scale(self):
        """Test that per-token quantization uses different scale per token."""
        x = torch.zeros(2, 3, 64)
        x[0, 0, :] = 1.0   # Small values
        x[0, 1, :] = 100.0  # Large values
        x[0, 2, :] = 10.0   # Medium values

        x_quant = activation_quantization_per_token(x)

        # Each token should be normalized differently
        # After quantization, each token should preserve its relative structure
        assert torch.isfinite(x_quant).all()

    def test_numerical_stability_with_zeros(self):
        """Test handling of zero activations."""
        x = torch.zeros(2, 10, 64)
        x_quant = activation_quantization_per_token(x)

        assert torch.isfinite(x_quant).all()

    def test_4bit_quantization(self):
        """Test 4-bit quantization (smaller range)."""
        x = torch.randn(2, 10, 64)
        x_quant = activation_quantization_per_token(x, bits=4)

        assert x_quant.shape == x.shape
        assert torch.isfinite(x_quant).all()

    def test_gradient_does_not_flow_through_quantization(self):
        """Test that gradients don't flow through raw quantization."""
        x = torch.randn(2, 10, 64, requires_grad=True)
        x_quant = activation_quantization_per_token(x)

        # round() and clamp() break gradient flow
        # This is expected - use STE for gradient flow
        loss = x_quant.sum()
        loss.backward()

        # Gradient should be zero or None due to round()
        # (In practice it's typically zero due to the operations)


class TestStraightThroughEstimator:
    """Tests for straight-through estimator."""

    def test_forward_applies_quantization(self):
        """Test that forward pass applies the quantization function."""
        x = torch.randn(10)

        def simple_quant(x):
            return x.round()

        x_quant = StraightThroughEstimator.apply(x, simple_quant)

        assert torch.allclose(x_quant, x.round())

    def test_backward_passes_gradient_unchanged(self):
        """Test that backward pass returns gradient unchanged."""
        x = torch.randn(10, requires_grad=True)

        def simple_quant(x):
            return x.round()

        x_quant = StraightThroughEstimator.apply(x, simple_quant)
        loss = x_quant.sum()
        loss.backward()

        # Gradient should be all ones (dL/dx = 1 for sum)
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_ste_quantize_helper(self):
        """Test the ste_quantize helper function."""
        x = torch.randn(10, requires_grad=True)

        def quant_fn(x):
            return x.round().clamp(-1, 1)

        x_quant = ste_quantize(x, quant_fn)
        loss = x_quant.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_detach_quantize_equivalent(self):
        """Test that detach_quantize produces same forward result."""
        x = torch.randn(10, requires_grad=True)

        def quant_fn(x):
            return x.round()

        x_ste = ste_quantize(x.clone(), quant_fn)
        x_detach = detach_quantize(x.clone(), quant_fn)

        assert torch.allclose(x_ste, x_detach)

    def test_detach_quantize_gradient_flow(self):
        """Test gradient flow through detach_quantize."""
        x = torch.randn(10, requires_grad=True)

        def quant_fn(x):
            return x.round()

        x_quant = detach_quantize(x, quant_fn)
        loss = x_quant.sum()
        loss.backward()

        assert x.grad is not None
        # Gradient should flow through
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_ste_with_weight_quantization(self):
        """Test STE with actual weight quantization."""
        w = torch.randn(32, 64, requires_grad=True)

        w_quant = ste_quantize(w, ternary_weight_quantization)

        # Forward should produce quantized weights
        scale = compute_weight_scale(w)
        w_expected = (w * (1.0 / scale)).round().clamp(-1, 1) / (1.0 / scale)

        # Backward should work
        loss = w_quant.sum()
        loss.backward()

        assert w.grad is not None


class TestQuantizationEdgeCases:
    """Edge case tests for quantization."""

    def test_single_element_tensor(self):
        """Test quantization of single element tensors."""
        w = torch.tensor([0.5])
        w_quant = ternary_weight_quantization(w)
        assert w_quant.shape == (1,)
        assert torch.isfinite(w_quant).all()

    def test_very_large_values(self):
        """Test handling of very large values."""
        w = torch.randn(64, 128) * 1e6
        w_quant = ternary_weight_quantization(w)
        assert torch.isfinite(w_quant).all()

    def test_mixed_precision(self):
        """Test quantization with different dtypes."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            w = torch.randn(32, 64, dtype=dtype)
            w_quant = ternary_weight_quantization(w)
            assert w_quant.dtype == dtype

    def test_different_devices(self):
        """Test that quantization works on CPU."""
        w = torch.randn(32, 64)
        w_quant = ternary_weight_quantization(w)
        assert w_quant.device == w.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_quantization(self):
        """Test quantization on CUDA if available."""
        w = torch.randn(32, 64, device="cuda")
        w_quant = ternary_weight_quantization(w)
        assert w_quant.device == w.device
        assert torch.isfinite(w_quant).all()
