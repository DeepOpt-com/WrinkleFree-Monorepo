"""Tests for BitLinearLRC layer."""

import pytest
import torch
import torch.nn as nn

from wf_arch.layers.bitlinear import BitLinear
from wf_arch.layers.bitlinear_lrc import (
    BitLinearLRC,
    convert_bitlinear_to_lrc,
    freeze_model_except_lrc,
    get_lrc_stats,
)


class TestBitLinearLRC:
    """Test BitLinearLRC layer functionality."""

    def test_init_default_rank(self):
        """Test default rank calculation (10% of min dimension)."""
        layer = BitLinearLRC(128, 256)
        # rank = 10% of min(128, 256) = 12.8 -> 12
        assert layer.rank == 12
        assert layer.lrc_U.shape == (256, 12)
        assert layer.lrc_V.shape == (128, 12)

    def test_init_explicit_rank(self):
        """Test explicit rank specification."""
        layer = BitLinearLRC(128, 256, rank=32)
        assert layer.rank == 32
        assert layer.lrc_U.shape == (256, 32)
        assert layer.lrc_V.shape == (128, 32)

    def test_init_custom_rank_percentage(self):
        """Test custom rank percentage."""
        layer = BitLinearLRC(100, 200, rank_percentage=0.2)
        # rank = 20% of min(100, 200) = 20
        assert layer.rank == 20

    def test_forward_shape(self):
        """Test output shape is correct."""
        layer = BitLinearLRC(128, 256)
        x = torch.randn(4, 32, 128)
        output = layer(x)
        assert output.shape == (4, 32, 256)

    def test_only_lrc_params_trainable(self):
        """CRITICAL: Verify ONLY U, V are trainable."""
        layer = BitLinearLRC(128, 256)

        trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
        frozen = [n for n, p in layer.named_parameters() if not p.requires_grad]

        assert set(trainable) == {"lrc_U", "lrc_V"}
        assert "weight" in frozen

    def test_weight_frozen(self):
        """Test that weight.requires_grad is False."""
        layer = BitLinearLRC(128, 256)
        assert not layer.weight.requires_grad

    def test_bias_frozen_when_present(self):
        """Test that bias.requires_grad is False when bias exists."""
        layer = BitLinearLRC(128, 256, bias=True)
        assert layer.bias is not None
        assert not layer.bias.requires_grad

    def test_gradient_flow_to_lrc_only(self):
        """Test gradients flow to U, V but NOT to frozen params."""
        layer = BitLinearLRC(64, 32)
        x = torch.randn(2, 8, 64, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Gradients should flow to LRC params
        assert layer.lrc_U.grad is not None
        assert layer.lrc_V.grad is not None
        # But NOT to frozen weights
        assert layer.weight.grad is None

    def test_lrc_contribution(self):
        """Test that LRC actually modifies output."""
        layer = BitLinearLRC(64, 32, rank=8)
        x = torch.randn(2, 8, 64)

        # Get output with zero LRC
        layer.lrc_U.data.zero_()
        layer.lrc_V.data.zero_()
        output_zero = layer(x).clone()

        # Set non-zero LRC
        layer.lrc_U.data.normal_(std=0.1)
        layer.lrc_V.data.normal_(std=0.1)
        output_lrc = layer(x).clone()

        # Outputs should differ
        assert not torch.allclose(output_zero, output_lrc, atol=1e-6)

    def test_lrc_on_unquantized_activations(self):
        """Test that LRC uses unquantized activations (linearity check)."""
        layer = BitLinearLRC(64, 32, rank=8)
        layer.lrc_U.data.normal_()
        layer.lrc_V.data.normal_()

        # Create input with known values
        x1 = torch.randn(2, 8, 64)
        x2 = torch.randn(2, 8, 64)

        # The LRC component should be linear in x
        # LRC(x1 + x2) = LRC(x1) + LRC(x2)
        # We can't test this directly because of quantization, but we can verify
        # LRC uses the original x by checking that scaling x affects LRC output linearly

        output_1x = layer(x1).clone()
        # Zero out quantized path to isolate LRC
        with torch.no_grad():
            layer.weight.zero_()
        output_lrc_only = layer(x1)

        # LRC should be linear in x
        output_2x = layer(2 * x1)
        assert torch.allclose(output_2x, 2 * output_lrc_only, atol=1e-5)

    def test_svd_init(self):
        """Test SVD-based initialization."""
        layer = BitLinearLRC(64, 32, rank=8)
        original_weight = torch.randn(32, 64)

        layer.init_lrc_from_svd(original_weight)

        # LRC matrices should be non-zero after SVD init
        assert not torch.allclose(layer.lrc_U, torch.zeros_like(layer.lrc_U))
        assert not torch.allclose(layer.lrc_V, torch.zeros_like(layer.lrc_V))

    def test_svd_init_approximates_residual(self):
        """Test that SVD init approximates the quantization residual."""
        layer = BitLinearLRC(32, 16, rank=8)
        original_weight = torch.randn(16, 32)

        # Copy original weight to layer
        layer.weight.data.copy_(original_weight)

        # Initialize LRC from SVD of residual
        layer.init_lrc_from_svd(original_weight)

        # Compute what LRC approximates: W - W_quant
        w_quant = layer.weight_quant(original_weight)
        residual = original_weight - w_quant

        # LRC approximation: U @ V^T
        lrc_approx = layer.lrc_U @ layer.lrc_V.t()

        # LRC should approximate the residual (not perfectly, but captures main components)
        # Check that Frobenius norm of error is less than residual norm
        approx_error = (residual - lrc_approx).norm()
        residual_norm = residual.norm()
        assert approx_error < residual_norm, "SVD init should reduce residual error"

    def test_extra_repr(self):
        """Test string representation."""
        layer = BitLinearLRC(128, 256, rank=16)
        repr_str = layer.extra_repr()
        assert "128" in repr_str
        assert "256" in repr_str
        assert "16" in repr_str

    def test_mixed_precision(self):
        """Test layer works with mixed precision (fp16/bf16)."""
        layer = BitLinearLRC(64, 32).cuda() if torch.cuda.is_available() else BitLinearLRC(64, 32)
        x = torch.randn(2, 8, 64, dtype=torch.float16)
        if torch.cuda.is_available():
            x = x.cuda()
            layer = layer.cuda()

        output = layer(x)
        assert output.dtype == torch.float16


class TestConvertBitLinearToLRC:
    """Test conversion utilities."""

    def test_convert_preserves_weights(self):
        """Test conversion preserves original weights."""
        linear = BitLinear(128, 64)
        original_weight = linear.weight.data.clone()

        model = nn.Sequential(linear)
        model = convert_bitlinear_to_lrc(model, rank_percentage=0.1)

        lrc_layer = model[0]
        assert isinstance(lrc_layer, BitLinearLRC)
        assert torch.allclose(lrc_layer.weight.data, original_weight)

    def test_convert_freezes_weights(self):
        """Test that converted layers have frozen weights."""
        linear = BitLinear(128, 64)
        model = nn.Sequential(linear)
        model = convert_bitlinear_to_lrc(model)

        lrc_layer = model[0]
        assert not lrc_layer.weight.requires_grad

    def test_convert_with_explicit_rank(self):
        """Test conversion with explicit rank."""
        linear = BitLinear(128, 64)
        model = nn.Sequential(linear)
        model = convert_bitlinear_to_lrc(model, rank=16)

        lrc_layer = model[0]
        assert lrc_layer.rank == 16

    def test_convert_with_svd_init(self):
        """Test conversion with SVD initialization."""
        linear = BitLinear(64, 32)
        linear.weight.data.normal_()
        model = nn.Sequential(linear)

        model = convert_bitlinear_to_lrc(model, init_method="svd_residual")

        lrc_layer = model[0]
        # SVD init should produce non-zero LRC matrices
        assert not torch.allclose(lrc_layer.lrc_U, torch.zeros_like(lrc_layer.lrc_U))

    def test_exclude_layers(self):
        """Test layer exclusion."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = BitLinear(64, 64)
                self.layer2 = BitLinear(64, 64)

        model = Model()
        model = convert_bitlinear_to_lrc(model, exclude_names=["layer1"])

        assert isinstance(model.layer1, BitLinear)
        assert not isinstance(model.layer1, BitLinearLRC)
        assert isinstance(model.layer2, BitLinearLRC)

    def test_recursive_conversion(self):
        """Test conversion works recursively through nested modules."""

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(
                    BitLinear(64, 64),
                    nn.ReLU(),
                    BitLinear(64, 32),
                )

        model = NestedModel()
        model = convert_bitlinear_to_lrc(model)

        assert isinstance(model.block[0], BitLinearLRC)
        assert isinstance(model.block[2], BitLinearLRC)


class TestFreezeModelExceptLRC:
    """Test model freezing utility."""

    def test_freeze_only_lrc_trainable(self):
        """Test that only LRC params remain trainable."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 64)
                self.layer = BitLinearLRC(64, 64)
                self.norm = nn.LayerNorm(64)

        model = Model()
        stats = freeze_model_except_lrc(model)

        # Only LRC params should be trainable
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert all("lrc_U" in n or "lrc_V" in n for n in trainable)

        # Embedding and norm should be frozen
        assert not model.embedding.weight.requires_grad
        assert not model.norm.weight.requires_grad

    def test_freeze_returns_stats(self):
        """Test that freeze returns correct statistics."""
        layer = BitLinearLRC(128, 64, rank=16)
        model = nn.Sequential(layer)

        stats = freeze_model_except_lrc(model)

        expected_lrc_params = 128 * 16 + 64 * 16  # V + U
        expected_frozen = 128 * 64 * 2  # weight + weight_quantized

        assert stats["trainable"] == expected_lrc_params
        assert stats["frozen"] == expected_frozen


class TestGetLRCStats:
    """Test LRC statistics utility."""

    def test_stats_single_layer(self):
        """Test stats for single LRC layer."""
        layer = BitLinearLRC(128, 64, rank=16)
        model = nn.Sequential(layer)

        stats = get_lrc_stats(model)

        assert stats["num_lrc_layers"] == 1
        assert stats["total_lrc_params"] == 128 * 16 + 64 * 16
        assert stats["average_rank"] == 16
        assert stats["ranks"] == [16]

    def test_stats_multiple_layers(self):
        """Test stats for multiple LRC layers."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = BitLinearLRC(128, 64, rank=16)
                self.layer2 = BitLinearLRC(64, 32, rank=8)

        model = Model()
        stats = get_lrc_stats(model)

        assert stats["num_lrc_layers"] == 2
        assert stats["average_rank"] == 12  # (16 + 8) / 2
        assert set(stats["ranks"]) == {16, 8}


class TestPrecomputedQuantizedWeights:
    """Test pre-computed quantized weights functionality."""

    def test_weight_quantized_exists(self):
        """Test that weight_quantized is created and is a parameter."""
        layer = BitLinearLRC(128, 64, rank=16)
        assert hasattr(layer, "weight_quantized")
        assert isinstance(layer.weight_quantized, nn.Parameter)
        assert not layer.weight_quantized.requires_grad

    def test_weight_quantized_matches_weight_quant(self):
        """Test that weight_quantized equals weight_quant(weight)."""
        layer = BitLinearLRC(128, 64, rank=16)
        expected = layer.weight_quant(layer.weight)
        assert torch.allclose(layer.weight_quantized, expected)

    def test_weight_quantized_shape(self):
        """Test weight_quantized has same shape as weight."""
        layer = BitLinearLRC(128, 64, rank=16)
        assert layer.weight_quantized.shape == layer.weight.shape

    def test_checkpoint_save_load_preserves_weight_quantized(self):
        """Test that weight_quantized persists through save/load."""
        import tempfile

        layer = BitLinearLRC(64, 32, rank=8)
        original_wq = layer.weight_quantized.clone()

        # Save state dict
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(layer.state_dict(), f.name)

            # Create new layer and load
            layer2 = BitLinearLRC(64, 32, rank=8)
            layer2.load_state_dict(torch.load(f.name, weights_only=True))

        assert torch.allclose(layer2.weight_quantized, original_wq)

    def test_forward_uses_cached_weights(self):
        """Test that forward uses weight_quantized, not recomputed."""
        layer = BitLinearLRC(64, 32, rank=8)
        x = torch.randn(2, 8, 64)

        # Get normal output
        output1 = layer(x).clone()

        # Modify weight_quantized directly (should affect output)
        with torch.no_grad():
            layer.weight_quantized.add_(0.1)

        output2 = layer(x)

        # Outputs should differ because we changed weight_quantized
        assert not torch.allclose(output1, output2)

    def test_keep_original_weight_true_by_default(self):
        """Test that original weight is kept by default."""
        layer = BitLinearLRC(64, 32, rank=8)
        assert layer.keep_original_weight is True
        assert layer.weight.numel() > 0

    def test_keep_original_weight_false_deletes_weight(self):
        """Test that weight is deleted when keep_original_weight=False."""
        layer = BitLinearLRC(64, 32, rank=8, keep_original_weight=False)
        assert layer.weight.numel() == 0  # Weight should be empty tensor
        # But weight_quantized should still exist
        assert layer.weight_quantized.numel() == 64 * 32

    def test_forward_works_without_original_weight(self):
        """Test forward pass works when original weight is deleted."""
        layer = BitLinearLRC(64, 32, rank=8, keep_original_weight=False)
        x = torch.randn(2, 8, 64)

        # Should not raise
        output = layer(x)
        assert output.shape == (2, 8, 32)

    def test_convert_with_keep_original_weight_false(self):
        """Test conversion with keep_original_weight=False."""
        linear = BitLinear(128, 64)
        original_weight = linear.weight.data.clone()
        model = nn.Sequential(linear)

        model = convert_bitlinear_to_lrc(model, keep_original_weight=False)

        lrc_layer = model[0]
        assert isinstance(lrc_layer, BitLinearLRC)
        # Original weight should be deleted
        assert lrc_layer.weight.numel() == 0
        # But quantized weights should match what we'd expect
        expected_wq = lrc_layer.weight_quant(original_weight)
        assert torch.allclose(lrc_layer.weight_quantized, expected_wq)

    def test_compute_quantized_weights_updates_cache(self):
        """Test that compute_quantized_weights updates the cache."""
        layer = BitLinearLRC(64, 32, rank=8)

        # Modify the weight
        with torch.no_grad():
            layer.weight.data.fill_(1.0)

        # Cache should be stale now
        old_wq = layer.weight_quantized.clone()

        # Recompute
        layer.compute_quantized_weights()

        # Cache should be updated
        expected = layer.weight_quant(layer.weight)
        assert torch.allclose(layer.weight_quantized, expected)
        assert not torch.allclose(layer.weight_quantized, old_wq)

    def test_weight_quantized_in_state_dict(self):
        """Test that weight_quantized is included in state_dict."""
        layer = BitLinearLRC(64, 32, rank=8)
        state_dict = layer.state_dict()
        assert "weight_quantized" in state_dict

    def test_stats_uses_weight_quantized_for_frozen_count(self):
        """Test get_lrc_stats uses weight_quantized for frozen param count."""
        layer = BitLinearLRC(128, 64, rank=16, keep_original_weight=False)
        model = nn.Sequential(layer)

        stats = get_lrc_stats(model)

        # Should count weight_quantized (128*64), not the empty weight
        assert stats["total_frozen_params"] == 128 * 64


class TestTrainableWeight:
    """Test trainable_weight option for STE gradient flow."""

    def test_trainable_weight_false_by_default(self):
        """Test that trainable_weight defaults to False."""
        layer = BitLinearLRC(64, 32, rank=8)
        assert layer.trainable_weight is False
        assert not layer.weight.requires_grad

    def test_trainable_weight_true_enables_grad(self):
        """Test that trainable_weight=True enables weight gradients."""
        layer = BitLinearLRC(64, 32, rank=8, trainable_weight=True)
        assert layer.trainable_weight is True
        assert layer.weight.requires_grad

    def test_trainable_weight_gradient_flow(self):
        """Test gradients flow to weights when trainable_weight=True."""
        layer = BitLinearLRC(64, 32, rank=8, trainable_weight=True)
        x = torch.randn(2, 8, 64, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Gradients should flow to weight
        assert layer.weight.grad is not None
        # And still to LRC params
        assert layer.lrc_U.grad is not None
        assert layer.lrc_V.grad is not None

    def test_trainable_weight_false_no_weight_grad(self):
        """Test no gradients flow to weights when trainable_weight=False."""
        layer = BitLinearLRC(64, 32, rank=8, trainable_weight=False)
        x = torch.randn(2, 8, 64, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # No gradients to weight (frozen)
        assert layer.weight.grad is None
        # But LRC params still get gradients
        assert layer.lrc_U.grad is not None
        assert layer.lrc_V.grad is not None

    def test_trainable_weight_uses_ste(self):
        """Test that trainable_weight=True uses STE (on-the-fly quantization)."""
        layer = BitLinearLRC(64, 32, rank=8, trainable_weight=True)
        x = torch.randn(2, 8, 64)

        # Modify weight - should affect output (STE uses weight, not cached)
        output1 = layer(x).clone()

        with torch.no_grad():
            layer.weight.add_(0.1)

        output2 = layer(x)

        # Outputs should differ because STE uses weight directly
        assert not torch.allclose(output1, output2)

    def test_trainable_weight_false_uses_cached(self):
        """Test that trainable_weight=False uses cached weight_quantized."""
        layer = BitLinearLRC(64, 32, rank=8, trainable_weight=False)
        x = torch.randn(2, 8, 64)

        output1 = layer(x).clone()

        # Modify weight (but NOT weight_quantized)
        with torch.no_grad():
            layer.weight.add_(0.1)

        output2 = layer(x)

        # Outputs should be identical (uses cached weight_quantized)
        assert torch.allclose(output1, output2)

    def test_convert_with_trainable_weight(self):
        """Test conversion with trainable_weight=True."""
        linear = BitLinear(128, 64)
        model = nn.Sequential(linear)

        model = convert_bitlinear_to_lrc(model, trainable_weight=True)

        lrc_layer = model[0]
        assert isinstance(lrc_layer, BitLinearLRC)
        assert lrc_layer.trainable_weight is True
        assert lrc_layer.weight.requires_grad

    def test_trainable_weight_with_lrc_both_paths_work(self):
        """Test that both quantized path and LRC path contribute when trainable."""
        layer = BitLinearLRC(64, 32, rank=8, trainable_weight=True)
        x = torch.randn(2, 8, 64)

        # Set non-zero LRC
        with torch.no_grad():
            layer.lrc_U.normal_(std=0.1)
            layer.lrc_V.normal_(std=0.1)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # All trainable params should have gradients
        assert layer.weight.grad is not None
        assert layer.lrc_U.grad is not None
        assert layer.lrc_V.grad is not None
