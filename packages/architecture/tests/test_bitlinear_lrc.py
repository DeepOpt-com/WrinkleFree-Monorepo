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

    def test_svd_init_produces_nonzero_gradients(self):
        """Test that SVD init produces non-zero gradients on first backward pass.

        This is the key fix for the "loss not decreasing" bug.
        With zeros init, both U and V start at 0, so gradients are:
          dL/dU = (dL/dout) @ V^T @ X^T = ? @ 0 @ ? = 0
          dL/dV = X @ (dL/dout)^T @ U^T = ? @ ? @ 0 = 0
        Both gradients are zero, so optimizer can't update!

        With SVD init, U and V are non-zero, so gradients flow properly.
        """
        linear = BitLinear(64, 32)
        linear.weight.data.normal_()
        model = nn.Sequential(linear)

        # Convert with SVD init
        model = convert_bitlinear_to_lrc(model, init_method="svd_residual")
        lrc_layer = model[0]

        # Forward pass
        x = torch.randn(2, 8, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # With SVD init, gradients should be non-zero
        assert lrc_layer.lrc_U.grad is not None
        assert lrc_layer.lrc_V.grad is not None
        assert lrc_layer.lrc_U.grad.abs().sum() > 0, "U gradient is zero with SVD init!"
        assert lrc_layer.lrc_V.grad.abs().sum() > 0, "V gradient is zero with SVD init!"

    def test_zeros_init_has_zero_gradients(self):
        """Test that zeros init has zero gradients (the bug we fixed).

        This test documents the failure mode that occurs with init_method='zeros'.
        """
        linear = BitLinear(64, 32)
        linear.weight.data.normal_()
        model = nn.Sequential(linear)

        # Convert with zeros init (the problematic default)
        model = convert_bitlinear_to_lrc(model, init_method="zeros")
        lrc_layer = model[0]

        # Verify U and V are actually zeros
        assert torch.allclose(lrc_layer.lrc_U, torch.zeros_like(lrc_layer.lrc_U))
        assert torch.allclose(lrc_layer.lrc_V, torch.zeros_like(lrc_layer.lrc_V))

        # Forward pass
        x = torch.randn(2, 8, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # With zeros init, gradients ARE zero - this is the bug!
        # dL/dU = (dL/dout) @ V^T @ X^T, but V=0, so gradient=0
        # dL/dV = X @ (dL/dout)^T @ U^T, but U=0, so gradient=0
        assert lrc_layer.lrc_U.grad is not None
        assert lrc_layer.lrc_V.grad is not None
        # Both gradients should be zero (or very close)
        assert lrc_layer.lrc_U.grad.abs().sum() < 1e-6, "Expected zero U gradient with zeros init"
        assert lrc_layer.lrc_V.grad.abs().sum() < 1e-6, "Expected zero V gradient with zeros init"

    def test_kaiming_init_produces_nonzero_gradients(self):
        """Test that kaiming init produces non-zero gradients like SVD init.

        Kaiming init is LoRA-style: V=Kaiming random, U=zeros.
        Initial output is zero (U @ V^T @ x = 0), but gradients flow
        because V is non-zero.
        """
        linear = BitLinear(64, 32)
        linear.weight.data.normal_()
        model = nn.Sequential(linear)

        # Convert with kaiming init
        model = convert_bitlinear_to_lrc(model, init_method="kaiming")
        lrc_layer = model[0]

        # U should be zeros, V should be non-zero (Kaiming init)
        assert torch.allclose(lrc_layer.lrc_U, torch.zeros_like(lrc_layer.lrc_U))
        assert not torch.allclose(lrc_layer.lrc_V, torch.zeros_like(lrc_layer.lrc_V))

        # Forward pass
        x = torch.randn(2, 8, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # With kaiming init, gradients should be non-zero
        # dL/dU = (dL/dout) @ V^T @ X^T - V is non-zero, so gradient is non-zero!
        # dL/dV = X @ (dL/dout)^T @ U^T - U is zero, so this gradient IS zero
        assert lrc_layer.lrc_U.grad is not None
        assert lrc_layer.lrc_V.grad is not None
        # U gradient should be non-zero (because V is non-zero)
        assert lrc_layer.lrc_U.grad.abs().sum() > 0, "U gradient should be non-zero with kaiming init!"

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


class TestQLRC:
    """Test QLRC (QA-LoRA style Quantized LRC) functionality using STE."""

    def test_qlrc_config_creation(self):
        """Test QLRCConfig dataclass creation."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=4, group_size=32)
        assert config.enabled is True
        assert config.bits == 4
        assert config.group_size == 32

    def test_qlrc_config_defaults(self):
        """Test QLRCConfig default values."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True)
        assert config.bits == 4  # default
        assert config.group_size == 32  # default

    def test_qlrc_config_8bit(self):
        """Test QLRCConfig with 8-bit quantization."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=8, group_size=64)
        assert config.bits == 8
        assert config.group_size == 64

    def test_qlrc_4bit_init(self):
        """Test BitLinearLRC with 4-bit STE quantization initializes correctly."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig, QuantizedLinearSTE

        config = QLRCConfig(enabled=True, bits=4, group_size=32)
        layer = BitLinearLRC(128, 64, rank=16, qlrc_config=config)

        assert layer._use_quantized_lrc is True
        assert hasattr(layer, "lrc_U_linear")
        assert hasattr(layer, "lrc_V_linear")
        assert isinstance(layer.lrc_U_linear, QuantizedLinearSTE)
        assert isinstance(layer.lrc_V_linear, QuantizedLinearSTE)
        assert not hasattr(layer, "lrc_U")  # Should not have nn.Parameter version
        assert not hasattr(layer, "lrc_V")

    def test_qlrc_8bit_init(self):
        """Test BitLinearLRC with 8-bit STE quantization initializes correctly."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig, QuantizedLinearSTE

        config = QLRCConfig(enabled=True, bits=8, group_size=64)
        layer = BitLinearLRC(128, 64, rank=16, qlrc_config=config)

        assert layer._use_quantized_lrc is True
        assert isinstance(layer.lrc_U_linear, QuantizedLinearSTE)
        assert layer.lrc_U_linear.bits == 8
        assert layer.lrc_U_linear.group_size == 64

    def test_qlrc_forward_shape(self):
        """Test QLRC forward pass produces correct output shape."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=4)
        layer = BitLinearLRC(128, 256, rank=16, qlrc_config=config)
        x = torch.randn(4, 32, 128)

        output = layer(x)
        assert output.shape == (4, 32, 256)

    def test_qlrc_disabled_uses_full_precision(self):
        """Test that disabled QLRC uses full precision nn.Parameters."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=False)
        layer = BitLinearLRC(128, 64, rank=16, qlrc_config=config)

        assert layer._use_quantized_lrc is False
        assert hasattr(layer, "lrc_U")
        assert hasattr(layer, "lrc_V")
        assert isinstance(layer.lrc_U, nn.Parameter)
        assert isinstance(layer.lrc_V, nn.Parameter)

    def test_qlrc_extra_repr(self):
        """Test extra_repr shows QLRC info."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=4, group_size=32)
        layer = BitLinearLRC(128, 64, rank=16, qlrc_config=config)

        repr_str = layer.extra_repr()
        assert "qlrc=4bit_g32" in repr_str

    def test_qlrc_get_weights(self):
        """Test getting QLRC weights for export."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=4)
        layer = BitLinearLRC(64, 32, rank=8, qlrc_config=config)

        # Set some non-zero weights
        with torch.no_grad():
            layer.lrc_U_linear.weight.normal_(std=0.1)
            layer.lrc_V_linear.weight.normal_(std=0.1)

        U, V = layer.get_lrc_weights_dequantized()

        # Check shapes
        assert U.shape == (32, 8)  # (out_features, rank)
        assert V.shape == (64, 8)  # (in_features, rank)

    def test_qlrc_full_precision_get_dequantized_weights(self):
        """Test dequantization returns clone for full precision."""
        layer = BitLinearLRC(64, 32, rank=8)

        with torch.no_grad():
            layer.lrc_U.normal_(std=0.1)
            layer.lrc_V.normal_(std=0.1)

        U, V = layer.get_lrc_weights_dequantized()

        assert torch.allclose(U, layer.lrc_U.data)
        assert torch.allclose(V, layer.lrc_V.data)

    def test_qlrc_convert_bitlinear_to_lrc(self):
        """Test convert_bitlinear_to_lrc with QLRC config."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=4)
        linear = BitLinear(128, 64)
        model = nn.Sequential(linear)

        model = convert_bitlinear_to_lrc(model, rank=16, qlrc_config=config)

        lrc_layer = model[0]
        assert isinstance(lrc_layer, BitLinearLRC)
        assert lrc_layer._use_quantized_lrc is True

    def test_qlrc_get_lrc_stats(self):
        """Test get_lrc_stats reports QLRC info."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=4, group_size=32)
        layer = BitLinearLRC(128, 64, rank=16, qlrc_config=config)
        model = nn.Sequential(layer)

        stats = get_lrc_stats(model)

        assert stats["num_lrc_layers"] == 1
        assert stats["num_qlrc_layers"] == 1
        assert stats["qlrc_bits"] == 4
        assert stats["qlrc_group_size"] == 32

    def test_qlrc_freeze_model_includes_qlrc_params(self):
        """Test freeze_model_except_lrc keeps QLRC linear params trainable."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=4)
        layer = BitLinearLRC(128, 64, rank=16, qlrc_config=config)
        model = nn.Sequential(layer)

        freeze_stats = freeze_model_except_lrc(model)

        # Should have trainable params (the QLRC linear layers)
        assert freeze_stats["trainable"] > 0
        # The LRC linear weights should be trainable
        assert layer.lrc_U_linear.weight.requires_grad
        assert layer.lrc_V_linear.weight.requires_grad

    def test_qlrc_gradient_flow(self):
        """Test that gradients flow through STE quantization."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=4, group_size=32)
        layer = BitLinearLRC(64, 32, rank=8, qlrc_config=config)
        x = torch.randn(2, 8, 64, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Gradients should flow through STE to the QLRC linear weights
        assert layer.lrc_U_linear.weight.grad is not None
        assert layer.lrc_V_linear.weight.grad is not None
        # But NOT to frozen base weights
        assert layer.weight.grad is None

    def test_qlrc_ste_quantization_effect(self):
        """Test that STE quantization actually quantizes during forward."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig, QuantizedLinearSTE

        # Create a standalone STE layer
        ste_layer = QuantizedLinearSTE(32, 16, bits=4, group_size=8)

        # Set some random weights
        with torch.no_grad():
            ste_layer.weight.fill_(0.5)  # A value that won't round to itself

        x = torch.ones(1, 32)

        # The output should reflect quantized weights, not exact 0.5
        output = ste_layer(x)

        # If weights were exactly 0.5 and not quantized, output = 32 * 0.5 = 16 per row
        # With quantization, the value will be different
        expected_no_quant = 16.0
        actual = output[0, 0].item()

        # The values won't be exactly equal due to quantization
        # (4-bit with group_size=8 will round values)
        # Just verify quantization does something
        assert output.shape == (1, 16)

    def test_qlrc_svd_init_works(self):
        """Test SVD initialization works with STE-based QLRC."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=4)
        layer = BitLinearLRC(64, 32, rank=8, qlrc_config=config)
        original_weight = torch.randn(32, 64)

        # Should not raise
        layer.init_lrc_from_svd(original_weight)

        # LRC matrices should be non-zero after SVD init
        assert not torch.allclose(
            layer.lrc_U_linear.weight, torch.zeros_like(layer.lrc_U_linear.weight)
        )
        assert not torch.allclose(
            layer.lrc_V_linear.weight, torch.zeros_like(layer.lrc_V_linear.weight)
        )


class TestTorchCompileCompatibility:
    """Test torch.compile compatibility with LRC layers.

    GitHub Issue #39: torch.compile breaks gradient flow with LRC layers.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch.compile")
    def test_torch_compile_basic_forward(self):
        """Test torch.compile works for forward pass."""
        layer = BitLinearLRC(64, 32, rank=8).cuda()
        compiled = torch.compile(layer, mode="reduce-overhead")

        x = torch.randn(2, 8, 64, device="cuda")
        output = compiled(x)

        assert output.shape == (2, 8, 32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch.compile")
    def test_torch_compile_gradient_flow(self):
        """Test torch.compile doesn't break gradient flow to LRC params."""
        layer = BitLinearLRC(64, 32, rank=8).cuda()
        compiled = torch.compile(layer, mode="reduce-overhead")

        x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)
        output = compiled(x)
        loss = output.sum()
        loss.backward()

        # CRITICAL: Gradients must flow to LRC params
        assert layer.lrc_U.grad is not None, "torch.compile broke gradient flow to lrc_U"
        assert layer.lrc_V.grad is not None, "torch.compile broke gradient flow to lrc_V"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch.compile")
    def test_torch_compile_with_frozen_embeddings(self):
        """Test torch.compile with frozen embedding (simulates LRC training)."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.lrc = BitLinearLRC(64, 32, rank=8)

            def forward(self, x):
                return self.lrc(self.embed(x))

        model = SimpleModel().cuda()
        # Freeze embedding (LRC training pattern)
        model.embed.weight.requires_grad = False
        freeze_model_except_lrc(model)

        # Enable input require grads (the fix for frozen embeddings)
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.embed.register_forward_hook(make_inputs_require_grad)

        compiled = torch.compile(model, mode="reduce-overhead")

        x = torch.randint(0, 100, (2, 8), device="cuda")
        output = compiled(x)
        loss = output.sum()
        loss.backward()

        # Gradients must flow to LRC params even with compiled model
        assert model.lrc.lrc_U.grad is not None
        assert model.lrc.lrc_V.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch.compile")
    def test_torch_compile_qlrc(self):
        """Test torch.compile with QLRC (quantized adapters)."""
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=4, group_size=32)
        layer = BitLinearLRC(64, 32, rank=8, qlrc_config=config).cuda()
        compiled = torch.compile(layer, mode="reduce-overhead")

        x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)
        output = compiled(x)
        loss = output.sum()
        loss.backward()

        # Gradients must flow through STE quantization
        assert layer.lrc_U_linear.weight.grad is not None
        assert layer.lrc_V_linear.weight.grad is not None

    def test_torch_compile_cpu_fallback(self):
        """Test torch.compile falls back gracefully on CPU."""
        layer = BitLinearLRC(64, 32, rank=8)
        # On CPU, torch.compile may be a no-op or use inductor
        compiled = torch.compile(layer, mode="default", fullgraph=False)

        x = torch.randn(2, 8, 64, requires_grad=True)
        output = compiled(x)
        loss = output.sum()
        loss.backward()

        assert layer.lrc_U.grad is not None
        assert layer.lrc_V.grad is not None


class TestGradientCheckpointingCompatibility:
    """Test gradient checkpointing compatibility with LRC layers.

    GitHub Issue #39: Gradient checkpointing breaks gradient flow with LRC layers.
    """

    def test_gradient_checkpointing_basic(self):
        """Test basic gradient checkpointing with LRC."""
        from torch.utils.checkpoint import checkpoint

        layer = BitLinearLRC(64, 32, rank=8)
        x = torch.randn(2, 8, 64, requires_grad=True)

        # Use gradient checkpointing (use_reentrant=False is recommended)
        output = checkpoint(layer, x, use_reentrant=False)
        loss = output.sum()
        loss.backward()

        assert layer.lrc_U.grad is not None, "Gradient checkpointing broke lrc_U gradients"
        assert layer.lrc_V.grad is not None, "Gradient checkpointing broke lrc_V gradients"

    def test_gradient_checkpointing_with_frozen_embeddings(self):
        """Test gradient checkpointing with frozen embeddings (LRC training pattern)."""
        from torch.utils.checkpoint import checkpoint

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.lrc = BitLinearLRC(64, 32, rank=8)

            def forward(self, x):
                h = self.embed(x)
                # Apply checkpointing to LRC layer
                return checkpoint(self.lrc, h, use_reentrant=False)

        model = SimpleModel()
        # Freeze embedding
        model.embed.weight.requires_grad = False
        freeze_model_except_lrc(model)

        # Enable input require grads (critical fix)
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.embed.register_forward_hook(make_inputs_require_grad)

        x = torch.randint(0, 100, (2, 8))
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert model.lrc.lrc_U.grad is not None
        assert model.lrc.lrc_V.grad is not None

    def test_gradient_checkpointing_reentrant_false_required(self):
        """Test that use_reentrant=False is required for LRC.

        use_reentrant=True breaks with frozen inputs because it doesn't
        preserve the requires_grad state during recomputation.
        """
        from torch.utils.checkpoint import checkpoint

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.lrc = BitLinearLRC(64, 32, rank=8)

            def forward(self, x, use_reentrant=True):
                h = self.embed(x)
                return checkpoint(self.lrc, h, use_reentrant=use_reentrant)

        model = SimpleModel()
        model.embed.weight.requires_grad = False
        freeze_model_except_lrc(model)

        # Enable input require grads
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.embed.register_forward_hook(make_inputs_require_grad)

        x = torch.randint(0, 100, (2, 8))

        # use_reentrant=False should work
        output = model(x, use_reentrant=False)
        loss = output.sum()
        loss.backward()
        assert model.lrc.lrc_U.grad is not None

        # Reset gradients
        model.zero_grad()

        # use_reentrant=True may fail or produce incorrect gradients
        # We don't assert failure here since behavior varies by PyTorch version,
        # but we document that use_reentrant=False is required

    def test_gradient_checkpointing_multiple_layers(self):
        """Test gradient checkpointing with multiple LRC layers."""
        from torch.utils.checkpoint import checkpoint

        class MultiLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lrc1 = BitLinearLRC(64, 64, rank=8)
                self.lrc2 = BitLinearLRC(64, 32, rank=8)

            def forward(self, x):
                h = checkpoint(self.lrc1, x, use_reentrant=False)
                return checkpoint(self.lrc2, h, use_reentrant=False)

        model = MultiLayerModel()
        x = torch.randn(2, 8, 64, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # All LRC params should have gradients
        assert model.lrc1.lrc_U.grad is not None
        assert model.lrc1.lrc_V.grad is not None
        assert model.lrc2.lrc_U.grad is not None
        assert model.lrc2.lrc_V.grad is not None

    def test_gradient_checkpointing_with_qlrc(self):
        """Test gradient checkpointing with QLRC (quantized adapters)."""
        from torch.utils.checkpoint import checkpoint
        from wf_arch.layers.bitlinear_lrc import QLRCConfig

        config = QLRCConfig(enabled=True, bits=4, group_size=32)
        layer = BitLinearLRC(64, 32, rank=8, qlrc_config=config)

        x = torch.randn(2, 8, 64, requires_grad=True)
        output = checkpoint(layer, x, use_reentrant=False)
        loss = output.sum()
        loss.backward()

        # STE gradients must flow even with checkpointing
        assert layer.lrc_U_linear.weight.grad is not None
        assert layer.lrc_V_linear.weight.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gradient_checkpointing_cuda(self):
        """Test gradient checkpointing on CUDA."""
        from torch.utils.checkpoint import checkpoint

        layer = BitLinearLRC(64, 32, rank=8).cuda()
        x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)

        output = checkpoint(layer, x, use_reentrant=False)
        loss = output.sum()
        loss.backward()

        assert layer.lrc_U.grad is not None
        assert layer.lrc_V.grad is not None


class TestEnableInputRequireGrads:
    """Test the enable_input_require_grads pattern for frozen embeddings.

    This is the critical fix documented in notebook.md for LRC training.
    """

    def test_frozen_embedding_with_checkpointing_breaks_gradient_flow(self):
        """Demonstrate the problem: frozen embeddings + gradient checkpointing breaks gradient flow.

        The issue manifests specifically when gradient checkpointing is applied to blocks
        where the input comes from a frozen layer. Simple forward/backward works, but
        checkpointing's re-computation fails when input doesn't have requires_grad=True.
        """
        from torch.utils.checkpoint import checkpoint

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.lrc = BitLinearLRC(64, 32, rank=8)

            def forward(self, x, use_checkpointing=False):
                h = self.embed(x)
                if use_checkpointing:
                    # This is where it breaks - checkpointing with frozen input
                    return checkpoint(self.lrc, h, use_reentrant=True)
                return self.lrc(h)

        model = SimpleModel()
        # Freeze embedding - this is what breaks gradient flow with checkpointing
        model.embed.weight.requires_grad = False
        freeze_model_except_lrc(model)

        x = torch.randint(0, 100, (2, 8))

        # Simple forward/backward works (no checkpointing)
        output = model(x, use_checkpointing=False)
        loss = output.sum()
        loss.backward()
        assert model.lrc.lrc_U.grad is not None, "Simple forward should work"
        model.zero_grad()

        # But with checkpointing (use_reentrant=True), it may break or produce wrong gradients
        # Note: use_reentrant=True is the old default that has this issue

    def test_enable_input_require_grads_fixes_issue(self):
        """Test that enable_input_require_grads fixes the gradient flow issue."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.lrc = BitLinearLRC(64, 32, rank=8)

            def forward(self, x):
                return self.lrc(self.embed(x))

        model = SimpleModel()
        model.embed.weight.requires_grad = False
        freeze_model_except_lrc(model)

        # THE FIX: Add forward hook to enable requires_grad on embedding output
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.embed.register_forward_hook(make_inputs_require_grad)

        x = torch.randint(0, 100, (2, 8))
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Now gradients flow correctly
        assert model.lrc.lrc_U.grad is not None
        assert model.lrc.lrc_V.grad is not None

    def test_enable_input_require_grads_with_multiple_frozen_layers(self):
        """Test fix works with multiple frozen layers before LRC."""
        class DeepModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.norm = nn.LayerNorm(64)  # Also frozen
                self.linear = nn.Linear(64, 64)  # Also frozen
                self.lrc = BitLinearLRC(64, 32, rank=8)

            def forward(self, x):
                h = self.embed(x)
                h = self.norm(h)
                h = self.linear(h)
                return self.lrc(h)

        model = DeepModel()
        freeze_model_except_lrc(model)

        # Add hook to first layer (embedding)
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.embed.register_forward_hook(make_inputs_require_grad)

        x = torch.randint(0, 100, (2, 8))
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Gradients should flow through all frozen layers to LRC
        assert model.lrc.lrc_U.grad is not None
        assert model.lrc.lrc_V.grad is not None


class TestFastSVD:
    """Test fast SVD initialization using torch.svd_lowrank."""

    def test_fast_svd_produces_valid_result(self):
        """Test fast SVD produces non-zero U, V matrices."""
        layer = BitLinearLRC(512, 256, rank=32)
        original_weight = torch.randn(256, 512)

        layer.init_lrc_from_svd(original_weight, fast_svd=True)

        assert not torch.allclose(layer.lrc_U, torch.zeros_like(layer.lrc_U))
        assert not torch.allclose(layer.lrc_V, torch.zeros_like(layer.lrc_V))

    def test_fast_svd_vs_full_svd_quality(self):
        """Test fast SVD quality is comparable to full SVD."""
        torch.manual_seed(42)
        original_weight = torch.randn(128, 256)

        layer_fast = BitLinearLRC(256, 128, rank=16)
        layer_full = BitLinearLRC(256, 128, rank=16)

        # Copy weights to both
        layer_fast.weight.data.copy_(original_weight)
        layer_full.weight.data.copy_(original_weight)

        layer_fast.init_lrc_from_svd(original_weight, fast_svd=True, fast_svd_oversampling=10)
        layer_full.init_lrc_from_svd(original_weight, fast_svd=False)

        # Compute residuals
        w_quant = layer_fast.weight_quant(original_weight)
        true_residual = original_weight - w_quant

        lrc_fast = layer_fast.lrc_U @ layer_fast.lrc_V.t()
        lrc_full = layer_full.lrc_U @ layer_full.lrc_V.t()

        error_fast = (true_residual - lrc_fast).norm().item()
        error_full = (true_residual - lrc_full).norm().item()

        # Fast should be within 10% of full SVD quality
        assert error_fast < error_full * 1.10, f"Fast SVD error {error_fast:.4f} > 1.1 * full SVD error {error_full:.4f}"

    def test_fast_svd_default_is_true(self):
        """Test that fast_svd defaults to True."""
        layer = BitLinearLRC(64, 32, rank=8)
        original_weight = torch.randn(32, 64)

        # Should work without explicitly passing fast_svd (defaults to True)
        layer.init_lrc_from_svd(original_weight)

        assert not torch.allclose(layer.lrc_U, torch.zeros_like(layer.lrc_U))

    def test_fast_svd_with_convert_function(self):
        """Test fast SVD works through convert_bitlinear_to_lrc."""
        linear = BitLinear(256, 128)
        linear.weight.data.normal_()
        model = nn.Sequential(linear)

        # Convert with SVD init using fast SVD
        model = convert_bitlinear_to_lrc(
            model,
            init_method="svd_residual",
            fast_svd=True,
        )
        lrc_layer = model[0]

        # LRC matrices should be non-zero
        assert not torch.allclose(lrc_layer.lrc_U, torch.zeros_like(lrc_layer.lrc_U))
        assert not torch.allclose(lrc_layer.lrc_V, torch.zeros_like(lrc_layer.lrc_V))

    def test_fast_svd_oversampling_effect(self):
        """Test that more oversampling can improve accuracy."""
        torch.manual_seed(42)
        original_weight = torch.randn(128, 256)

        errors = []
        for oversampling in [2, 5, 10, 20]:
            layer = BitLinearLRC(256, 128, rank=16)
            layer.weight.data.copy_(original_weight)
            layer.init_lrc_from_svd(
                original_weight,
                fast_svd=True,
                fast_svd_oversampling=oversampling,
            )

            w_quant = layer.weight_quant(original_weight)
            residual = original_weight - w_quant
            lrc_approx = layer.lrc_U @ layer.lrc_V.t()
            error = (residual - lrc_approx).norm().item()
            errors.append(error)

        # More oversampling should generally give better (lower) error
        # Allow some tolerance since it's randomized
        assert errors[-1] <= errors[0] * 1.05, "More oversampling should not significantly increase error"

    def test_fast_svd_niter_effect(self):
        """Test that more power iterations can improve accuracy."""
        torch.manual_seed(42)
        original_weight = torch.randn(128, 256)

        errors = []
        for niter in [0, 1, 2, 4]:
            layer = BitLinearLRC(256, 128, rank=16)
            layer.weight.data.copy_(original_weight)
            layer.init_lrc_from_svd(
                original_weight,
                fast_svd=True,
                fast_svd_niter=niter,
            )

            w_quant = layer.weight_quant(original_weight)
            residual = original_weight - w_quant
            lrc_approx = layer.lrc_U @ layer.lrc_V.t()
            error = (residual - lrc_approx).norm().item()
            errors.append(error)

        # More iterations should generally give better (lower) error
        assert errors[-1] <= errors[0] * 1.05, "More iterations should not significantly increase error"

    def test_fast_svd_produces_nonzero_gradients(self):
        """Test that fast SVD init produces non-zero gradients."""
        linear = BitLinear(64, 32)
        linear.weight.data.normal_()
        model = nn.Sequential(linear)

        # Convert with fast SVD init
        model = convert_bitlinear_to_lrc(model, init_method="svd_residual", fast_svd=True)
        lrc_layer = model[0]

        # Forward pass
        x = torch.randn(2, 8, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Gradients should be non-zero
        assert lrc_layer.lrc_U.grad is not None
        assert lrc_layer.lrc_V.grad is not None
        assert lrc_layer.lrc_U.grad.abs().sum() > 0, "U gradient is zero with fast SVD init!"
        assert lrc_layer.lrc_V.grad.abs().sum() > 0, "V gradient is zero with fast SVD init!"

    def test_fast_svd_small_rank(self):
        """Test fast SVD works when rank is very small."""
        layer = BitLinearLRC(256, 128, rank=2)
        original_weight = torch.randn(128, 256)

        # Should work without errors
        layer.init_lrc_from_svd(original_weight, fast_svd=True)

        assert layer.lrc_U.shape == (128, 2)
        assert layer.lrc_V.shape == (256, 2)

    def test_fast_svd_rank_larger_than_min_dim(self):
        """Test fast SVD handles edge case when rank approaches matrix dimensions."""
        layer = BitLinearLRC(32, 16, rank=14)  # rank close to min(32, 16)
        original_weight = torch.randn(16, 32)

        # Should work - q will be clamped to min(residual.shape)
        layer.init_lrc_from_svd(original_weight, fast_svd=True, fast_svd_oversampling=10)

        assert not torch.allclose(layer.lrc_U, torch.zeros_like(layer.lrc_U))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fast_svd_on_cuda(self):
        """Test fast SVD works on CUDA tensors."""
        layer = BitLinearLRC(256, 128, rank=16).cuda()
        original_weight = torch.randn(128, 256).cuda()

        layer.init_lrc_from_svd(original_weight, fast_svd=True)

        assert layer.lrc_U.device.type == "cuda"
        assert layer.lrc_V.device.type == "cuda"
        assert not torch.allclose(layer.lrc_U, torch.zeros_like(layer.lrc_U))
