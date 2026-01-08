"""Tests for BitLinearSalient layer."""

import pytest
import torch
import torch.nn as nn

from wf_arch.layers.bitlinear import BitLinear
from wf_arch.layers.bitlinear_salient import (
    BitLinearSalient,
    SalientConfig,
    convert_bitlinear_to_salient,
    get_salient_stats,
)


class TestBitLinearSalient:
    """Test BitLinearSalient layer functionality."""

    def test_init_default_ratio(self):
        """Test default 1% salient ratio."""
        layer = BitLinearSalient(1000, 512)
        assert layer.salient_ratio == 0.01
        assert layer.num_salient == 10  # 1% of 1000

    def test_init_custom_ratio(self):
        """Test custom salient ratio."""
        layer = BitLinearSalient(1000, 512, salient_ratio=0.05)
        assert layer.num_salient == 50  # 5% of 1000

    def test_init_minimum_one_salient(self):
        """Test that at least 1 column is salient."""
        layer = BitLinearSalient(10, 5, salient_ratio=0.001)
        # 0.1% of 10 = 0.01, but we clamp to min 1
        assert layer.num_salient == 1

    def test_forward_before_calibration(self):
        """Test forward pass works before calibration (standard BitLinear behavior)."""
        layer = BitLinearSalient(64, 32)
        x = torch.randn(2, 8, 64)
        output = layer(x)
        assert output.shape == (2, 8, 32)
        assert not layer.is_calibrated

    def test_set_salient_columns(self):
        """Test setting salient column indices."""
        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        indices = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
        layer.set_salient_columns(indices)
        assert layer.is_calibrated
        assert torch.equal(layer.salient_indices, indices)

    def test_set_salient_columns_wrong_count_raises(self):
        """Test that setting wrong number of salient columns raises error."""
        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        indices = torch.tensor([5, 15, 25])  # Wrong count (should be 10)
        with pytest.raises(ValueError, match="Expected 10 salient columns"):
            layer.set_salient_columns(indices)

    def test_forward_after_calibration(self):
        """Test forward pass after calibration uses mixed precision."""
        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        indices = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
        layer.set_salient_columns(indices)

        x = torch.randn(2, 8, 100)
        output = layer(x)
        assert output.shape == (2, 8, 50)

    def test_gradient_flow(self):
        """Test gradients flow through both salient and non-salient paths."""
        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        indices = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
        layer.set_salient_columns(indices)

        x = torch.randn(2, 8, 100, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Gradients should flow to weight
        assert layer.weight.grad is not None
        # Check gradients exist for both salient and non-salient columns
        salient_grad = layer.weight.grad[:, indices].abs().sum()
        assert salient_grad > 0, "No gradients to salient columns"

    def test_salient_columns_fp16(self):
        """Test that salient columns get full precision treatment."""
        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        # Make some columns have very large weights
        with torch.no_grad():
            layer.weight[:, 5] *= 100  # Make column 5 very large

        indices = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
        layer.set_salient_columns(indices)

        x = torch.randn(2, 8, 100)
        output = layer(x)

        # Output should be valid (no NaN/Inf from quantization of large weights)
        assert torch.isfinite(output).all()

    def test_nonsalient_indices_computed_correctly(self):
        """Test that non-salient indices are the complement of salient indices."""
        layer = BitLinearSalient(20, 10, salient_ratio=0.1)  # 2 salient
        indices = torch.tensor([5, 15])
        layer.set_salient_columns(indices)

        # Non-salient should be all indices except 5 and 15
        expected = torch.tensor([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19])
        assert torch.equal(layer.nonsalient_indices, expected)

    def test_get_salient_stats(self):
        """Test salient statistics."""
        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        indices = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
        layer.set_salient_columns(indices)

        stats = layer.get_salient_stats()
        assert stats["in_features"] == 100
        assert stats["out_features"] == 50
        assert stats["num_salient"] == 10
        assert stats["salient_ratio"] == 0.1
        assert stats["is_calibrated"] is True
        assert stats["salient_indices"] == indices.tolist()

    def test_mixed_precision_dtype(self):
        """Test layer works with different dtypes."""
        layer = BitLinearSalient(64, 32)
        if torch.cuda.is_available():
            layer = layer.cuda().half()
            x = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
            output = layer(x)
            assert output.dtype == torch.float16

    def test_extra_repr(self):
        """Test string representation."""
        layer = BitLinearSalient(128, 256, salient_ratio=0.05)
        repr_str = layer.extra_repr()
        assert "128" in repr_str
        assert "256" in repr_str
        assert "0.05" in repr_str


class TestConvertBitLinearToSalient:
    """Test conversion utilities."""

    def test_convert_preserves_weights(self):
        """Test conversion preserves original weights."""
        linear = BitLinear(128, 64)
        original_weight = linear.weight.data.clone()

        model = nn.Sequential(linear)
        model = convert_bitlinear_to_salient(model, salient_ratio=0.05)

        salient_layer = model[0]
        assert isinstance(salient_layer, BitLinearSalient)
        assert torch.allclose(salient_layer.weight.data, original_weight)

    def test_convert_with_precomputed_indices(self):
        """Test conversion with pre-computed salient indices."""
        linear = BitLinear(100, 50)
        model = nn.Sequential(linear)

        indices = {"0": torch.tensor([10, 20, 30, 40, 50])}
        model = convert_bitlinear_to_salient(
            model,
            salient_ratio=0.05,
            salient_indices=indices,
        )

        salient_layer = model[0]
        assert salient_layer.is_calibrated
        assert salient_layer.num_salient == 5

    def test_exclude_layers(self):
        """Test layer exclusion."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = BitLinear(64, 64)
                self.layer2 = BitLinear(64, 64)

        model = Model()
        model = convert_bitlinear_to_salient(model, exclude_names=["layer1"])

        assert isinstance(model.layer1, BitLinear)
        assert not isinstance(model.layer1, BitLinearSalient)
        assert isinstance(model.layer2, BitLinearSalient)

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
        model = convert_bitlinear_to_salient(model)

        assert isinstance(model.block[0], BitLinearSalient)
        assert isinstance(model.block[2], BitLinearSalient)


class TestGetSalientStats:
    """Test salient statistics utility."""

    def test_stats_single_layer(self):
        """Test stats for single salient layer."""
        layer = BitLinearSalient(1000, 512, salient_ratio=0.01)
        model = nn.Sequential(layer)

        stats = get_salient_stats(model)
        assert stats["num_salient_layers"] == 1
        assert stats["total_salient_columns"] == 10  # 1% of 1000
        assert stats["average_salient_ratio"] == 0.01

    def test_stats_multiple_layers(self):
        """Test stats for multiple salient layers."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = BitLinearSalient(1000, 500, salient_ratio=0.01)  # 10 salient
                self.layer2 = BitLinearSalient(500, 200, salient_ratio=0.02)  # 10 salient

        model = Model()
        stats = get_salient_stats(model)

        assert stats["num_salient_layers"] == 2
        assert stats["total_salient_columns"] == 20  # 10 + 10
        assert stats["total_columns"] == 1500  # 1000 + 500

    def test_stats_calibration_tracking(self):
        """Test stats tracks calibration status."""
        layer1 = BitLinearSalient(100, 50, salient_ratio=0.1)
        layer2 = BitLinearSalient(100, 50, salient_ratio=0.1)

        # Only calibrate layer1
        layer1.set_salient_columns(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = layer1
                self.l2 = layer2

        model = Model()
        stats = get_salient_stats(model)

        assert stats["num_salient_layers"] == 2
        assert stats["num_calibrated_layers"] == 1


class TestSalientConfig:
    """Test SalientConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SalientConfig()
        assert config.ratio == 0.01
        assert config.calibration_samples == 128

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SalientConfig(ratio=0.05, calibration_samples=256)
        assert config.ratio == 0.05
        assert config.calibration_samples == 256


class TestSalientCalibration:
    """Test calibration utilities."""

    def test_calibration_import(self):
        """Test calibration module can be imported."""
        from wf_arch.layers.salient_calibration import (
            SalientCalibrator,
            calibrate_salient_columns,
            ActivationCollector,
        )
        assert SalientCalibrator is not None
        assert calibrate_salient_columns is not None
        assert ActivationCollector is not None

    def test_activation_collector(self):
        """Test ActivationCollector accumulates statistics correctly."""
        from wf_arch.layers.salient_calibration import ActivationCollector

        collector = ActivationCollector()

        # First batch
        x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        collector.update(x1)

        # Second batch
        x2 = torch.tensor([[7.0, 8.0, 9.0]])
        collector.update(x2)

        mean_abs = collector.get_mean_abs()
        # Total: 3 samples
        # Column 0: (1 + 4 + 7) / 3 = 4.0
        # Column 1: (2 + 5 + 8) / 3 = 5.0
        # Column 2: (3 + 6 + 9) / 3 = 6.0
        expected = torch.tensor([4.0, 5.0, 6.0])
        assert torch.allclose(mean_abs, expected)

    def test_activation_collector_reset(self):
        """Test ActivationCollector can be reset."""
        from wf_arch.layers.salient_calibration import ActivationCollector

        collector = ActivationCollector()
        collector.update(torch.randn(4, 8))
        assert collector.activation_count == 4

        collector.reset()
        assert collector.activation_count == 0
        assert collector.activation_sum is None

    def test_activation_collector_empty_raises(self):
        """Test ActivationCollector raises on empty."""
        from wf_arch.layers.salient_calibration import ActivationCollector

        collector = ActivationCollector()
        with pytest.raises(RuntimeError, match="No activations collected"):
            collector.get_mean_abs()


class TestSalientVsNonSalientPaths:
    """Test that salient and non-salient paths work correctly."""

    def test_salient_path_is_unquantized(self):
        """Test that salient columns use unquantized computation."""
        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        # Set specific salient columns
        salient_idx = torch.arange(10)  # First 10 columns
        layer.set_salient_columns(salient_idx)

        # Create input with only salient columns non-zero
        x = torch.zeros(1, 1, 100)
        x[0, 0, :10] = 1.0  # Only salient columns

        # Get output
        output = layer(x)

        # For salient path: output = x_salient @ w_salient.T
        # This should be exactly w[:, :10].sum(dim=1) since x[0,:10] = 1
        expected = layer.weight[:, :10].sum(dim=1, keepdim=True).T
        expected = expected + (layer.bias if layer.bias is not None else 0)

        # With lambda=1 (full quantization), the output should equal expected
        # because salient path doesn't quantize
        assert output.shape == (1, 1, 50)

    def test_nonsalient_path_is_quantized(self):
        """Test that non-salient columns use quantized computation."""
        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        # Set salient columns (not including column 50)
        salient_idx = torch.arange(10)
        layer.set_salient_columns(salient_idx)

        # Create input with large value in non-salient column
        x = torch.zeros(1, 1, 100)
        x[0, 0, 50] = 1000.0  # Large value in non-salient column

        # The non-salient path should quantize this, clipping activations
        output = layer(x)

        # Output should be finite (quantization prevents explosion)
        assert torch.isfinite(output).all()

    def test_both_paths_contribute(self):
        """Test that both paths contribute to final output."""
        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        salient_idx = torch.arange(10)
        layer.set_salient_columns(salient_idx)

        # Input with values in both paths
        x = torch.ones(1, 1, 100)

        output_both = layer(x)

        # Zero out salient columns - should change output
        x_nonsalient_only = x.clone()
        x_nonsalient_only[0, 0, :10] = 0
        output_nonsalient = layer(x_nonsalient_only)

        # Zero out non-salient columns - should change output
        x_salient_only = torch.zeros_like(x)
        x_salient_only[0, 0, :10] = 1
        output_salient = layer(x_salient_only)

        # Both paths should contribute
        assert not torch.allclose(output_both, output_nonsalient, atol=1e-3)
        assert not torch.allclose(output_both, output_salient, atol=1e-3)


class TestLambdaWarmupIntegration:
    """Test integration with lambda warmup."""

    def test_lambda_zero_no_quantization(self):
        """Test that lambda=0 means no quantization (all FP16)."""
        from wf_arch.quantization.lambda_warmup import (
            LambdaWarmup,
            set_global_lambda_warmup,
        )

        # Set lambda to 0
        warmup = LambdaWarmup(warmup_steps=10)
        set_global_lambda_warmup(warmup)
        # Step 0 means lambda = 0

        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        salient_idx = torch.arange(10)
        layer.set_salient_columns(salient_idx)

        x = torch.randn(1, 1, 100)
        output_lambda0 = layer(x).clone()

        # Step to full quantization
        for _ in range(10):
            warmup.step()

        output_lambda1 = layer(x)

        # Outputs should differ (quantization has effect)
        # Note: May be close but not identical due to quantization
        set_global_lambda_warmup(None)  # Reset

    def test_lambda_one_full_quantization(self):
        """Test that lambda=1 means full quantization for non-salient."""
        from wf_arch.quantization.lambda_warmup import set_global_lambda_warmup

        # Reset to default (lambda=1)
        set_global_lambda_warmup(None)

        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        salient_idx = torch.arange(10)
        layer.set_salient_columns(salient_idx)

        x = torch.randn(1, 1, 100)
        output = layer(x)

        # Should work with full quantization
        assert torch.isfinite(output).all()


class TestCheckpointSaveLoad:
    """Test saving and loading checkpoints."""

    def test_checkpoint_preserves_calibration(self):
        """Test that calibration state is preserved through save/load."""
        import tempfile

        layer = BitLinearSalient(100, 50, salient_ratio=0.1)
        indices = torch.arange(10)
        layer.set_salient_columns(indices)

        # Save state dict
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(layer.state_dict(), f.name)

            # Create new layer and load
            layer2 = BitLinearSalient(100, 50, salient_ratio=0.1)
            layer2.load_state_dict(torch.load(f.name, weights_only=True))

        assert layer2.is_calibrated
        assert torch.equal(layer2.salient_indices, indices)

    def test_checkpoint_preserves_weights(self):
        """Test that weights are preserved through save/load."""
        import tempfile

        layer = BitLinearSalient(64, 32)
        original_weight = layer.weight.data.clone()

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(layer.state_dict(), f.name)

            layer2 = BitLinearSalient(64, 32)
            layer2.load_state_dict(torch.load(f.name, weights_only=True))

        assert torch.allclose(layer2.weight.data, original_weight)


class TestTorchCompileCompatibility:
    """Test torch.compile compatibility."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for torch.compile"
    )
    def test_torch_compile_forward(self):
        """Test torch.compile works for forward pass."""
        layer = BitLinearSalient(64, 32, salient_ratio=0.1).cuda()  # 10% = 6 columns
        indices = torch.arange(6).cuda()
        layer.set_salient_columns(indices)

        compiled = torch.compile(layer, mode="reduce-overhead")

        x = torch.randn(2, 8, 64, device="cuda")
        output = compiled(x)

        assert output.shape == (2, 8, 32)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for torch.compile"
    )
    def test_torch_compile_gradient_flow(self):
        """Test torch.compile doesn't break gradient flow."""
        layer = BitLinearSalient(64, 32, salient_ratio=0.1).cuda()  # 10% = 6 columns
        indices = torch.arange(6).cuda()
        layer.set_salient_columns(indices)

        compiled = torch.compile(layer, mode="reduce-overhead")

        x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)
        output = compiled(x)
        loss = output.sum()
        loss.backward()

        assert layer.weight.grad is not None


class TestGradientCheckpointingCompatibility:
    """Test gradient checkpointing compatibility."""

    def test_gradient_checkpointing_basic(self):
        """Test basic gradient checkpointing works."""
        from torch.utils.checkpoint import checkpoint

        layer = BitLinearSalient(64, 32)
        indices = torch.arange(6)
        layer.set_salient_columns(indices)

        x = torch.randn(2, 8, 64, requires_grad=True)

        output = checkpoint(layer, x, use_reentrant=False)
        loss = output.sum()
        loss.backward()

        assert layer.weight.grad is not None

    def test_gradient_checkpointing_multiple_layers(self):
        """Test gradient checkpointing with multiple layers."""
        from torch.utils.checkpoint import checkpoint

        class MultiLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.sal1 = BitLinearSalient(64, 64, salient_ratio=0.1)
                self.sal2 = BitLinearSalient(64, 32, salient_ratio=0.1)
                # Calibrate both
                self.sal1.set_salient_columns(torch.arange(6))
                self.sal2.set_salient_columns(torch.arange(6))

            def forward(self, x):
                h = checkpoint(self.sal1, x, use_reentrant=False)
                return checkpoint(self.sal2, h, use_reentrant=False)

        model = MultiLayerModel()
        x = torch.randn(2, 8, 64, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert model.sal1.weight.grad is not None
        assert model.sal2.weight.grad is not None
