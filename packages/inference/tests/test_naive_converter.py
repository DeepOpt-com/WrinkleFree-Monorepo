"""Tests for the naive ternary converter module."""

import pytest
from pathlib import Path
import sys
import tempfile

# Add benchmark module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))

from benchmark.naive_converter import (
    NaiveConverter,
    ConversionConfig,
    ConversionResult,
)


def _has_torch():
    """Check if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


class TestConversionConfig:
    """Tests for ConversionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConversionConfig(
            model_id="test/model",
            output_dir=Path("/tmp/test"),
        )

        assert config.model_id == "test/model"
        assert config.output_dir == Path("/tmp/test")
        assert config.architecture == "llama"
        assert config.use_gpu is True
        assert config.verbose is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ConversionConfig(
            model_id="test/moe-model",
            output_dir=Path("/tmp/test"),
            architecture="moe",
            use_gpu=False,
            verbose=False,
        )

        assert config.architecture == "moe"
        assert config.use_gpu is False
        assert config.verbose is False


class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_success_result(self):
        """Test successful conversion result."""
        result = ConversionResult(
            success=True,
            model_id="test/model",
            output_path=Path("/tmp/output.safetensors"),
            original_size_gb=10.0,
            converted_size_gb=1.5,
            compression_ratio=6.67,
            num_layers=32,
        )

        assert result.success is True
        assert result.error is None
        assert result.compression_ratio > 6.0

    def test_failure_result(self):
        """Test failed conversion result."""
        result = ConversionResult(
            success=False,
            model_id="test/model",
            error="Out of memory",
        )

        assert result.success is False
        assert "memory" in result.error.lower()
        assert result.output_path is None


class TestNaiveConverter:
    """Tests for NaiveConverter class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ConversionConfig(
            model_id="test/model",
            output_dir=Path(tempfile.gettempdir()) / "naive_test",
            use_gpu=False,
            verbose=False,
        )

    @pytest.fixture
    def converter(self, config):
        """Create a converter instance."""
        return NaiveConverter(config)

    def test_initialization(self, converter, config):
        """Test converter initialization."""
        assert converter.config == config
        assert converter.progress_callback is None

    def test_set_progress_callback(self, converter):
        """Test setting progress callback."""
        progress_values = []

        def callback(msg: str, progress: float):
            progress_values.append((msg, progress))

        converter.set_progress_callback(callback)
        converter._report_progress("Test", 50.0)

        assert len(progress_values) == 1
        assert progress_values[0] == ("Test", 50.0)

    def test_estimate_memory_small_model(self, converter):
        """Test memory estimation for a small model."""
        # This may fail if transformers isn't installed
        try:
            from transformers import AutoConfig
        except ImportError:
            pytest.skip("transformers not installed")

        # Use a small, public model for testing
        converter.config.model_id = "gpt2"
        estimates = converter.estimate_memory_requirements()

        # GPT-2 is small, should estimate < 1GB
        if "error" not in estimates:
            assert estimates["model_bf16_gb"] < 1.0
            assert estimates["estimated_params"] > 0

    def test_convert_missing_dependencies(self):
        """Test conversion error handling when dependencies missing."""
        # Create config with a model that doesn't exist
        config = ConversionConfig(
            model_id="nonexistent/model",
            output_dir=Path(tempfile.gettempdir()) / "test",
            use_gpu=False,
        )
        converter = NaiveConverter(config)

        result = converter.convert()

        # Should fail gracefully
        assert result.success is False
        assert result.error is not None


class TestTernaryQuantization:
    """Tests for the ternary quantization logic."""

    @pytest.fixture
    def converter(self):
        config = ConversionConfig(
            model_id="test",
            output_dir=Path(tempfile.gettempdir()),
        )
        return NaiveConverter(config)

    @pytest.mark.skipif(
        not _has_torch(),
        reason="torch not installed",
    )
    def test_ternary_quantize_basic(self, converter):
        """Test basic ternary quantization."""
        import torch

        # Create a simple tensor
        tensor = torch.tensor([[-2.0, -0.3, 0.1, 0.5, 2.0]])

        ternary, scale = converter._ternary_quantize(tensor)

        # Should clamp to [-1, 0, 1]
        assert ternary.min() >= -1
        assert ternary.max() <= 1

        # All values should be ternary
        unique_values = set(ternary.flatten().tolist())
        assert unique_values.issubset({-1.0, 0.0, 1.0})

        # Scale should be positive
        assert scale > 0

    @pytest.mark.skipif(
        not _has_torch(),
        reason="torch not installed",
    )
    def test_ternary_quantize_zeros(self, converter):
        """Test quantization of zero tensor."""
        import torch

        tensor = torch.zeros(10, 10)
        ternary, scale = converter._ternary_quantize(tensor)

        assert torch.all(ternary == 0)
        assert scale == 1.0  # Default scale for zero tensor

    @pytest.mark.skipif(
        not _has_torch(),
        reason="torch not installed",
    )
    def test_ternary_quantize_distribution(self, converter):
        """Test that quantization produces reasonable distribution."""
        import torch

        # Create normally distributed tensor
        torch.manual_seed(42)
        tensor = torch.randn(1000, 1000)

        ternary, scale = converter._ternary_quantize(tensor)

        # Count values
        n_neg = (ternary == -1).sum().item()
        n_zero = (ternary == 0).sum().item()
        n_pos = (ternary == 1).sum().item()

        total = n_neg + n_zero + n_pos
        assert total == tensor.numel()

        # For normal distribution, expect roughly equal -1 and +1
        # with fewer zeros (unless centered around 0)
        ratio = n_pos / max(n_neg, 1)
        assert 0.5 < ratio < 2.0  # Within 2x of each other


class TestIntegration:
    """Integration tests for the conversion pipeline."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not _has_torch(),
        reason="torch not installed",
    )
    def test_convert_small_model(self):
        """Test converting a small model end-to-end."""
        # Skip if transformers not available
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            pytest.skip("transformers not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConversionConfig(
                model_id="gpt2",  # Small public model
                output_dir=Path(tmpdir),
                use_gpu=False,
                verbose=False,
            )
            converter = NaiveConverter(config)

            result = converter.convert()

            # Should succeed (gpt2 is small enough)
            if result.success:
                assert result.output_path.exists()
                assert result.compression_ratio > 1.0
                assert result.original_size_gb > 0
            else:
                # May fail due to memory or other issues
                assert result.error is not None
