"""Tests for model conversion utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wrinklefree_inference.converter.hf_to_gguf import ConversionConfig, HFToGGUFConverter


class TestConversionConfig:
    """Tests for ConversionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConversionConfig(hf_repo="microsoft/BitNet-b1.58-2B-4T")

        assert config.hf_repo == "microsoft/BitNet-b1.58-2B-4T"
        assert config.quant_type == "i2_s"
        assert config.quant_embd is False
        assert config.use_pretuned is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ConversionConfig(
            hf_repo="test/model",
            quant_type="tl2",
            quant_embd=True,
            output_dir=Path("/custom/path"),
        )

        assert config.quant_type == "tl2"
        assert config.quant_embd is True
        assert config.output_dir == Path("/custom/path")


class TestHFToGGUFConverter:
    """Tests for HFToGGUFConverter class."""

    def test_default_bitnet_path(self):
        """Test that default BitNet path is calculated correctly."""
        # If BitNet submodule is initialized, converter should work
        # If not, it should raise FileNotFoundError
        try:
            converter = HFToGGUFConverter()
            # Submodule exists - verify path is correct
            assert converter.bitnet_path.exists()
            assert (converter.bitnet_path / "setup_env.py").exists() or \
                   (converter.bitnet_path / "README.md").exists()
        except FileNotFoundError:
            # Expected if submodule not initialized
            pass

    def test_custom_bitnet_path(self, tmp_path):
        """Test converter with custom BitNet path."""
        # Create a mock BitNet directory
        bitnet_dir = tmp_path / "BitNet"
        bitnet_dir.mkdir()
        (bitnet_dir / "setup_env.py").write_text("# mock")

        converter = HFToGGUFConverter(bitnet_dir)
        assert converter.bitnet_path == bitnet_dir

    def test_validate_gguf_missing_file(self, tmp_path):
        """Test validation fails for missing file."""
        bitnet_dir = tmp_path / "BitNet"
        bitnet_dir.mkdir()
        (bitnet_dir / "setup_env.py").write_text("# mock")

        converter = HFToGGUFConverter(bitnet_dir)

        with pytest.raises(FileNotFoundError):
            converter._validate_gguf(tmp_path / "nonexistent.gguf")

    def test_validate_gguf_too_small(self, tmp_path):
        """Test validation fails for files that are too small."""
        bitnet_dir = tmp_path / "BitNet"
        bitnet_dir.mkdir()
        (bitnet_dir / "setup_env.py").write_text("# mock")

        converter = HFToGGUFConverter(bitnet_dir)

        # Create tiny file
        gguf_file = tmp_path / "tiny.gguf"
        gguf_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with pytest.raises(ValueError, match="too small"):
            converter._validate_gguf(gguf_file)

    def test_validate_gguf_wrong_magic(self, tmp_path):
        """Test validation fails for wrong magic bytes."""
        bitnet_dir = tmp_path / "BitNet"
        bitnet_dir.mkdir()
        (bitnet_dir / "setup_env.py").write_text("# mock")

        converter = HFToGGUFConverter(bitnet_dir)

        # Create file with wrong magic
        gguf_file = tmp_path / "wrong.gguf"
        gguf_file.write_bytes(b"XXXX" + b"\x00" * (20 * 1024 * 1024))

        with pytest.raises(ValueError, match="Invalid GGUF magic"):
            converter._validate_gguf(gguf_file)


class TestConversionIntegration:
    """Integration tests for conversion (require actual BitNet setup)."""

    @pytest.mark.integration
    def test_list_models(self, tmp_path):
        """Test listing available models."""
        bitnet_dir = tmp_path / "BitNet"
        bitnet_dir.mkdir()
        (bitnet_dir / "setup_env.py").write_text("# mock")

        # Create mock models directory
        models_dir = bitnet_dir / "models" / "test-model"
        models_dir.mkdir(parents=True)
        (models_dir / "ggml-model-i2_s.gguf").write_bytes(b"GGUF" + b"\x00" * 100)

        converter = HFToGGUFConverter(bitnet_dir)
        models = converter.list_available_models()

        assert len(models) == 1
        assert "test-model" in str(models[0])
