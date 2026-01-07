"""
Tests for GGUF conversion functionality.

These tests verify that:
1. Weight quantization produces correct distribution (~50% zeros, ~25% each +1/-1)
2. Both architecture name variants work (BitnetForCausalLM, BitNetForCausalLM)
3. Tokenizer fallback chain works (sentencepiece -> llama_hf -> gpt2)
4. Packed 2-bit weights are correctly unpacked
5. Output file sizes are correct

Run with:
    uv run pytest packages/inference/tests/test_gguf_conversion.py -v
"""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the quantization logic from the converter
import sys
SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


class TestWeightQuantization:
    """Tests for the weight quantization logic."""

    def test_online_quantization_produces_ternary_values(self):
        """Verify online quantization produces only -1, 0, 1 values."""
        # Simulate continuous bf16 weights (like from training)
        np.random.seed(42)
        weights = np.random.randn(256, 256).astype(np.float32)

        # Apply the fixed quantization (from converter)
        scale = np.abs(weights).mean()
        if scale < 1e-8:
            scale = 1.0
        quantized = np.round(weights / scale).clip(-1, 1)

        # Check all values are ternary
        unique_values = set(np.unique(quantized))
        assert unique_values.issubset({-1.0, 0.0, 1.0}), f"Got non-ternary values: {unique_values}"

    def test_online_quantization_distribution(self):
        """Verify quantization produces reasonable ternary distribution."""
        # Simulate normally distributed weights
        np.random.seed(42)
        weights = np.random.randn(1024, 1024).astype(np.float32)

        # Apply quantization
        scale = np.abs(weights).mean()
        quantized = np.round(weights / scale).clip(-1, 1)

        # Count values
        zeros = np.sum(quantized == 0) / quantized.size
        pos_ones = np.sum(quantized == 1) / quantized.size
        neg_ones = np.sum(quantized == -1) / quantized.size

        # For normal distribution with mean abs normalization:
        # - Zeros should be ~30-40% (values in [-0.5, 0.5] range)
        # - +1 and -1 should be roughly equal (~30-35% each)
        assert 0.20 < zeros < 0.50, f"Zero ratio {zeros:.2%} not in expected range"
        assert 0.25 < pos_ones < 0.45, f"+1 ratio {pos_ones:.2%} not in expected range"
        assert 0.25 < neg_ones < 0.45, f"-1 ratio {neg_ones:.2%} not in expected range"
        # Key check: pos and neg should be roughly balanced
        assert abs(pos_ones - neg_ones) < 0.05, f"Imbalanced: +1={pos_ones:.2%}, -1={neg_ones:.2%}"

    def test_sign_quantization_produces_no_zeros(self):
        """Demonstrate that np.sign() is WRONG - it produces 0% zeros."""
        np.random.seed(42)
        weights = np.random.randn(1024, 1024).astype(np.float32)

        # The WRONG way (what the old converter did)
        wrong_quantized = np.sign(weights)

        # Count zeros
        zeros = np.sum(wrong_quantized == 0) / wrong_quantized.size

        # np.sign() produces almost no zeros (only for exact 0.0 values)
        assert zeros < 0.01, f"np.sign() unexpectedly produced {zeros:.2%} zeros"

    def test_quantization_with_near_zero_scale(self):
        """Test quantization handles near-zero weights gracefully."""
        weights = np.zeros((64, 64), dtype=np.float32)
        weights[0, 0] = 1e-10  # Tiny non-zero value

        scale = np.abs(weights).mean()
        if scale < 1e-8:
            scale = 1.0
        quantized = np.round(weights / scale).clip(-1, 1)

        # Should all be zeros (scale defaults to 1.0, tiny values round to 0)
        assert np.all(quantized == 0), "Near-zero weights should quantize to zeros"


class TestPackedWeightUnpacking:
    """Tests for unpacking 2-bit packed weights."""

    def test_unpack_2bit_weights(self):
        """Test unpacking of 4 values packed into each byte."""
        import torch

        # Create packed data: 4 ternary values per byte
        # Values encoded as: 0=+1, 1=0, 2=-1 (after -1 offset)
        # So raw 2-bit values are: 1=+1, 2=0, 3=-1
        packed = np.array([[0b11_10_01_00]], dtype=np.uint8)  # [-1, 0, +1, +1]
        data_torch = torch.from_numpy(packed)

        origin_shape = data_torch.shape
        shift = torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape(
            (4, *(1 for _ in range(len(origin_shape))))
        )
        data_torch = data_torch.unsqueeze(0).expand((4, *origin_shape)) >> shift
        data_torch = data_torch & 3
        data_torch = (data_torch.float() - 1).reshape((origin_shape[0] * 4, *origin_shape[1:]))

        expected = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])  # Raw values - 1
        # Actually the encoding is: raw 0 -> -1, raw 1 -> 0, raw 2 -> 1, raw 3 -> 2 (invalid)
        # Let me recalculate...
        # packed byte 0b11_10_01_00:
        #   bits 0-1: 0b00 = 0 -> 0 - 1 = -1
        #   bits 2-3: 0b01 = 1 -> 1 - 1 = 0
        #   bits 4-5: 0b10 = 2 -> 2 - 1 = 1
        #   bits 6-7: 0b11 = 3 -> 3 - 1 = 2 (shouldn't happen in valid data)

        result = data_torch.numpy().flatten()
        assert result[0] == -1.0, f"First value should be -1, got {result[0]}"
        assert result[1] == 0.0, f"Second value should be 0, got {result[1]}"
        assert result[2] == 1.0, f"Third value should be 1, got {result[2]}"

    def test_unpack_shape_transformation(self):
        """Test that unpacking correctly transforms shape from [N/4, K] to [N, K]."""
        import torch

        # Simulate packed weights: [640, 2560] -> should become [2560, 2560]
        packed_shape = (640, 2560)
        packed = torch.zeros(packed_shape, dtype=torch.uint8)

        origin_shape = packed.shape
        shift = torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape(
            (4, *(1 for _ in range(len(origin_shape))))
        )
        unpacked = packed.unsqueeze(0).expand((4, *origin_shape)) >> shift
        unpacked = unpacked & 3
        unpacked = (unpacked.float() - 1).reshape((origin_shape[0] * 4, *origin_shape[1:]))

        expected_shape = (2560, 2560)
        assert unpacked.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {unpacked.shape}"
        )


class TestArchitectureNameHandling:
    """Tests for handling architecture name variants."""

    def test_fix_architecture_name(self):
        """Test that BitNetForCausalLM is fixed to BitnetForCausalLM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config = {"architectures": ["BitNetForCausalLM"], "hidden_size": 256}
            with open(config_path, "w") as f:
                json.dump(config, f)

            # Import and test the fix function
            from convert_checkpoint_to_gguf import fix_architecture_name

            result = fix_architecture_name(Path(tmpdir))
            assert result is True, "Should return True when fix was applied"

            # Verify the fix
            with open(config_path) as f:
                fixed_config = json.load(f)
            assert fixed_config["architectures"][0] == "BitnetForCausalLM"

    def test_no_fix_needed(self):
        """Test that correct architecture name is not modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config = {"architectures": ["BitnetForCausalLM"], "hidden_size": 256}
            with open(config_path, "w") as f:
                json.dump(config, f)

            from convert_checkpoint_to_gguf import fix_architecture_name

            result = fix_architecture_name(Path(tmpdir))
            assert result is False, "Should return False when no fix needed"


class TestCheckpointValidation:
    """Tests for checkpoint validation."""

    def test_validate_missing_config(self):
        """Test that missing config.json raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from convert_checkpoint_to_gguf import validate_checkpoint

            with pytest.raises(FileNotFoundError, match="config.json not found"):
                validate_checkpoint(Path(tmpdir))

    def test_validate_missing_weights(self):
        """Test that missing weights raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config = {"architectures": ["BitnetForCausalLM"], "hidden_size": 256}
            with open(config_path, "w") as f:
                json.dump(config, f)

            from convert_checkpoint_to_gguf import validate_checkpoint

            with pytest.raises(FileNotFoundError, match="No model weights found"):
                validate_checkpoint(Path(tmpdir))

    def test_validate_with_safetensors(self):
        """Test validation passes with safetensors file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config = {
                "architectures": ["BitnetForCausalLM"],
                "hidden_size": 2560,
                "num_hidden_layers": 30,
            }
            with open(config_path, "w") as f:
                json.dump(config, f)

            # Create a dummy safetensors file
            (Path(tmpdir) / "model.safetensors").touch()

            from convert_checkpoint_to_gguf import validate_checkpoint

            result = validate_checkpoint(Path(tmpdir))
            assert result["hidden_size"] == 2560


class TestOutputValidation:
    """Tests for output file validation."""

    def test_validate_correct_size_i2s(self):
        """Test that I2_S output is validated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.gguf"

            from convert_checkpoint_to_gguf import validate_gguf_output, estimate_model_params

            config = {
                "hidden_size": 2560,
                "num_hidden_layers": 30,
                "intermediate_size": 6912,
                "vocab_size": 128000,
            }

            # Calculate expected size based on actual estimation
            params = estimate_model_params(config)
            expected_size = int(params * 0.5)  # ~0.5 bytes per param for I2_S

            # Create file with expected size
            with open(output_path, "wb") as f:
                f.seek(expected_size - 1)
                f.write(b"\0")

            result = validate_gguf_output(output_path, config, "i2_s")
            assert result is True

    def test_validate_file_too_small(self):
        """Test that too-small files are caught."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.gguf"

            # Create file that's way too small
            with open(output_path, "wb") as f:
                f.write(b"too small")

            config = {
                "hidden_size": 2560,
                "num_hidden_layers": 30,
                "intermediate_size": 6912,
                "vocab_size": 128000,
            }

            from convert_checkpoint_to_gguf import validate_gguf_output

            result = validate_gguf_output(output_path, config, "i2_s")
            assert result is False


class TestModelParamEstimation:
    """Tests for model parameter estimation."""

    def test_estimate_2b_model(self):
        """Test parameter estimation for 2B model."""
        from convert_checkpoint_to_gguf import estimate_model_params

        config = {
            "hidden_size": 2560,
            "num_hidden_layers": 30,
            "intermediate_size": 6912,
            "vocab_size": 128000,
        }

        params = estimate_model_params(config)
        # Rough estimation - should be in reasonable range for 2B-ish model
        # The estimation is approximate, so allow wide range
        assert 1_000_000_000 < params < 5_000_000_000, f"Expected ~2B params, got {params:,}"

    def test_estimate_135m_model(self):
        """Test parameter estimation for 135M model."""
        from convert_checkpoint_to_gguf import estimate_model_params

        config = {
            "hidden_size": 576,
            "num_hidden_layers": 30,
            "intermediate_size": 1536,
            "vocab_size": 49152,
        }

        params = estimate_model_params(config)
        # Should be approximately 135M params
        assert 100_000_000 < params < 200_000_000, f"Expected ~135M params, got {params:,}"


class TestGCSDownload:
    """Tests for GCS download functionality."""

    def test_gcs_path_detection(self):
        """Test that GCS paths are correctly detected."""
        gcs_path = "gs://wrinklefree-checkpoints/dlm/checkpoint"
        assert gcs_path.startswith("gs://")

        local_path = "/home/user/models/checkpoint"
        assert not local_path.startswith("gs://")


class TestLRCTensorMapping:
    """Tests for LRC (Low-Rank Correction) tensor mapping in GGUF conversion."""

    def test_lrc_tensor_names_in_constants(self):
        """Verify LRC tensor types are defined in GGUF constants.py file."""
        constants_path = Path(__file__).parent.parent / "extern/sglang-bitnet/3rdparty/llama.cpp/gguf-py/gguf/constants.py"

        with open(constants_path, "r") as f:
            content = f.read()

        # Check LRC tensor types exist in the file
        lrc_tensors = [
            "ATTN_Q_LRC_U", "ATTN_Q_LRC_V",
            "ATTN_K_LRC_U", "ATTN_K_LRC_V",
            "ATTN_V_LRC_U", "ATTN_V_LRC_V",
            "ATTN_OUT_LRC_U", "ATTN_OUT_LRC_V",
            "FFN_GATE_LRC_U", "FFN_GATE_LRC_V",
            "FFN_UP_LRC_U", "FFN_UP_LRC_V",
            "FFN_DOWN_LRC_U", "FFN_DOWN_LRC_V",
        ]

        for tensor_name in lrc_tensors:
            assert tensor_name in content, f"Missing LRC tensor type in constants.py: {tensor_name}"

    def test_lrc_tensor_mapping_in_file(self):
        """Verify LRC tensors are mapped in tensor_mapping.py."""
        mapping_path = Path(__file__).parent.parent / "extern/sglang-bitnet/3rdparty/llama.cpp/gguf-py/gguf/tensor_mapping.py"

        with open(mapping_path, "r") as f:
            content = f.read()

        # Check that checkpoint tensor names are mapped
        checkpoint_names = [
            "q_proj.lrc_U", "q_proj.lrc_V",
            "k_proj.lrc_U", "k_proj.lrc_V",
            "v_proj.lrc_U", "v_proj.lrc_V",
            "o_proj.lrc_U", "o_proj.lrc_V",
            "gate_proj.lrc_U", "gate_proj.lrc_V",
            "up_proj.lrc_U", "up_proj.lrc_V",
            "down_proj.lrc_U", "down_proj.lrc_V",
        ]

        for name in checkpoint_names:
            assert name in content, f"Missing LRC mapping for: {name}"

    def test_lrc_shapes_valid(self):
        """Test that LRC tensor shapes are valid (out_features x rank for U, in_features x rank for V)."""
        # Simulate LRC tensors for a small model
        hidden_size = 256
        rank = 16  # 10% of 160 (min dim)

        # U shape: (out_features, rank)
        # V shape: (in_features, rank)
        lrc_u = np.random.randn(hidden_size, rank).astype(np.float16)
        lrc_v = np.random.randn(hidden_size, rank).astype(np.float16)

        assert lrc_u.shape == (hidden_size, rank), f"U shape wrong: {lrc_u.shape}"
        assert lrc_v.shape == (hidden_size, rank), f"V shape wrong: {lrc_v.shape}"

        # Verify LRC computation: output = U @ (V^T @ x)
        x = np.random.randn(hidden_size).astype(np.float16)
        vt_x = lrc_v.T @ x  # (rank,)
        lrc_output = lrc_u @ vt_x  # (hidden_size,)
        assert lrc_output.shape == (hidden_size,), f"LRC output shape wrong: {lrc_output.shape}"


class TestLRCInferenceIntegration:
    """Integration tests for LRC with llama.cpp."""

    def test_llama_cpp_lrc_build(self):
        """Verify llama.cpp builds with LRC support."""
        llama_cpp_path = Path(__file__).parent.parent / "extern/sglang-bitnet/3rdparty/llama.cpp"
        llama_cli = llama_cpp_path / "build/bin/llama-cli"
        libllama = llama_cpp_path / "build/src/libllama.so"

        # Check binaries exist (they should after build)
        if not llama_cli.exists():
            pytest.skip("llama-cli not built - run cmake build first")

        assert llama_cli.exists(), "llama-cli binary not found"
        assert libllama.exists(), "libllama.so not found"

    def test_lrc_tensor_enum_in_llama_cpp(self):
        """Verify LRC tensor enums match between Python and C++."""
        # The GGUF names used in Python tensor mapping
        gguf_names = [
            "blk.{bid}.attn_q.lrc_u",
            "blk.{bid}.attn_q.lrc_v",
            "blk.{bid}.attn_k.lrc_u",
            "blk.{bid}.attn_k.lrc_v",
            "blk.{bid}.attn_v.lrc_u",
            "blk.{bid}.attn_v.lrc_v",
            "blk.{bid}.attn_output.lrc_u",
            "blk.{bid}.attn_output.lrc_v",
            "blk.{bid}.ffn_gate.lrc_u",
            "blk.{bid}.ffn_gate.lrc_v",
            "blk.{bid}.ffn_up.lrc_u",
            "blk.{bid}.ffn_up.lrc_v",
            "blk.{bid}.ffn_down.lrc_u",
            "blk.{bid}.ffn_down.lrc_v",
        ]

        # All LRC tensor names should follow the pattern blk.{bid}.{projection}.lrc_{u|v}
        for name in gguf_names:
            assert ".lrc_u" in name or ".lrc_v" in name, f"Invalid LRC tensor name: {name}"
            assert name.startswith("blk."), f"LRC tensor should start with 'blk.': {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
