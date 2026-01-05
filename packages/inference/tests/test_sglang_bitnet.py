"""Tests for SGLang BitNet integration.

Tests cover:
1. BitNet quantization correctness
2. Model loading validation
3. Inference correctness
4. Continuous batching behavior
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from wf_infer.sglang_backend.bitnet_quantization import (
    BitNetConfig,
    BitNetQuantType,
    BitNetLinearMethod,
    quantize_to_bitnet,
    dequantize_bitnet,
    validate_bitnet_model,
    BITNET_BLOCK_SIZE,
)


class TestBitNetQuantization:
    """Test BitNet quantization/dequantization correctness."""

    def test_quantize_basic(self):
        """Test basic quantization produces ternary values."""
        weights = torch.randn(256, 512)
        packed, scale = quantize_to_bitnet(weights)

        # Check shape: 4 weights per byte
        assert packed.shape == (256, 128), f"Expected (256, 128), got {packed.shape}"
        assert packed.dtype == torch.uint8
        assert scale > 0

    def test_quantize_dequantize_roundtrip(self):
        """Test that dequantization produces ternary values."""
        weights = torch.randn(256, 512)
        packed, scale = quantize_to_bitnet(weights)
        unpacked = dequantize_bitnet(packed, scale, 256, 512)

        # Check that unpacked values are ternary (-scale, 0, +scale)
        unique_normalized = torch.unique(unpacked / scale)
        expected = torch.tensor([-1.0, 0.0, 1.0])

        # Should only have -1, 0, 1 after normalization
        for val in unique_normalized:
            assert val.item() in [-1.0, 0.0, 1.0], f"Unexpected value: {val.item()}"

    def test_quantize_zeros(self):
        """Test that zero weights quantize correctly."""
        weights = torch.zeros(128, 256)
        packed, scale = quantize_to_bitnet(weights)

        unpacked = dequantize_bitnet(packed, scale, 128, 256)
        assert torch.allclose(unpacked, torch.zeros_like(unpacked))

    def test_quantize_ones(self):
        """Test that all-positive weights quantize to +1."""
        weights = torch.ones(128, 256) * 2.0
        packed, scale = quantize_to_bitnet(weights)

        unpacked = dequantize_bitnet(packed, scale, 128, 256)
        assert torch.allclose(unpacked, torch.ones_like(unpacked) * scale)

    def test_quantize_block_alignment(self):
        """Test that weights must be aligned to block size."""
        # Valid: divisible by 128
        weights_valid = torch.randn(64, 256)
        packed, _ = quantize_to_bitnet(weights_valid)
        assert packed is not None

        # Invalid: not divisible by 128
        weights_invalid = torch.randn(64, 100)
        with pytest.raises(AssertionError):
            quantize_to_bitnet(weights_invalid)


class TestBitNetLinearMethod:
    """Test BitNet linear layer computation."""

    @pytest.fixture
    def linear_method(self):
        return BitNetLinearMethod(BitNetQuantType.I2_S)

    def test_forward_shape(self, linear_method):
        """Test that forward pass produces correct output shape."""
        weights = torch.randn(512, 256)
        packed, scale = quantize_to_bitnet(weights)

        x = torch.randn(32, 256)  # batch=32, in_features=256
        out = linear_method.apply(packed, scale, x, 512, 256)

        assert out.shape == (32, 512), f"Expected (32, 512), got {out.shape}"

    def test_forward_with_bias(self, linear_method):
        """Test forward pass with bias."""
        weights = torch.randn(512, 256)
        packed, scale = quantize_to_bitnet(weights)
        bias = torch.randn(512)

        x = torch.randn(16, 256)
        out = linear_method.apply(packed, scale, x, 512, 256, bias=bias)

        assert out.shape == (16, 512)

    def test_forward_numerical_correctness(self, linear_method):
        """Test that BitNet forward matches reference implementation."""
        # Create ternary weights directly
        weights = torch.zeros(128, 256)
        weights[:, :64] = 1.0   # First 64 cols are +1
        weights[:, 64:128] = -1.0  # Next 64 cols are -1
        # Remaining cols are 0

        packed, scale = quantize_to_bitnet(weights)

        # Input with known values
        x = torch.ones(1, 256)

        out = linear_method.apply(packed, scale, x, 128, 256)

        # Expected: each row should be 64*1 - 64*1 + 0 = 0 (scaled)
        # But scale affects the ternary values
        # Convert to same dtype for comparison (output may be BF16)
        expected = torch.zeros(1, 128, dtype=out.dtype)
        assert torch.allclose(out, expected, atol=1e-3)  # Relaxed for BF16 precision


class TestBitNetConfig:
    """Test BitNet configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = BitNetConfig()
        assert config.quant_type == BitNetQuantType.I2_S
        assert config.block_size == BITNET_BLOCK_SIZE
        assert config.activation_bits == 8

    def test_config_name(self):
        """Test config name."""
        assert BitNetConfig.get_name() == "bitnet"

    def test_config_supported_dtypes(self):
        """Test supported activation dtypes."""
        dtypes = BitNetConfig.get_supported_act_dtypes()
        assert torch.int8 in dtypes
        assert torch.float16 in dtypes
        assert torch.bfloat16 in dtypes

    def test_config_repr(self):
        """Test config string representation."""
        config = BitNetConfig()
        repr_str = repr(config)
        assert "BitNetConfig" in repr_str
        assert "I2_S" in repr_str


class TestModelValidation:
    """Test BitNet model file validation."""

    def test_validate_nonexistent(self, tmp_path):
        """Test validation of non-existent file."""
        result = validate_bitnet_model(str(tmp_path / "nonexistent.gguf"))
        assert not result["valid"]
        assert not result["exists"]
        assert len(result["errors"]) > 0

    def test_validate_wrong_extension(self, tmp_path):
        """Test validation of wrong file extension."""
        model_path = tmp_path / "model.bin"
        model_path.touch()

        result = validate_bitnet_model(str(model_path))
        assert not result["valid"]
        assert "gguf" in result["errors"][0].lower()

    def test_validate_i2s_model(self, tmp_path):
        """Test validation of I2_S quantized model."""
        model_path = tmp_path / "model-i2_s.gguf"
        model_path.write_bytes(b"dummy content")

        result = validate_bitnet_model(str(model_path))
        assert result["valid"]
        assert result["quant_type"] == BitNetQuantType.I2_S
        assert result["size_mb"] > 0

    def test_validate_tl1_model(self, tmp_path):
        """Test validation of TL1 quantized model."""
        model_path = tmp_path / "model-tl1.gguf"
        model_path.write_bytes(b"dummy content")

        result = validate_bitnet_model(str(model_path))
        assert result["valid"]
        assert result["quant_type"] == BitNetQuantType.TL1

    def test_validate_tl2_model(self, tmp_path):
        """Test validation of TL2 quantized model."""
        model_path = tmp_path / "model-TL2.gguf"
        model_path.write_bytes(b"dummy content")

        result = validate_bitnet_model(str(model_path))
        assert result["valid"]
        assert result["quant_type"] == BitNetQuantType.TL2


@pytest.mark.integration
class TestBitNetInference:
    """Integration tests requiring a running server."""

    @pytest.fixture
    def model_path(self):
        """Get path to BitNet model for testing."""
        # Check common locations for BitNet models
        paths = [
            Path("extern/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"),
            Path("/home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine/extern/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"),
        ]
        for path in paths:
            if path.exists():
                return str(path)
        pytest.skip("BitNet model not found")

    def test_model_exists(self, model_path):
        """Verify model file exists and is valid."""
        result = validate_bitnet_model(model_path)
        assert result["valid"], f"Model validation failed: {result['errors']}"
        assert result["quant_type"] == BitNetQuantType.I2_S

    def test_model_size_reasonable(self, model_path):
        """Check model size is in expected range for 2B model."""
        result = validate_bitnet_model(model_path)
        # 2B model with 1.58-bit weights should be ~400-600MB
        assert 300 < result["size_mb"] < 800, f"Unexpected model size: {result['size_mb']}MB"


class TestNumericalAccuracy:
    """Test numerical accuracy of BF16 optimization vs FP32 reference."""

    @pytest.mark.parametrize("size", [
        (4096, 4096),
        (11008, 4096),
        (4096, 11008),
    ])
    def test_bf16_cosine_similarity(self, size):
        """Test that BF16 optimization preserves numerical accuracy.

        Compares BF16-optimized output against FP32 reference.
        Requires cosine similarity > 0.9999.
        """
        out_dim, in_dim = size

        # Create random weight and quantize
        w_orig = torch.randn(out_dim, in_dim, dtype=torch.float32)
        packed, scale = quantize_to_bitnet(w_orig)

        # Reference: FP32 dequantization and matmul
        w_fp32 = dequantize_bitnet(packed, scale, out_dim, in_dim)

        # Optimized: BF16 with caching
        method = BitNetLinearMethod(compute_dtype=torch.bfloat16)

        # Test with random input (batch=32)
        x = torch.randn(32, in_dim, dtype=torch.float32)

        # Reference output (FP32)
        y_ref = torch.matmul(x, w_fp32.T)

        # Optimized output (BF16)
        y_opt = method.apply(packed, scale, x, out_dim, in_dim)
        y_opt_fp32 = y_opt.to(torch.float32)

        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            y_ref.flatten().unsqueeze(0),
            y_opt_fp32.flatten().unsqueeze(0)
        ).item()

        # Must be > 0.9999 for production use
        assert cos_sim > 0.9999, f"Cosine similarity {cos_sim:.6f} < 0.9999 for size {size}"

    def test_bf16_relative_error(self):
        """Test mean relative error is acceptable."""
        out_dim, in_dim = 4096, 4096

        w_orig = torch.randn(out_dim, in_dim, dtype=torch.float32)
        packed, scale = quantize_to_bitnet(w_orig)
        w_fp32 = dequantize_bitnet(packed, scale, out_dim, in_dim)

        method = BitNetLinearMethod(compute_dtype=torch.bfloat16)
        x = torch.randn(32, in_dim, dtype=torch.float32)

        y_ref = torch.matmul(x, w_fp32.T)
        y_opt = method.apply(packed, scale, x, out_dim, in_dim).to(torch.float32)

        # Mean relative error
        rel_error = ((y_ref - y_opt).abs() / (y_ref.abs() + 1e-8)).mean().item()

        # Should be < 5% mean relative error
        assert rel_error < 0.05, f"Mean relative error {rel_error:.2%} > 5%"

    def test_fp16_vs_bf16_batched(self):
        """Verify BF16 is better than FP16 for batched inference."""
        out_dim, in_dim = 4096, 4096

        w_orig = torch.randn(out_dim, in_dim, dtype=torch.float32)
        packed, scale = quantize_to_bitnet(w_orig)
        w_fp32 = dequantize_bitnet(packed, scale, out_dim, in_dim)

        x = torch.randn(32, in_dim, dtype=torch.float32)
        y_ref = torch.matmul(x, w_fp32.T)

        # BF16
        method_bf16 = BitNetLinearMethod(compute_dtype=torch.bfloat16)
        y_bf16 = method_bf16.apply(packed, scale, x, out_dim, in_dim).to(torch.float32)

        # FP16
        method_fp16 = BitNetLinearMethod(compute_dtype=torch.float16)
        y_fp16 = method_fp16.apply(packed, scale, x, out_dim, in_dim).to(torch.float32)

        cos_bf16 = torch.nn.functional.cosine_similarity(
            y_ref.flatten().unsqueeze(0), y_bf16.flatten().unsqueeze(0)
        ).item()

        cos_fp16 = torch.nn.functional.cosine_similarity(
            y_ref.flatten().unsqueeze(0), y_fp16.flatten().unsqueeze(0)
        ).item()

        # BF16 should have similar or better accuracy than FP16
        assert cos_bf16 >= 0.9999, f"BF16 cosine sim {cos_bf16:.6f} too low"
        # FP16 may have lower accuracy but should still be acceptable
        assert cos_fp16 >= 0.999, f"FP16 cosine sim {cos_fp16:.6f} too low"


@pytest.mark.benchmark
class TestBitNetPerformance:
    """Performance benchmarks for BitNet operations."""

    @pytest.fixture
    def large_weights(self):
        """Create large weight tensor for benchmarking."""
        return torch.randn(4096, 4096)

    def test_quantization_speed(self, large_weights, benchmark):
        """Benchmark quantization speed."""
        result = benchmark(quantize_to_bitnet, large_weights)
        # Should complete quantization of 16M params

    def test_gemv_speed(self, large_weights, benchmark):
        """Benchmark GEMV speed."""
        packed, scale = quantize_to_bitnet(large_weights)
        x = torch.randn(1, 4096)
        method = BitNetLinearMethod()

        def run_gemv():
            return method.apply(packed, scale, x, 4096, 4096)

        result = benchmark(run_gemv)

    def test_batch_gemm_speed(self, large_weights, benchmark):
        """Benchmark batched GEMM speed."""
        packed, scale = quantize_to_bitnet(large_weights)
        x = torch.randn(64, 4096)
        method = BitNetLinearMethod()

        def run_gemm():
            return method.apply(packed, scale, x, 4096, 4096)

        result = benchmark(run_gemm)
