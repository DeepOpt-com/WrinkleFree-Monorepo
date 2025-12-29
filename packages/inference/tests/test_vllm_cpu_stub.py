"""Tests for vllm-cpu-stub and sglang-bitnet CPU integration.

Tests cover:
1. vllm-cpu-stub imports work correctly
2. PyTorch fallback implementations are correct
3. sglang can import without CUDA
4. Weight packing/unpacking functions work
"""

import pytest
import torch
import numpy as np


class TestVLLMCPUStubImports:
    """Test that vllm-cpu-stub imports work correctly."""

    def test_import_vllm(self):
        """Test basic vllm import."""
        import vllm
        assert hasattr(vllm, '__version__')
        assert 'cpu' in vllm.__version__

    def test_import_custom_ops(self):
        """Test vllm._custom_ops import."""
        from vllm import _custom_ops
        assert hasattr(_custom_ops, 'silu_and_mul')
        assert hasattr(_custom_ops, 'gelu_and_mul')
        assert hasattr(_custom_ops, 'rms_norm')
        assert hasattr(_custom_ops, 'rotary_embedding')

    def test_import_activation_layers(self):
        """Test vllm.model_executor.layers.activation import."""
        from vllm.model_executor.layers.activation import (
            SiluAndMul,
            GeluAndMul,
        )
        assert SiluAndMul is not None
        assert GeluAndMul is not None

    def test_import_layernorm_layers(self):
        """Test vllm.model_executor.layers.layernorm import."""
        from vllm.model_executor.layers.layernorm import (
            RMSNorm,
            GemmaRMSNorm,
        )
        assert RMSNorm is not None
        assert GemmaRMSNorm is not None

    def test_import_distributed(self):
        """Test vllm.distributed.parallel_state import."""
        from vllm.distributed import parallel_state
        assert hasattr(parallel_state, 'get_pp_group')
        assert hasattr(parallel_state, 'get_tp_group')
        assert hasattr(parallel_state, 'get_world_group')

    def test_import_logger(self):
        """Test vllm.logger import."""
        from vllm.logger import logger
        import logging
        assert isinstance(logger, logging.Logger)


class TestVLLMActivationOps:
    """Test vllm activation operations."""

    def test_silu_and_mul(self):
        """Test SiLU activation with gating."""
        from vllm import _custom_ops as ops

        # Create input with doubled features (for gating)
        x = torch.randn(2, 8)  # batch=2, features=8 (4 + 4 for gating)
        out = torch.zeros(2, 4)

        ops.silu_and_mul(out, x)

        # Verify output shape
        assert out.shape == (2, 4)

        # Verify correctness: out = silu(x[:4]) * x[4:]
        expected = torch.nn.functional.silu(x[..., :4]) * x[..., 4:]
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gelu_and_mul(self):
        """Test GELU activation with gating."""
        from vllm import _custom_ops as ops

        x = torch.randn(2, 8)
        out = torch.zeros(2, 4)

        ops.gelu_and_mul(out, x)

        expected = torch.nn.functional.gelu(x[..., :4]) * x[..., 4:]
        assert torch.allclose(out, expected, atol=1e-5)

    def test_silu_and_mul_module(self):
        """Test SiluAndMul as nn.Module."""
        from vllm.model_executor.layers.activation import SiluAndMul

        module = SiluAndMul()
        x = torch.randn(4, 16)  # batch=4, features=16 (8+8)
        out = module(x)

        assert out.shape == (4, 8)
        expected = torch.nn.functional.silu(x[..., :8]) * x[..., 8:]
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gelu_and_mul_module(self):
        """Test GeluAndMul as nn.Module."""
        from vllm.model_executor.layers.activation import GeluAndMul

        module = GeluAndMul()
        x = torch.randn(4, 16)
        out = module(x)

        assert out.shape == (4, 8)
        expected = torch.nn.functional.gelu(x[..., :8]) * x[..., 8:]
        assert torch.allclose(out, expected, atol=1e-5)


class TestVLLMLayerNorm:
    """Test vllm layer normalization implementations."""

    def test_rms_norm_basic(self):
        """Test basic RMSNorm functionality."""
        from vllm.model_executor.layers.layernorm import RMSNorm

        hidden_size = 256
        norm = RMSNorm(hidden_size, eps=1e-6)
        x = torch.randn(2, 10, hidden_size)

        out = norm(x)

        assert out.shape == x.shape
        # Verify normalization happened (variance should be ~1)
        var = out.pow(2).mean(-1)
        assert var.mean() > 0.5  # Normalized, but with weight applied

    def test_rms_norm_with_residual(self):
        """Test RMSNorm with residual connection."""
        from vllm.model_executor.layers.layernorm import RMSNorm

        hidden_size = 128
        norm = RMSNorm(hidden_size)
        x = torch.randn(2, 5, hidden_size)
        residual = torch.randn(2, 5, hidden_size)

        out, new_residual = norm(x, residual=residual)

        assert out.shape == x.shape
        assert new_residual.shape == residual.shape
        # Residual should be x + original residual
        expected_residual = x + residual
        assert torch.allclose(new_residual, expected_residual, atol=1e-5)

    def test_gemma_rms_norm(self):
        """Test Gemma-style RMSNorm with +1 weight offset."""
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm

        hidden_size = 256
        norm = GemmaRMSNorm(hidden_size, eps=1e-6)
        x = torch.randn(2, 10, hidden_size)

        out = norm(x)

        assert out.shape == x.shape
        # Weight is initialized to 0, so effective weight is 1.0
        # Output should be normalized

    def test_rms_norm_custom_ops(self):
        """Test rms_norm from _custom_ops."""
        from vllm import _custom_ops as ops

        x = torch.randn(2, 10, 256)
        weight = torch.ones(256)
        out = torch.zeros_like(x)

        ops.rms_norm(out, x, weight, 1e-6)

        # Verify normalization
        variance = x.pow(2).mean(-1, keepdim=True)
        expected = x * torch.rsqrt(variance + 1e-6) * weight
        assert torch.allclose(out, expected, atol=1e-4)


class TestVLLMDistributed:
    """Test vllm distributed stubs."""

    def test_get_groups(self):
        """Test process group getters return valid objects."""
        from vllm.distributed.parallel_state import (
            get_pp_group,
            get_tp_group,
            get_world_group,
        )

        pp_group = get_pp_group()
        tp_group = get_tp_group()
        world_group = get_world_group()

        # All should return fake process groups
        assert pp_group.rank() == 0
        assert tp_group.rank() == 0
        assert world_group.rank() == 0
        assert pp_group.size() == 1
        assert tp_group.size() == 1
        assert world_group.size() == 1

    def test_world_size_functions(self):
        """Test world size functions return 1 for CPU."""
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
            get_tensor_model_parallel_rank,
            get_pipeline_model_parallel_world_size,
            get_pipeline_model_parallel_rank,
        )

        assert get_tensor_model_parallel_world_size() == 1
        assert get_tensor_model_parallel_rank() == 0
        assert get_pipeline_model_parallel_world_size() == 1
        assert get_pipeline_model_parallel_rank() == 0


class TestSGLangBitNetWeightPacking:
    """Test weight packing functions from sglang-bitnet."""

    @pytest.fixture
    def random_ternary_weights(self):
        """Create random ternary float weights."""
        # Create weights with values in {-1, 0, +1}
        weights = torch.randint(-1, 2, (256, 512)).float()
        return weights

    def test_pack_ternary_weights(self, random_ternary_weights):
        """Test packing ternary float weights to uint8."""
        try:
            from sglang.srt.models.bitnet import _pack_ternary_weights
        except ImportError:
            pytest.skip("sglang-bitnet not installed")

        weights = random_ternary_weights
        packed, scale = _pack_ternary_weights(weights)

        # Check shapes
        assert packed.shape == (256, 128)  # 512/4 = 128
        assert packed.dtype == torch.uint8
        assert scale.numel() == 1

    def test_pack_unpack_roundtrip(self, random_ternary_weights):
        """Test that pack/unpack preserves values."""
        try:
            from sglang.srt.models.bitnet import (
                _pack_ternary_weights,
                _unpack_i2_to_ternary,
            )
        except ImportError:
            pytest.skip("sglang-bitnet not installed")

        weights = random_ternary_weights
        packed, scale = _pack_ternary_weights(weights)
        unpacked = _unpack_i2_to_ternary(packed, weights.shape[0], weights.shape[1])

        # Scale the unpacked values
        unpacked_scaled = unpacked * scale

        # Should match original (within scale factor)
        assert torch.allclose(unpacked_scaled, weights * scale.item(), atol=1e-5)

    def test_is_packed_weight(self):
        """Test packed weight detection."""
        try:
            from sglang.srt.models.bitnet import _is_packed_weight
        except ImportError:
            pytest.skip("sglang-bitnet not installed")

        packed = torch.zeros(256, 128, dtype=torch.uint8)
        unpacked = torch.zeros(256, 512, dtype=torch.float32)

        assert _is_packed_weight(packed) == True
        assert _is_packed_weight(unpacked) == False

    def test_is_ternary_float(self):
        """Test ternary float weight detection."""
        try:
            from sglang.srt.models.bitnet import _is_ternary_float
        except ImportError:
            pytest.skip("sglang-bitnet not installed")

        ternary = torch.randint(-1, 2, (100, 100)).float()
        non_ternary = torch.randn(100, 100)

        assert _is_ternary_float(ternary) == True
        assert _is_ternary_float(non_ternary) == False


class TestSGLKernelAvailability:
    """Test sgl-kernel native kernel availability."""

    def test_bitnet_kernel_check(self):
        """Test BitNet kernel availability check."""
        try:
            from sgl_kernel.quantization import bitnet_check_kernel_available
            available = bitnet_check_kernel_available()
            # On CPU with AVX2/AVX512, should be True
            # Just check it doesn't crash
            assert isinstance(available, bool)
        except ImportError:
            pytest.skip("sgl-kernel not installed")

    def test_simd_detection(self):
        """Test SIMD capability detection."""
        try:
            from sgl_kernel.quantization.bitnet import check_kernel_available
            available = check_kernel_available()
            print(f"BitNet SIMD kernels available: {available}")
            assert isinstance(available, bool)
        except ImportError:
            pytest.skip("sgl-kernel BitNet module not available")


@pytest.mark.integration
class TestSGLangServerImports:
    """Test that sglang server can be imported without CUDA."""

    def test_import_launch_server(self):
        """Test importing launch_server module."""
        try:
            from sglang import launch_server
            assert launch_server is not None
        except ImportError as e:
            if 'cuda' in str(e).lower() or 'libtorch_cuda' in str(e):
                pytest.fail(f"CUDA dependency leaked through: {e}")
            # Other import errors are acceptable (missing optional deps)
            pytest.skip(f"sglang not fully installed: {e}")

    def test_import_server_args(self):
        """Test importing server args."""
        try:
            from sglang.srt.server_args import ServerArgs
            assert ServerArgs is not None
        except ImportError as e:
            if 'cuda' in str(e).lower():
                pytest.fail(f"CUDA dependency: {e}")
            pytest.skip(f"sglang not installed: {e}")

    def test_import_bitnet_model(self):
        """Test importing BitNet model."""
        try:
            from sglang.srt.models.bitnet import BitNetForCausalLM
            assert BitNetForCausalLM is not None
        except ImportError as e:
            if 'cuda' in str(e).lower():
                pytest.fail(f"CUDA dependency: {e}")
            pytest.skip(f"BitNet model not available: {e}")
