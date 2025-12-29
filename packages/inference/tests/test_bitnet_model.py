"""Tests for BitNet model loading and conversion."""

import pytest
import numpy as np


class TestNativeKernel:
    """Test native kernel correctness."""

    def test_kernel_builds(self):
        """Test that the native kernel builds successfully."""
        from wrinklefree_inference.kernels.native import build_kernel

        kernel = build_kernel()
        assert hasattr(kernel, "gemv")
        assert hasattr(kernel, "gemm")
        assert hasattr(kernel, "num_threads")
        assert kernel.num_threads() > 0

    def test_pack_unpack_roundtrip(self):
        """Test that pack/unpack is a perfect roundtrip."""
        from wrinklefree_inference.kernels.native import pack_weights, unpack_weights

        # Create random ternary weights
        weights = np.random.choice([-1, 0, 1], size=(256, 512)).astype(np.float32)

        # Pack and unpack
        packed = pack_weights(weights)
        unpacked = unpack_weights(packed)

        np.testing.assert_array_equal(weights, unpacked)

    def test_gemv_correctness(self):
        """Test GEMV produces correct results."""
        from wrinklefree_inference.kernels.native import (
            build_kernel,
            pack_weights,
            unpack_weights,
            quantize_activations,
        )

        kernel = build_kernel()

        for shape in [(128, 256), (512, 512), (1024, 2048)]:
            out_f, in_f = shape
            weights = np.random.choice([-1, 0, 1], size=shape).astype(np.float32)
            packed = pack_weights(weights)

            act = np.random.randn(in_f).astype(np.float32)
            act_i8, scale = quantize_activations(act)

            # Native kernel result
            out_native = kernel.gemv(packed, act_i8, scale)

            # Reference result
            out_ref = np.dot(weights, act_i8.astype(np.float32)) * scale

            # Check cosine similarity
            cosine = np.dot(out_native, out_ref) / (
                np.linalg.norm(out_native) * np.linalg.norm(out_ref)
            )
            assert cosine > 0.9999, f"Cosine similarity too low: {cosine}"

    def test_gemm_correctness(self):
        """Test GEMM produces correct results."""
        from wrinklefree_inference.kernels.native import (
            build_kernel,
            pack_weights,
            quantize_activations,
        )

        kernel = build_kernel()

        for batch, out_f, in_f in [(8, 256, 512), (32, 512, 1024)]:
            weights = np.random.choice([-1, 0, 1], size=(out_f, in_f)).astype(np.float32)
            packed = pack_weights(weights)

            act = np.random.randn(batch, in_f).astype(np.float32)
            act_i8, scale = quantize_activations(act)

            # Native kernel result
            out_native = kernel.gemm(packed, act_i8, scale)

            # Reference result
            out_ref = np.dot(act_i8.astype(np.float32), weights.T) * scale

            # Check cosine similarity for each batch
            for b in range(batch):
                cosine = np.dot(out_native[b], out_ref[b]) / (
                    np.linalg.norm(out_native[b]) * np.linalg.norm(out_ref[b])
                )
                assert cosine > 0.9999, f"Batch {b} cosine too low: {cosine}"


class TestHFWeightConversion:
    """Test HuggingFace weight format conversion."""

    def test_repack_hf_weights(self):
        """Test repacking HF format to kernel format."""
        from wrinklefree_inference.kernels.native import (
            repack_hf_weights,
            unpack_weights,
        )

        # Simulate HF packed weights [out/4, in]
        out_features = 256
        in_features = 512
        out_packed = out_features // 4

        # Create original ternary weights
        original = np.random.choice([-1, 0, 1], size=(out_features, in_features)).astype(np.float32)

        # Pack in HF format (along output dim)
        hf_packed = np.zeros((out_packed, in_features), dtype=np.uint8)
        for i in range(4):
            w = (original[i::4, :].astype(np.int32) + 1).clip(0, 2)
            hf_packed |= (w.astype(np.uint8) << (i * 2))

        # Repack to kernel format
        kernel_packed = repack_hf_weights(hf_packed)

        # Verify shape
        assert kernel_packed.shape == (out_features, in_features // 4)

        # Verify values
        unpacked = unpack_weights(kernel_packed)
        np.testing.assert_array_equal(original, unpacked)

    def test_repack_preserves_correctness(self):
        """Test that repacked weights produce correct GEMV results."""
        from wrinklefree_inference.kernels.native import (
            build_kernel,
            repack_hf_weights,
            quantize_activations,
        )

        kernel = build_kernel()

        # Create original ternary weights
        out_features = 256
        in_features = 512
        original = np.random.choice([-1, 0, 1], size=(out_features, in_features)).astype(np.float32)

        # Pack in HF format
        out_packed = out_features // 4
        hf_packed = np.zeros((out_packed, in_features), dtype=np.uint8)
        for i in range(4):
            w = (original[i::4, :].astype(np.int32) + 1).clip(0, 2)
            hf_packed |= (w.astype(np.uint8) << (i * 2))

        # Repack for kernel
        kernel_packed = repack_hf_weights(hf_packed)

        # Run GEMV
        act = np.random.randn(in_features).astype(np.float32)
        act_i8, scale = quantize_activations(act)

        out_native = kernel.gemv(kernel_packed, act_i8, scale)
        out_ref = np.dot(original, act_i8.astype(np.float32)) * scale

        cosine = np.dot(out_native, out_ref) / (
            np.linalg.norm(out_native) * np.linalg.norm(out_ref)
        )
        assert cosine > 0.9999, f"Cosine too low: {cosine}"


@pytest.mark.slow
class TestBitNetModel:
    """Integration tests for full model loading."""

    def test_load_config(self):
        """Test loading model config from HuggingFace."""
        from wrinklefree_inference.models.bitnet import BitNetConfig

        config = BitNetConfig.from_hf("microsoft/BitNet-b1.58-2B-4T")

        assert config.hidden_size == 2560
        assert config.intermediate_size == 6912
        assert config.num_hidden_layers == 30
        assert config.num_attention_heads == 20
        assert config.num_key_value_heads == 5
        assert config.vocab_size == 128256

    @pytest.mark.benchmark
    def test_load_single_layer(self):
        """Test loading and running a single layer."""
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from wrinklefree_inference.models.bitnet import BitNetConfig, BitNetLayer
        from wrinklefree_inference.kernels.native import build_kernel

        kernel = build_kernel()
        config = BitNetConfig.from_hf("microsoft/BitNet-b1.58-2B-4T")

        model_path = hf_hub_download("microsoft/BitNet-b1.58-2B-4T", "model.safetensors")

        with safe_open(model_path, framework="np") as f:
            layer_weights = {}
            prefix = "model.layers.0."
            for key in f.keys():
                if key.startswith(prefix):
                    layer_weights[key[len(prefix):]] = f.get_tensor(key)

        layer = BitNetLayer(config, layer_weights, kernel)

        # Run forward pass
        hidden = np.random.randn(config.hidden_size).astype(np.float32)
        output = layer.forward(hidden)

        assert output.shape == (config.hidden_size,)
        assert not np.isnan(output).any()
