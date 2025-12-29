"""Fast unit tests for BitNet weight loading and conversion.

These tests catch the gibberish output bugs without requiring a server:
1. HF stores weights as [out/4, in] but kernel expects [out, in/4]
2. HF stores weight_scale as separate tensors (not 1.0)
3. Activation sum correction must account for weight_scale

Run with: uv run pytest tests/test_bitnet_weight_loading.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestHFFormatConversion:
    """Test HuggingFace pre-packed weight format conversion."""

    def test_hf_unpack_produces_ternary(self):
        """_unpack_ternary_weights should produce {-1, 0, +1} from HF uint8."""
        from sglang.srt.models.bitnet import _unpack_ternary_weights

        # Simulate HF packed weights: [out/4, in] uint8
        # Byte value 0b01010101 = 85 means all zeros (01 = 0 after -1 mapping)
        packed = torch.full((64, 256), 85, dtype=torch.uint8)

        unpacked = _unpack_ternary_weights(packed)

        assert unpacked.shape == (256, 256), f"Expected (256, 256), got {unpacked.shape}"
        assert unpacked.dtype == torch.float32
        assert torch.all(unpacked == 0), "All 0b01 bits should unpack to 0"

    def test_hf_unpack_all_values(self):
        """Test unpacking all possible 2-bit values."""
        from sglang.srt.models.bitnet import _unpack_ternary_weights

        # Create test pattern: each byte encodes 4 values
        # 0b00 = -1, 0b01 = 0, 0b10 = +1
        # Byte 0b10_01_00_11 = 0x93 encodes: [+1, 0, -1, invalid->+1]
        # But HF uses: bits 0-1 -> row 0, bits 2-3 -> row n, etc.

        # Test with simple pattern: byte 0b00_00_00_00 = 0 -> all -1
        packed_neg1 = torch.zeros((1, 4), dtype=torch.uint8)
        unpacked = _unpack_ternary_weights(packed_neg1)
        assert unpacked.shape == (4, 4)
        assert torch.all(unpacked == -1), f"Expected all -1, got unique: {torch.unique(unpacked)}"

        # Byte 0b01_01_01_01 = 85 -> all 0
        packed_zero = torch.full((1, 4), 85, dtype=torch.uint8)
        unpacked = _unpack_ternary_weights(packed_zero)
        assert torch.all(unpacked == 0), f"Expected all 0, got unique: {torch.unique(unpacked)}"

        # Byte 0b10_10_10_10 = 170 -> all +1
        packed_pos1 = torch.full((1, 4), 170, dtype=torch.uint8)
        unpacked = _unpack_ternary_weights(packed_pos1)
        assert torch.all(unpacked == 1), f"Expected all +1, got unique: {torch.unique(unpacked)}"

    def test_hf_to_kernel_format_conversion(self):
        """Test full HF [out/4, in] -> kernel [out, in/4] conversion."""
        from sglang.srt.models.bitnet import (
            _unpack_ternary_weights,
            _pack_ternary_weights,
            _unpack_i2_to_ternary,
        )

        # Create random HF-format packed weights
        out_features, in_features = 256, 512
        hf_packed = torch.randint(0, 256, (out_features // 4, in_features), dtype=torch.uint8)

        # Step 1: Unpack HF format
        unpacked = _unpack_ternary_weights(hf_packed)
        assert unpacked.shape == (out_features, in_features)
        unique = torch.unique(unpacked)
        assert all(v in [-1, 0, 1] for v in unique.tolist()), f"Non-ternary values: {unique}"

        # Step 2: Repack to kernel format
        kernel_packed, scale = _pack_ternary_weights(unpacked)
        assert kernel_packed.shape == (out_features, in_features // 4)
        assert kernel_packed.dtype == torch.uint8

        # Step 3: Verify roundtrip
        reunpacked = _unpack_i2_to_ternary(kernel_packed, out_features, in_features)
        assert reunpacked.shape == unpacked.shape
        assert torch.allclose(reunpacked, unpacked), "HF -> kernel -> unpack roundtrip failed!"


class TestWeightScaleHandling:
    """Test that weight_scale from HF is correctly preserved."""

    def test_weight_scale_not_overwritten(self):
        """Ensure HF's weight_scale is not replaced with 1.0."""
        from sglang.srt.models.bitnet import BitNetLinear

        linear = BitNetLinear(256, 512)

        # Simulate HF weight_scale (non-trivial value)
        hf_scale = torch.tensor([1.7969])
        linear.weight_scale.data.copy_(hf_scale)

        assert linear.weight_scale.item() == pytest.approx(1.7969, rel=1e-3), \
            f"weight_scale should be 1.7969, got {linear.weight_scale.item()}"

    def test_forward_uses_weight_scale(self):
        """Test that forward() correctly applies weight_scale."""
        from sglang.srt.models.bitnet import (
            BitNetLinear,
            _pack_ternary_weights,
        )

        out_features, in_features = 128, 256

        # Create ternary weights
        weight = torch.randint(-1, 2, (out_features, in_features)).float()

        # Create two identical layers with different scales
        linear1 = BitNetLinear(in_features, out_features)
        linear2 = BitNetLinear(in_features, out_features)

        packed, _ = _pack_ternary_weights(weight)
        linear1.qweight.data.copy_(packed)
        linear2.qweight.data.copy_(packed)

        # Different weight scales
        linear1.weight_scale.data.fill_(1.0)
        linear2.weight_scale.data.fill_(2.0)

        x = torch.randn(1, in_features)
        out1 = linear1(x)
        out2 = linear2(x)

        # Output should scale proportionally
        # Note: Due to quantization, this won't be exact 2x, but should be close
        ratio = (out2.abs().mean() / out1.abs().mean()).item()
        assert 1.5 < ratio < 2.5, f"Expected ~2x ratio with 2x scale, got {ratio}"


class TestActivationSumCorrection:
    """Test the activation sum correction in forward pass."""

    def test_activation_sum_with_scale(self):
        """Test that activation sum correction accounts for weight_scale."""
        from sglang.srt.models.bitnet import (
            BitNetLinear,
            _pack_ternary_weights,
        )

        out_features, in_features = 128, 256

        # Create all-ones weights (sum of weights = in_features)
        weight = torch.ones(out_features, in_features)

        linear = BitNetLinear(in_features, out_features)
        packed, _ = _pack_ternary_weights(weight)
        linear.qweight.data.copy_(packed)
        linear.weight_scale.data.fill_(2.0)  # Non-trivial scale

        # Input with known sum
        x = torch.ones(1, in_features)  # sum = in_features

        # Reference: weight_scale * (W @ x) = 2.0 * (in_features for each output)
        ref_output = 2.0 * in_features

        out = linear(x)

        # Check output is close to reference (allowing for quantization)
        mean_out = out.mean().item()
        assert abs(mean_out - ref_output) / ref_output < 0.1, \
            f"Expected ~{ref_output}, got {mean_out}"

    def test_forward_vs_reference_matmul(self):
        """Test BitNetLinear matches reference matmul with weight_scale."""
        from sglang.srt.models.bitnet import (
            BitNetLinear,
            _pack_ternary_weights,
        )

        out_features, in_features = 256, 512

        # Create random ternary weights
        weight = torch.randint(-1, 2, (out_features, in_features)).float()
        weight_scale = 1.7969  # Realistic HF value

        linear = BitNetLinear(in_features, out_features)
        packed, _ = _pack_ternary_weights(weight)
        linear.qweight.data.copy_(packed)
        linear.weight_scale.data.fill_(weight_scale)

        # Random input
        x = torch.randn(4, in_features)

        # Reference: weight_scale * (W @ x)
        ref_output = weight_scale * torch.matmul(x, weight.t())

        # BitNetLinear output
        bitnet_output = linear(x)

        # Compare with cosine similarity (allows for quantization error)
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_output.flatten().unsqueeze(0),
            bitnet_output.flatten().unsqueeze(0)
        ).item()

        assert cos_sim > 0.99, f"Cosine similarity {cos_sim} too low!"


class TestGibberishDetection:
    """Tests that would have caught the gibberish output bug."""

    def test_output_has_meaningful_variance(self):
        """Gibberish often has wrong variance - test output statistics."""
        from sglang.srt.models.bitnet import (
            BitNetLinear,
            _pack_ternary_weights,
        )

        out_features, in_features = 256, 512

        # Create random ternary weights
        weight = torch.randint(-1, 2, (out_features, in_features)).float()

        linear = BitNetLinear(in_features, out_features)
        packed, _ = _pack_ternary_weights(weight)
        linear.qweight.data.copy_(packed)
        linear.weight_scale.data.fill_(1.5)

        # Random input with known statistics
        x = torch.randn(8, in_features)

        # Reference stats
        ref_out = 1.5 * torch.matmul(x, weight.t())
        ref_std = ref_out.std().item()

        # BitNet output
        out = linear(x)
        out_std = out.std().item()

        # Output std should be similar to reference (within 2x)
        ratio = out_std / ref_std if ref_std > 0 else float('inf')
        assert 0.5 < ratio < 2.0, f"Output std ratio {ratio} suggests incorrect computation"

    def test_output_not_constant(self):
        """Gibberish often produces constant or near-constant output."""
        from sglang.srt.models.bitnet import (
            BitNetLinear,
            _pack_ternary_weights,
        )

        out_features, in_features = 256, 512

        # Create diverse ternary weights
        weight = torch.randint(-1, 2, (out_features, in_features)).float()

        linear = BitNetLinear(in_features, out_features)
        packed, _ = _pack_ternary_weights(weight)
        linear.qweight.data.copy_(packed)
        linear.weight_scale.data.fill_(1.5)

        # Different inputs
        x1 = torch.randn(1, in_features)
        x2 = torch.randn(1, in_features)

        out1 = linear(x1)
        out2 = linear(x2)

        # Outputs should be different for different inputs
        diff = (out1 - out2).abs().mean().item()
        assert diff > 0.1, f"Different inputs produced too similar outputs (diff={diff})"

    def test_gemv_vs_gemm_consistency(self):
        """GEMV (batch=1) and GEMM (batch>1) should be consistent."""
        from sglang.srt.models.bitnet import (
            BitNetLinear,
            _pack_ternary_weights,
        )

        out_features, in_features = 256, 512

        weight = torch.randint(-1, 2, (out_features, in_features)).float()

        linear = BitNetLinear(in_features, out_features)
        packed, _ = _pack_ternary_weights(weight)
        linear.qweight.data.copy_(packed)
        linear.weight_scale.data.fill_(1.7)

        # Same input, different batch sizes
        x = torch.randn(1, in_features)

        # GEMV path (batch=1)
        out_gemv = linear(x)

        # GEMM path (batch>1, take first row)
        x_batched = x.repeat(4, 1)
        out_gemm = linear(x_batched)[0:1]

        # Should be identical (or very close)
        cos_sim = torch.nn.functional.cosine_similarity(
            out_gemv.flatten().unsqueeze(0),
            out_gemm.flatten().unsqueeze(0)
        ).item()

        assert cos_sim > 0.999, f"GEMV vs GEMM mismatch: cos_sim={cos_sim}"


class TestSafetensorsLoading:
    """Test loading from actual HF safetensors format."""

    @pytest.mark.skipif(
        not Path("/home/lev/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T").exists(),
        reason="HF model not cached"
    )
    def test_load_hf_safetensors_weight(self):
        """Load actual HF weights and verify conversion."""
        from safetensors import safe_open
        from sglang.srt.models.bitnet import (
            _unpack_ternary_weights,
            _pack_ternary_weights,
            _unpack_i2_to_ternary,
            BitNetLinear,
        )
        import glob

        # Find safetensors file
        cache_path = Path("/home/lev/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T")
        safetensors_files = list(cache_path.glob("**/model.safetensors"))
        if not safetensors_files:
            pytest.skip("No safetensors file found")

        safetensors_path = str(safetensors_files[0])

        with safe_open(safetensors_path, framework="pt") as f:
            # Load weight and scale
            hf_packed = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
            hf_scale = f.get_tensor("model.layers.0.self_attn.q_proj.weight_scale")

        assert hf_packed.dtype == torch.uint8, f"Expected uint8, got {hf_packed.dtype}"
        assert hf_scale.numel() == 1, "weight_scale should be scalar"

        # Verify shape is HF format [out/4, in]
        packed_rows, in_features = hf_packed.shape
        out_features = packed_rows * 4

        # Convert to kernel format
        unpacked = _unpack_ternary_weights(hf_packed)
        kernel_packed, _ = _pack_ternary_weights(unpacked)

        # Create BitNetLinear and load
        linear = BitNetLinear(in_features, out_features)
        linear.qweight.data.copy_(kernel_packed)
        linear.weight_scale.data.copy_(hf_scale.float())

        # Verify weight_scale is preserved (not 1.0)
        assert linear.weight_scale.item() != 1.0, \
            f"weight_scale should not be 1.0, got {linear.weight_scale.item()}"

        # Test forward pass produces reasonable output
        x = torch.randn(1, in_features)
        out = linear(x)

        # Reference
        ref = hf_scale.item() * torch.matmul(x, unpacked.t())

        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0),
            out.flatten().unsqueeze(0)
        ).item()

        assert cos_sim > 0.99, f"Cosine similarity {cos_sim} too low with real HF weights!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
