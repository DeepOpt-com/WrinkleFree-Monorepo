"""Fast standalone unit tests for BitNet weight handling.

These tests run WITHOUT importing sglang - they copy the core functions
directly to avoid dependency chain issues.

Tests catch the gibberish output bugs:
1. HF stores weights as [out/4, in] but kernel expects [out, in/4]
2. HF stores weight_scale as separate tensors (not 1.0)
3. Activation sum correction must account for weight_scale

Run with: uv run python tests/test_bitnet_core.py
"""

import torch
from typing import Tuple
from pathlib import Path


# === Copy of core functions from bitnet.py (to avoid sglang imports) ===

def _unpack_ternary_weights(packed_weights: torch.Tensor) -> torch.Tensor:
    """Unpack 2-bit packed weights to ternary {-1, 0, +1} values.

    This matches the transformers.integrations.bitnet.unpack_weights function.
    Weight format in safetensors: [packed_out_features, in_features] = [out//4, in]
    Output format: [out_features, in_features]
    Weight encoding: 00=-1, 01=0, 10=+1 (packed value - 1)

    Unpacking is BLOCKED (not interleaved):
    - packed[0:n] bits 0-1 -> unpacked[0:n]
    - packed[0:n] bits 2-3 -> unpacked[n:2n]
    - packed[0:n] bits 4-5 -> unpacked[2n:3n]
    - packed[0:n] bits 6-7 -> unpacked[3n:4n]
    """
    packed_shape = packed_weights.shape
    packed_rows = packed_shape[0]
    out_features = packed_rows * 4

    if len(packed_shape) == 1:
        unpacked_shape = (out_features,)
    else:
        unpacked_shape = (out_features, *packed_shape[1:])

    unpacked = torch.zeros(unpacked_shape, device=packed_weights.device, dtype=torch.uint8)

    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        mask = 3 << (2 * i)
        unpacked[start:end] = (packed_weights & mask) >> (2 * i)

    # Convert to float and apply mapping: 0->-1, 1->0, 2->+1
    return unpacked.float() - 1.0


def _pack_ternary_weights(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack float ternary weights {-1, 0, +1} to uint8 (4 per byte).

    Input: [out_features, in_features] float with values in {-1, 0, +1}
    Output: (packed [out_features, in_features/4] uint8, scale tensor)

    Kernel expects: packed_weights[out_features, in_features/4]
    Encoding: -1 -> 0 (00), 0 -> 1 (01), +1 -> 2 (10)

    Packing is BLOCKED (matching kernel's AVX2 layout):
    For packed byte k (k=0..31):
    - bits 6-7: weight for input k (xq8_0 * yq8_0[k])
    - bits 4-5: weight for input k+32 (xq8_1 * yq8_1[k])
    - bits 2-3: weight for input k+64 (xq8_2 * yq8_2[k])
    - bits 0-1: weight for input k+96 (xq8_3 * yq8_3[k])
    """
    out_features, in_features = weight.shape

    if in_features % 4 != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by 4")

    # Round to nearest ternary value and clamp
    ternary = torch.round(weight.float()).clamp(-1, 1)

    # Encode: -1 -> 0, 0 -> 1, +1 -> 2
    encoded = (ternary + 1).to(torch.uint8)

    # Pack 4 values per byte using BLOCKED layout
    block_size = 32
    packed = torch.zeros(out_features, in_features // 4, dtype=torch.uint8, device=weight.device)

    num_blocks = in_features // (block_size * 4)
    if num_blocks == 0:
        block_size = in_features // 4
        num_blocks = 1

    for b in range(num_blocks):
        base = b * block_size * 4
        for k in range(block_size):
            packed[:, b * block_size + k] = (
                (encoded[:, base + k] << 6) |
                (encoded[:, base + k + block_size] << 4) |
                (encoded[:, base + k + block_size * 2] << 2) |
                (encoded[:, base + k + block_size * 3])
            )

    scale = weight.abs().max().item()
    if scale < 1e-6:
        scale = 1.0

    return packed, torch.tensor([scale], dtype=torch.float32, device=weight.device)


def _unpack_i2_to_ternary(
    packed_weights: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Unpack 2-bit packed weights [out, in//4] to float [out, in].

    Uses BLOCKED layout matching the pack function and kernel.
    """
    packed = packed_weights.to(torch.int32)
    weights = torch.zeros(out_features, in_features, dtype=torch.float32, device=packed_weights.device)

    block_size = 32
    num_blocks = in_features // (block_size * 4)
    if num_blocks == 0:
        block_size = in_features // 4
        num_blocks = 1

    for b in range(num_blocks):
        base = b * block_size * 4
        for k in range(block_size):
            byte_val = packed[:, b * block_size + k]
            weights[:, base + k] = ((byte_val >> 6) & 0x03).float() - 1.0
            weights[:, base + k + block_size] = ((byte_val >> 4) & 0x03).float() - 1.0
            weights[:, base + k + block_size * 2] = ((byte_val >> 2) & 0x03).float() - 1.0
            weights[:, base + k + block_size * 3] = (byte_val & 0x03).float() - 1.0

    return weights


def quantize_activations_i8(activations: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize FP32 activations to INT8."""
    max_val = activations.abs().max().item()
    if max_val < 1e-6:
        max_val = 1.0
    scale = max_val / 127.0
    quantized = (activations / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def bitnet_gemv_reference(
    packed_weights: torch.Tensor,
    activations: torch.Tensor,
    weight_scale: float,
) -> torch.Tensor:
    """Reference implementation of BitNet GEMV (matches kernel behavior)."""
    out_features = packed_weights.shape[0]
    in_features = packed_weights.shape[1] * 4

    # Unpack weights
    weights = _unpack_i2_to_ternary(packed_weights, out_features, in_features)

    # Kernel computes: scale * sum((weight + 1) * activation)
    # which equals: scale * (sum(weight * activation) + sum(activation))
    encoded = weights + 1.0  # {0, 1, 2}
    activations_f = activations.to(torch.float32)
    output = torch.matmul(encoded, activations_f) * weight_scale

    return output


# === Test Classes ===

class TestHFFormatConversion:
    """Test HuggingFace pre-packed weight format conversion."""

    def test_hf_unpack_produces_ternary(self):
        """_unpack_ternary_weights should produce {-1, 0, +1} from HF uint8."""
        # Byte 0b01010101 = 85 means all zeros (01 = 0 after -1 mapping)
        packed = torch.full((64, 256), 85, dtype=torch.uint8)
        unpacked = _unpack_ternary_weights(packed)

        assert unpacked.shape == (256, 256), f"Expected (256, 256), got {unpacked.shape}"
        assert unpacked.dtype == torch.float32
        assert torch.all(unpacked == 0), "All 0b01 bits should unpack to 0"
        print("  PASSED: HF unpack produces ternary")

    def test_hf_unpack_all_values(self):
        """Test unpacking all possible 2-bit values."""
        # Byte 0b00_00_00_00 = 0 -> all -1
        packed_neg1 = torch.zeros((1, 4), dtype=torch.uint8)
        unpacked = _unpack_ternary_weights(packed_neg1)
        assert torch.all(unpacked == -1), f"Expected all -1, got unique: {torch.unique(unpacked)}"

        # Byte 0b01_01_01_01 = 85 -> all 0
        packed_zero = torch.full((1, 4), 85, dtype=torch.uint8)
        unpacked = _unpack_ternary_weights(packed_zero)
        assert torch.all(unpacked == 0), f"Expected all 0, got unique: {torch.unique(unpacked)}"

        # Byte 0b10_10_10_10 = 170 -> all +1
        packed_pos1 = torch.full((1, 4), 170, dtype=torch.uint8)
        unpacked = _unpack_ternary_weights(packed_pos1)
        assert torch.all(unpacked == 1), f"Expected all +1, got unique: {torch.unique(unpacked)}"

        print("  PASSED: HF unpack all values correct")

    def test_hf_to_kernel_format_conversion(self):
        """Test full HF [out/4, in] -> kernel [out, in/4] conversion."""
        out_features, in_features = 256, 512

        # Create valid HF-packed weights (only 00, 01, 10 bits, not 11)
        # Each 2-bit value must be in {0, 1, 2} for {-1, 0, +1}
        # Valid bytes: combinations of 00, 01, 10 but not 11
        valid_bytes = []
        for b0 in range(3):  # bits 0-1
            for b1 in range(3):  # bits 2-3
                for b2 in range(3):  # bits 4-5
                    for b3 in range(3):  # bits 6-7
                        byte_val = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)
                        valid_bytes.append(byte_val)
        valid_bytes = torch.tensor(valid_bytes, dtype=torch.uint8)

        # Sample from valid bytes
        indices = torch.randint(0, len(valid_bytes), (out_features // 4, in_features))
        hf_packed = valid_bytes[indices]

        # Unpack HF format
        unpacked = _unpack_ternary_weights(hf_packed)
        assert unpacked.shape == (out_features, in_features)
        unique = torch.unique(unpacked)
        assert all(v in [-1, 0, 1] for v in unique.tolist()), f"Non-ternary values: {unique}"

        # Repack to kernel format
        kernel_packed, scale = _pack_ternary_weights(unpacked)
        assert kernel_packed.shape == (out_features, in_features // 4)

        # Verify roundtrip
        reunpacked = _unpack_i2_to_ternary(kernel_packed, out_features, in_features)
        assert torch.allclose(reunpacked, unpacked), "HF -> kernel -> unpack roundtrip failed!"

        print("  PASSED: HF to kernel format conversion")


class TestKernelPackRoundtrip:
    """Test our pack/unpack functions for kernel format."""

    def test_pack_unpack_roundtrip(self):
        """Test packing then unpacking gives same values."""
        out_features, in_features = 512, 256
        weight = torch.randint(-1, 2, (out_features, in_features)).float()

        packed, scale = _pack_ternary_weights(weight)
        assert packed.shape == (out_features, in_features // 4)
        assert packed.dtype == torch.uint8

        unpacked = _unpack_i2_to_ternary(packed, out_features, in_features)
        assert unpacked.shape == weight.shape
        assert torch.allclose(unpacked, weight), "Pack/unpack roundtrip failed!"

        print("  PASSED: Pack/unpack roundtrip")

    def test_pack_unpack_various_sizes(self):
        """Test with various dimension sizes."""
        test_cases = [
            (128, 128),
            (256, 512),
            (512, 256),
            (2560, 2560),  # Realistic BitNet size
        ]

        for out_f, in_f in test_cases:
            weight = torch.randint(-1, 2, (out_f, in_f)).float()
            packed, _ = _pack_ternary_weights(weight)
            unpacked = _unpack_i2_to_ternary(packed, out_f, in_f)
            assert torch.allclose(unpacked, weight), f"Failed for size ({out_f}, {in_f})"

        print(f"  PASSED: Pack/unpack with {len(test_cases)} different sizes")


class TestWeightScaleHandling:
    """Test weight_scale handling in forward pass."""

    def test_forward_with_different_scales(self):
        """Test that weight_scale affects output proportionally."""
        out_features, in_features = 128, 256
        weight = torch.randint(-1, 2, (out_features, in_features)).float()
        packed, _ = _pack_ternary_weights(weight)

        x = torch.randn(in_features)
        x_int8, act_scale = quantize_activations_i8(x)

        # Compute with scale=1.0
        out1 = bitnet_gemv_reference(packed, x_int8, 1.0)

        # Compute with scale=2.0
        out2 = bitnet_gemv_reference(packed, x_int8, 2.0)

        # Output should scale proportionally
        ratio = (out2.abs().mean() / out1.abs().mean()).item()
        assert 1.8 < ratio < 2.2, f"Expected ~2x ratio with 2x scale, got {ratio}"

        print("  PASSED: Forward uses weight_scale correctly")


class TestActivationSumCorrection:
    """Test the activation sum correction formula."""

    def test_activation_sum_formula(self):
        """Verify the kernel output formula and correction."""
        out_features, in_features = 64, 128
        weight = torch.randint(-1, 2, (out_features, in_features)).float()
        packed, _ = _pack_ternary_weights(weight)

        x = torch.randn(in_features)
        x_int8, act_scale = quantize_activations_i8(x)

        weight_scale = 1.7  # Non-trivial scale

        # Kernel computes: weight_scale * sum((weight + 1) * activation)
        kernel_out = bitnet_gemv_reference(packed, x_int8, weight_scale)

        # To get correct result: kernel_out - weight_scale * sum(activation)
        act_sum = x_int8.float().sum()
        corrected = (kernel_out - weight_scale * act_sum) * act_scale

        # Reference: weight_scale * sum(weight * activation) * act_scale
        weights = _unpack_i2_to_ternary(packed, out_features, in_features)
        ref = weight_scale * torch.matmul(weights, x.float())

        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0),
            corrected.flatten().unsqueeze(0)
        ).item()

        assert cos_sim > 0.99, f"Cosine similarity {cos_sim} too low!"
        print(f"  PASSED: Activation sum correction formula (cos_sim={cos_sim:.4f})")


class TestGibberishDetection:
    """Tests that would have caught the gibberish output bug."""

    def test_output_has_meaningful_variance(self):
        """Gibberish often has wrong variance."""
        out_features, in_features = 256, 512
        weight = torch.randint(-1, 2, (out_features, in_features)).float()
        packed, _ = _pack_ternary_weights(weight)

        weight_scale = 1.5
        x = torch.randn(8, in_features)

        # Reference stats
        ref_out = weight_scale * torch.matmul(x, weight.t())
        ref_std = ref_out.std().item()

        # Our output (simulated)
        results = []
        for i in range(x.shape[0]):
            xi = x[i]
            x_int8, act_scale = quantize_activations_i8(xi)
            kernel_out = bitnet_gemv_reference(packed, x_int8, weight_scale)
            act_sum = x_int8.float().sum()
            corrected = (kernel_out - weight_scale * act_sum) * act_scale
            results.append(corrected)
        out = torch.stack(results)
        out_std = out.std().item()

        ratio = out_std / ref_std if ref_std > 0 else float('inf')
        assert 0.5 < ratio < 2.0, f"Output std ratio {ratio} suggests incorrect computation"
        print(f"  PASSED: Output has meaningful variance (ratio={ratio:.2f})")

    def test_output_not_constant(self):
        """Different inputs should produce different outputs."""
        out_features, in_features = 256, 512
        weight = torch.randint(-1, 2, (out_features, in_features)).float()
        packed, _ = _pack_ternary_weights(weight)

        x1 = torch.randn(in_features)
        x2 = torch.randn(in_features)

        x1_int8, act_scale1 = quantize_activations_i8(x1)
        x2_int8, act_scale2 = quantize_activations_i8(x2)

        out1 = bitnet_gemv_reference(packed, x1_int8, 1.5)
        out2 = bitnet_gemv_reference(packed, x2_int8, 1.5)

        diff = (out1 - out2).abs().mean().item()
        assert diff > 0.1, f"Different inputs produced too similar outputs (diff={diff})"
        print(f"  PASSED: Different inputs produce different outputs (diff={diff:.2f})")


class TestSafetensorsLoading:
    """Test loading from actual HF safetensors format."""

    def test_load_hf_safetensors_weight(self):
        """Load actual HF weights and verify conversion."""
        cache_path = Path("/home/lev/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T")
        if not cache_path.exists():
            print("  SKIPPED: HF model not cached")
            return

        safetensors_files = list(cache_path.glob("**/model.safetensors"))
        if not safetensors_files:
            print("  SKIPPED: No safetensors file found")
            return

        from safetensors import safe_open

        with safe_open(str(safetensors_files[0]), framework="pt") as f:
            hf_packed = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
            hf_scale = f.get_tensor("model.layers.0.self_attn.q_proj.weight_scale")

        assert hf_packed.dtype == torch.uint8
        assert hf_scale.numel() == 1

        packed_rows, in_features = hf_packed.shape
        out_features = packed_rows * 4

        # Convert
        unpacked = _unpack_ternary_weights(hf_packed)
        kernel_packed, _ = _pack_ternary_weights(unpacked)

        # Test forward
        x = torch.randn(in_features)
        x_int8, act_scale = quantize_activations_i8(x)

        weight_scale = hf_scale.float().item()
        assert weight_scale != 1.0, f"weight_scale should not be 1.0, got {weight_scale}"

        kernel_out = bitnet_gemv_reference(kernel_packed, x_int8, weight_scale)
        act_sum = x_int8.float().sum()
        out = (kernel_out - weight_scale * act_sum) * act_scale

        # Reference
        ref = weight_scale * torch.matmul(x.float(), unpacked.t())

        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0),
            out.flatten().unsqueeze(0)
        ).item()

        assert cos_sim > 0.99, f"Cosine similarity {cos_sim} too low with real HF weights!"
        print(f"  PASSED: Real HF safetensors loading (cos_sim={cos_sim:.4f}, scale={weight_scale:.4f})")


def run_all_tests():
    """Run all tests."""
    print("\n=== TestHFFormatConversion ===")
    t1 = TestHFFormatConversion()
    t1.test_hf_unpack_produces_ternary()
    t1.test_hf_unpack_all_values()
    t1.test_hf_to_kernel_format_conversion()

    print("\n=== TestKernelPackRoundtrip ===")
    t2 = TestKernelPackRoundtrip()
    t2.test_pack_unpack_roundtrip()
    t2.test_pack_unpack_various_sizes()

    print("\n=== TestWeightScaleHandling ===")
    t3 = TestWeightScaleHandling()
    t3.test_forward_with_different_scales()

    print("\n=== TestActivationSumCorrection ===")
    t4 = TestActivationSumCorrection()
    t4.test_activation_sum_formula()

    print("\n=== TestGibberishDetection ===")
    t5 = TestGibberishDetection()
    t5.test_output_has_meaningful_variance()
    t5.test_output_not_constant()

    print("\n=== TestSafetensorsLoading ===")
    t6 = TestSafetensorsLoading()
    t6.test_load_hf_safetensors_weight()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
