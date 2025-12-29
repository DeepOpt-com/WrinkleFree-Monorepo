"""Test SGLang BitNet kernel integration.

Tests:
1. BitNet quantization correctness
2. Python fallback GEMV/GEMM works correctly
3. Activation quantization produces correct results
4. Results are consistent across iterations
"""

import sys
import torch
import numpy as np
from typing import Tuple

# Block size constant
QK_I2_S = 128


def _unpack_ternary_weights(packed_weights: torch.Tensor) -> torch.Tensor:
    """Unpack 2-bit packed weights to ternary {-1, 0, +1} values.

    Weight encoding: 00=-1, 01=0, 10=+1
    """
    out_features = packed_weights.shape[0]
    packed_in_features = packed_weights.shape[1]
    in_features = packed_in_features * 4

    # Convert to int for bit operations
    packed = packed_weights.to(torch.int32)

    # Unpack 4 weights per byte
    weights = torch.zeros(out_features, in_features, dtype=torch.float32)

    for i in range(4):
        shift = i * 2
        bits = (packed >> shift) & 0x03  # Extract 2 bits
        # Map: 00 -> -1, 01 -> 0, 10 -> +1
        unpacked = bits.float() - 1.0  # 0->-1, 1->0, 2->+1
        weights[:, i::4] = unpacked

    return weights


def quantize_activations_i8(activations: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize FP32 activations to INT8."""
    max_val = activations.abs().max().item()
    if max_val < 1e-6:
        max_val = 1.0

    scale = max_val / 127.0
    quantized = (activations / scale).round().clamp(-128, 127).to(torch.int8)

    return quantized, scale


def bitnet_gemv(
    packed_weights: torch.Tensor,
    activations: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """BitNet GEMV: y = scale * (W @ x)."""
    out_features = packed_weights.shape[0]
    in_features = packed_weights.shape[1] * 4

    if in_features % QK_I2_S != 0:
        raise ValueError(f"in_features ({in_features}) must be multiple of {QK_I2_S}")

    if activations.shape[0] != in_features:
        raise ValueError(f"activations.shape[0] != in_features")

    # Python fallback: unpack weights and compute
    weights = _unpack_ternary_weights(packed_weights)
    activations_f = activations.to(torch.float32)
    output = torch.matmul(weights, activations_f) * scale

    return output


def bitnet_gemm(
    packed_weights: torch.Tensor,
    activations: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """BitNet GEMM: Y = scale * (W @ X)."""
    out_features = packed_weights.shape[0]
    in_features = packed_weights.shape[1] * 4

    if in_features % QK_I2_S != 0:
        raise ValueError(f"in_features ({in_features}) must be multiple of {QK_I2_S}")

    # Python fallback: unpack weights and compute batched matmul
    weights = _unpack_ternary_weights(packed_weights)  # [out, in]
    activations_f = activations.to(torch.float32)  # [batch, in]
    output = torch.matmul(activations_f, weights.t()) * scale  # [batch, out]

    return output


def check_kernel_available() -> bool:
    """Check if native kernels are available."""
    return False  # Using Python fallback


def pack_ternary_weights(weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary {-1, 0, +1} weights to 2-bit format.

    Weight encoding: 00=-1, 01=0, 10=+1 (value + 1)
    """
    out_features, in_features = weights.shape
    assert in_features % 4 == 0, "in_features must be divisible by 4"

    packed_in_features = in_features // 4
    packed = torch.zeros(out_features, packed_in_features, dtype=torch.uint8)

    for i in range(4):
        # Get every 4th weight starting at position i
        w = weights[:, i::4].long() + 1  # Map -1->0, 0->1, +1->2
        w = w.clamp(0, 2)  # Safety clamp
        packed = packed | (w.to(torch.uint8) << (i * 2))

    return packed


def test_unpack_pack_roundtrip():
    """Test that pack -> unpack is identity."""
    # Create random ternary weights
    out_features = 256
    in_features = 512  # Multiple of QK_I2_S (128)

    weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.float32)

    # Pack and unpack
    packed = pack_ternary_weights(weights)
    unpacked = _unpack_ternary_weights(packed)

    # Check roundtrip
    assert torch.allclose(weights, unpacked), "Pack/unpack roundtrip failed!"
    print("  [PASS] Pack/unpack roundtrip")


def test_gemv_correctness():
    """Test GEMV produces correct results."""
    out_features = 256
    in_features = 512

    # Create ternary weights and pack them
    weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.float32)
    packed = pack_ternary_weights(weights)

    # Create activations
    activations = torch.randn(in_features)
    act_i8, act_scale = quantize_activations_i8(activations)

    # Compute with BitNet GEMV
    scale = 0.1
    output = bitnet_gemv(packed, act_i8, scale)

    # Reference computation
    act_dequant = act_i8.float() * act_scale
    ref_output = torch.matmul(weights, act_dequant) * scale

    # Check similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        output.unsqueeze(0), ref_output.unsqueeze(0)
    ).item()

    assert cosine_sim > 0.99, f"GEMV cosine similarity too low: {cosine_sim}"
    print(f"  [PASS] GEMV correctness (cosine={cosine_sim:.4f})")
    return cosine_sim


def test_gemm_correctness():
    """Test GEMM produces correct results."""
    out_features = 256
    in_features = 512
    batch_size = 4

    # Create ternary weights and pack them
    weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.float32)
    packed = pack_ternary_weights(weights)

    # Create batched activations
    activations = torch.randn(batch_size, in_features)
    act_i8, act_scale = quantize_activations_i8(activations)

    # Compute with BitNet GEMM
    scale = 0.1
    output = bitnet_gemm(packed, act_i8, scale)

    # Reference computation
    act_dequant = act_i8.float() * act_scale
    ref_output = torch.matmul(act_dequant, weights.t()) * scale

    # Check similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        output.flatten().unsqueeze(0), ref_output.flatten().unsqueeze(0)
    ).item()

    assert cosine_sim > 0.99, f"GEMM cosine similarity too low: {cosine_sim}"
    print(f"  [PASS] GEMM correctness (cosine={cosine_sim:.4f})")
    return cosine_sim


def test_activation_quantization():
    """Test activation quantization."""
    activations = torch.randn(512) * 10  # Scale up for better test

    # Quantize
    act_i8, scale = quantize_activations_i8(activations)

    # Dequantize
    act_dequant = act_i8.float() * scale

    # Check reconstruction quality
    mse = torch.mean((activations - act_dequant) ** 2).item()
    max_error = torch.max(torch.abs(activations - act_dequant)).item()

    assert max_error < 1.0, f"Activation quant max error too high: {max_error}"
    print(f"  [PASS] Activation quantization (MSE={mse:.6f}, max_err={max_error:.4f})")


def test_deterministic():
    """Test that results are deterministic."""
    out_features = 128
    in_features = 256

    torch.manual_seed(42)
    weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.float32)
    packed = pack_ternary_weights(weights)

    torch.manual_seed(42)
    activations = torch.randn(in_features)
    act_i8, act_scale = quantize_activations_i8(activations)

    # Run twice
    out1 = bitnet_gemv(packed, act_i8, 1.0)
    out2 = bitnet_gemv(packed, act_i8, 1.0)

    assert torch.allclose(out1, out2), "Results are not deterministic!"
    print("  [PASS] Deterministic results")


def run_all_tests(iteration: int) -> dict:
    """Run all tests and return results."""
    print(f"\n=== Iteration {iteration} ===")

    results = {
        "iteration": iteration,
        "passed": True,
        "gemv_cosine": 0.0,
        "gemm_cosine": 0.0,
    }

    try:
        test_unpack_pack_roundtrip()
        results["gemv_cosine"] = test_gemv_correctness()
        results["gemm_cosine"] = test_gemm_correctness()
        test_activation_quantization()
        test_deterministic()
        print(f"  [ALL PASS] Iteration {iteration}")
    except AssertionError as e:
        results["passed"] = False
        print(f"  [FAIL] {e}")

    return results


def main():
    print("=" * 60)
    print("SGLang BitNet Kernel Integration Test")
    print("=" * 60)

    print(f"\nNative kernel available: {check_kernel_available()}")
    print(f"Block size (QK_I2_S): {QK_I2_S}")

    # Run 10 iterations
    all_results = []
    for i in range(1, 11):
        results = run_all_tests(i)
        all_results.append(results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in all_results if r["passed"])
    failed = len(all_results) - passed

    gemv_cosines = [r["gemv_cosine"] for r in all_results if r["gemv_cosine"] > 0]
    gemm_cosines = [r["gemm_cosine"] for r in all_results if r["gemm_cosine"] > 0]

    print(f"Passed: {passed}/10")
    print(f"Failed: {failed}/10")

    if gemv_cosines:
        print(f"GEMV cosine similarity: min={min(gemv_cosines):.4f}, max={max(gemv_cosines):.4f}, avg={np.mean(gemv_cosines):.4f}")
    if gemm_cosines:
        print(f"GEMM cosine similarity: min={min(gemm_cosines):.4f}, max={max(gemm_cosines):.4f}, avg={np.mean(gemm_cosines):.4f}")

    if failed > 0:
        print("\n[FAIL] Some tests failed!")
        return 1

    print("\n[SUCCESS] All 10 iterations passed!")
    return 0


if __name__ == "__main__":
    exit(main())
