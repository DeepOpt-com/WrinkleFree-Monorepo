"""Debug native kernel issues."""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrinklefree_inference.sglang_backend.bitnet_quantization import (
    quantize_to_bitnet, dequantize_bitnet
)


def test_minimal():
    """Test with minimal NxK matrix."""
    print("=" * 60)
    print("Minimal 4x128 Test")
    print("=" * 60)

    N, K = 4, 128
    # Create simple weight matrix with known pattern
    W = torch.zeros(N, K, dtype=torch.float32)
    for i in range(N):
        for j in range(K):
            # Different pattern per row
            W[i, j] = [1, 0, -1, 0][j % 4] * (1 if i % 2 == 0 else -1)

    x = torch.arange(1, K + 1, dtype=torch.float32) / K  # 1/128 to 128/128

    # Reference matmul
    y_ref = torch.matmul(x, W.T)
    print(f"Reference: {y_ref.numpy()}")

    # Quantize
    packed, scale = quantize_to_bitnet(W)
    print(f"\nPacked shape: {packed.shape}")
    print(f"Scale: {scale}")
    print(f"Packed bytes (first 4): {packed[0, :4].numpy()}")

    # Python dequant
    W_dequant = dequantize_bitnet(packed, scale, N, K)
    print(f"\nDequantized W first 8: {W_dequant[0, :8].numpy()}")
    print(f"Original W first 8: {W[0, :8].numpy()}")

    # Python matmul
    y_py = torch.matmul(x, W_dequant.T)
    print(f"Python matmul: {y_py.numpy()}")

    # Test native
    try:
        import bitnet_native
        x_f32 = x.float().contiguous()
        y_native = bitnet_native.gemv(packed, x_f32, scale)
        print(f"Native GEMV: {y_native.numpy()}")

        cos_sim = torch.nn.functional.cosine_similarity(
            y_py.unsqueeze(0), y_native.unsqueeze(0)
        ).item()
        print(f"\nCosine similarity: {cos_sim:.6f}")

    except ImportError as e:
        print(f"Native not available: {e}")
        return


def test_byte_layout():
    """Verify the byte packing layout."""
    print("\n" + "=" * 60)
    print("Byte Packing Layout Test")
    print("=" * 60)

    K = 128  # Must be divisible by block size

    # Create known ternary pattern: all +1s
    W = torch.ones(1, K, dtype=torch.float32)
    packed, scale = quantize_to_bitnet(W)

    # +1 should be encoded as 2 (0b10)
    # 4 weights per byte: 10 10 10 10 = 0xAA = 170
    print(f"All +1s: packed byte = {packed[0, 0].item()} (expected 170 = 0xAA)")

    # All -1s
    W = -torch.ones(1, K, dtype=torch.float32)
    packed, scale = quantize_to_bitnet(W)
    # -1 should be encoded as 0 (0b00)
    # 4 weights per byte: 00 00 00 00 = 0x00
    print(f"All -1s: packed byte = {packed[0, 0].item()} (expected 0 = 0x00)")

    # All 0s
    W = torch.zeros(1, K, dtype=torch.float32)
    packed, scale = quantize_to_bitnet(W)
    # 0 should be encoded as 1 (0b01)
    # 4 weights per byte: 01 01 01 01 = 0x55 = 85
    print(f"All 0s: packed byte = {packed[0, 0].item()} (expected 85 = 0x55)")

    # Pattern: +1, 0, -1, 0 repeated
    W = torch.tensor([[1, 0, -1, 0] * 32], dtype=torch.float32)  # 128 elements
    packed, scale = quantize_to_bitnet(W)
    # +1=10, 0=01, -1=00, 0=01 -> 10 01 00 01 = 0x91 = 145
    print(f"+1,0,-1,0: packed byte = {packed[0, 0].item()} (expected 145 = 0x91)")

    # Verify dequantization
    W_dequant = dequantize_bitnet(packed, scale, 1, K)
    print(f"Dequant first 8: {W_dequant[0, :8].numpy()}")


def test_single_row_native():
    """Test native kernel on single row."""
    print("\n" + "=" * 60)
    print("Single Row Native Test")
    print("=" * 60)

    try:
        import bitnet_native
    except ImportError:
        print("Native not available")
        return

    K = 128  # Block size
    # Create simple pattern
    W = torch.tensor([[1, 0, -1, 0] * 32], dtype=torch.float32)
    x = torch.ones(K, dtype=torch.float32)

    packed, scale = quantize_to_bitnet(W)
    print(f"Packed first 4: {packed[0, :4].numpy()}")
    print(f"Scale: {scale}")

    # Python reference
    W_dequant = dequantize_bitnet(packed, scale, 1, K)
    y_ref = torch.matmul(x, W_dequant.T)
    print(f"Python result: {y_ref.item()}")

    # Native
    y_native = bitnet_native.gemv(packed, x.contiguous(), scale)
    print(f"Native result: {y_native.item()}")


def test_dimension_check():
    """Verify dimension handling in native kernel."""
    print("\n" + "=" * 60)
    print("Dimension Check")
    print("=" * 60)

    try:
        import bitnet_native
    except ImportError:
        print("Native not available")
        return

    # 4x8 weight matrix
    N, K = 4, 128  # K must be divisible by block_size 128
    W = torch.randn(N, K)
    packed, scale = quantize_to_bitnet(W)

    print(f"Weight shape: {W.shape}")
    print(f"Packed shape: {packed.shape}")
    print(f"Expected packed shape: ({N}, {K//4}) = ({N}, {K//4})")

    x = torch.randn(K, dtype=torch.float32).contiguous()
    print(f"Input shape: {x.shape}")

    # Python reference
    W_dequant = dequantize_bitnet(packed, scale, N, K)
    y_ref = torch.matmul(x, W_dequant.T)
    print(f"Python output shape: {y_ref.shape}")
    print(f"Python output: {y_ref.numpy()}")

    # Native
    y_native = bitnet_native.gemv(packed, x, scale)
    print(f"Native output shape: {y_native.shape}")
    print(f"Native output: {y_native.numpy()}")

    cos_sim = torch.nn.functional.cosine_similarity(
        y_ref.unsqueeze(0), y_native.unsqueeze(0)
    ).item()
    print(f"Cosine similarity: {cos_sim:.6f}")


if __name__ == "__main__":
    test_byte_layout()
    test_single_row_native()
    test_dimension_check()
    test_minimal()
