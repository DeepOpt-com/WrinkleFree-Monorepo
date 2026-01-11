#!/usr/bin/env python3
"""Verify that our I2_S decode+repack produces correct GEMM results."""

import struct
import mmap
import numpy as np


def read_string(mm, offset):
    length = struct.unpack_from('<Q', mm, offset)[0]
    s = mm[offset + 8:offset + 8 + length].decode('utf-8')
    return s, 8 + length


def skip_value(mm, offset, value_type):
    type_sizes = {
        0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
    }
    if value_type == 8:
        _, consumed = read_string(mm, offset)
        return consumed
    elif value_type == 9:
        arr_type = struct.unpack_from('<I', mm, offset)[0]
        arr_len = struct.unpack_from('<Q', mm, offset + 4)[0]
        consumed = 12
        if arr_type == 8:
            for _ in range(arr_len):
                _, c = read_string(mm, offset + consumed)
                consumed += c
        else:
            consumed += arr_len * type_sizes.get(arr_type, 0)
        return consumed
    return type_sizes.get(value_type, 0)


def decode_i2s_sequential(data, n_elements):
    """Decode I2_S as SEQUENTIAL layout (how GGUF stores it)."""
    output = np.zeros(n_elements, dtype=np.int8)
    idx = 0
    for byte in data:
        if idx >= n_elements:
            break
        for shift in [6, 4, 2, 0]:  # MSB first
            if idx >= n_elements:
                break
            val = (byte >> shift) & 0x03
            # 00=-1, 01=0, 10=+1
            output[idx] = val - 1
            idx += 1
    return output


def pack_interleaved(ternary, out_features, in_features):
    """Pack ternary weights in our kernel's interleaved format."""
    QK_BLOCK = 128
    BLOCK_BYTES = 32

    blocks_per_row = (in_features + QK_BLOCK - 1) // QK_BLOCK
    bytes_per_row = blocks_per_row * BLOCK_BYTES
    total_bytes = out_features * bytes_per_row

    output = np.zeros(total_bytes, dtype=np.uint8)
    ternary_2d = ternary.reshape(out_features, in_features)

    for row in range(out_features):
        row_weights = ternary_2d[row]
        row_offset = row * bytes_per_row

        for block in range(blocks_per_row):
            block_start = block * QK_BLOCK
            block_offset = row_offset + block * BLOCK_BYTES

            for byte_idx in range(BLOCK_BYTES):
                packed_byte = 0
                for shift, offset in [(6, 0), (4, 32), (2, 64), (0, 96)]:
                    weight_idx = block_start + byte_idx + offset
                    if weight_idx < in_features:
                        w = row_weights[weight_idx]
                        # -1 -> 0, 0 -> 1, +1 -> 2
                        encoded = int(w + 1)
                    else:
                        encoded = 1  # pad with 0
                    packed_byte |= (encoded << shift)
                output[block_offset + byte_idx] = packed_byte

    return output


def vec_dot_interleaved(packed, activations, k):
    """Compute dot product using interleaved weights."""
    QK_BLOCK = 128
    BLOCK_BYTES = 32

    k_packed = k // 4
    assert len(packed) >= k_packed

    num_blocks = k // QK_BLOCK
    total = 0

    for block in range(num_blocks):
        w_offset = block * BLOCK_BYTES
        a_offset = block * QK_BLOCK

        for byte_idx in range(BLOCK_BYTES):
            packed_byte = int(packed[w_offset + byte_idx])

            # Extract 4 weights
            w0 = ((packed_byte >> 6) & 0x03) - 1  # bits 6-7 -> activation[j]
            w1 = ((packed_byte >> 4) & 0x03) - 1  # bits 4-5 -> activation[j+32]
            w2 = ((packed_byte >> 2) & 0x03) - 1  # bits 2-3 -> activation[j+64]
            w3 = (packed_byte & 0x03) - 1         # bits 0-1 -> activation[j+96]

            # Compute contributions
            total += int(w0) * int(activations[a_offset + byte_idx])
            total += int(w1) * int(activations[a_offset + byte_idx + 32])
            total += int(w2) * int(activations[a_offset + byte_idx + 64])
            total += int(w3) * int(activations[a_offset + byte_idx + 96])

    return total


def reference_gemv(ternary_weights, activations, out_features, in_features):
    """Reference GEMV implementation (no quantization)."""
    W = ternary_weights.reshape(out_features, in_features).astype(np.float32)
    x = activations.astype(np.float32)
    return W @ x


def main():
    gguf_path = '/tmp/bitnet-gguf/ggml-model-i2_s.gguf'

    print("=== VERIFYING I2_S DECODE + REPACK + GEMM ===\n")

    with open(gguf_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        n_tensors = struct.unpack_from('<Q', mm, 8)[0]
        n_kv = struct.unpack_from('<Q', mm, 16)[0]
        offset = 24

        alignment = 32
        for _ in range(n_kv):
            _, consumed = read_string(mm, offset)
            offset += consumed
            vtype = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            offset += skip_value(mm, offset, vtype)

        tensors = {}
        for i in range(n_tensors):
            name, consumed = read_string(mm, offset)
            offset += consumed
            n_dims = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack_from('<Q', mm, offset)[0])
                offset += 8
            dtype = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            rel_offset = struct.unpack_from('<Q', mm, offset)[0]
            offset += 8
            tensors[name] = {
                'dims': dims,
                'dtype': dtype,
                'offset': rel_offset,
            }

        padding = offset % alignment
        if padding != 0:
            offset += alignment - padding
        data_offset = offset

        # Test with Q projection (smaller than gate)
        tensor_name = 'blk.0.attn_q.weight'
        tensor_info = tensors.get(tensor_name)

        if tensor_info and tensor_info['dtype'] == 36:  # I2_S
            in_features = tensor_info['dims'][0]
            out_features = tensor_info['dims'][1]
            n_elements = in_features * out_features
            packed_size = n_elements // 4

            print(f"Testing {tensor_name}")
            print(f"  dims: [{in_features}, {out_features}]")
            print(f"  n_elements: {n_elements}")
            print(f"  packed_size: {packed_size}")

            abs_offset = data_offset + tensor_info['offset']
            gguf_data = bytes(mm[abs_offset:abs_offset + packed_size])

            # Extract scale from extra 32 bytes
            scale_bytes = mm[abs_offset + packed_size:abs_offset + packed_size + 4]
            weight_scale = struct.unpack('<f', scale_bytes)[0]
            print(f"  weight_scale: {weight_scale}")

            # Decode GGUF (sequential) to ternary
            ternary = decode_i2s_sequential(gguf_data, n_elements)
            print(f"  Ternary distribution: -1:{np.sum(ternary==-1)}, 0:{np.sum(ternary==0)}, +1:{np.sum(ternary==1)}")

            # Repack to interleaved format
            packed = pack_interleaved(ternary, out_features, in_features)
            print(f"  Packed size: {len(packed)}")

            # Create test activation (INT8, all 1s for simplicity)
            activations = np.ones(in_features, dtype=np.int8)

            # Compute reference result (sum of all weights in each row)
            ternary_2d = ternary.reshape(out_features, in_features)
            reference = ternary_2d.sum(axis=1).astype(np.float32) * weight_scale

            print(f"\n  Reference output (first 8 rows): {reference[:8]}")

            # Compute using our interleaved kernel
            packed_per_row = in_features // 4
            kernel_output = []
            for row in range(out_features):
                row_packed = packed[row * packed_per_row:(row + 1) * packed_per_row]
                dot = vec_dot_interleaved(row_packed, activations, in_features)
                kernel_output.append(dot * weight_scale)
            kernel_output = np.array(kernel_output)

            print(f"  Kernel output (first 8 rows):    {kernel_output[:8]}")

            # Check if they match
            diff = np.abs(reference - kernel_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"\n  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")

            if max_diff < 0.01:
                print("  ✓ PASS - decode+repack+kernel produces correct results!")
            else:
                print("  ✗ FAIL - results don't match!")

                # Debug: find first mismatch
                mismatches = np.where(diff > 0.01)[0]
                if len(mismatches) > 0:
                    idx = mismatches[0]
                    print(f"\n  First mismatch at row {idx}:")
                    print(f"    Reference: {reference[idx]}")
                    print(f"    Kernel: {kernel_output[idx]}")


def test_simple():
    """Simple test with known values."""
    print("=== SIMPLE TEST ===")

    # 128 weights for one block
    ternary = np.array([1] * 128, dtype=np.int8)  # All +1
    activations = np.array([1] * 128, dtype=np.int8)  # All 1

    # Expected: sum of all weights = 128 (since all are +1 and all activations are 1)
    expected = 128

    # Pack to interleaved (single row, single block)
    packed = pack_interleaved(ternary.reshape(1, 128), 1, 128)
    print(f"Packed bytes: {[hex(b) for b in packed[:8]]}")

    # All +1 encodes as 2, so byte should be 10_10_10_10 = 0xAA
    print(f"Expected all 0xAA: {all(b == 0xAA for b in packed)}")

    # Compute dot product
    result = vec_dot_interleaved(packed, activations, 128)
    print(f"Expected: {expected}, Got: {result}")
    print(f"PASS: {result == expected}")

    # Now test with mixed weights
    print("\n--- Mixed weights test ---")
    # Weights: first 64 are +1, next 64 are -1
    ternary2 = np.array([1] * 64 + [-1] * 64, dtype=np.int8)
    expected2 = 64 - 64  # Should be 0

    packed2 = pack_interleaved(ternary2.reshape(1, 128), 1, 128)
    result2 = vec_dot_interleaved(packed2, activations, 128)
    print(f"Expected: {expected2}, Got: {result2}")
    print(f"PASS: {result2 == expected2}")


if __name__ == "__main__":
    test_simple()
    print()
    main()
