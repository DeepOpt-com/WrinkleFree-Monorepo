#!/usr/bin/env python3
"""Compare weight statistics between GGUF and HuggingFace to verify same model."""

import torch
import numpy as np
from safetensors import safe_open
import struct
import mmap


def load_gguf_i2s(gguf_path, tensor_name):
    """Load I2_S tensor from GGUF."""
    def read_string(mm, offset):
        length = struct.unpack_from('<Q', mm, offset)[0]
        s = mm[offset + 8:offset + 8 + length].decode('utf-8')
        return s, 8 + length

    def skip_value(mm, offset, value_type):
        type_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
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

    def decode_i2s(data, n_elements):
        output = np.zeros(n_elements, dtype=np.int8)
        idx = 0
        for byte in data:
            if idx >= n_elements:
                break
            for shift in [6, 4, 2, 0]:  # MSB first
                if idx >= n_elements:
                    break
                val = (byte >> shift) & 0x03
                output[idx] = val - 1
                idx += 1
        return output

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
        for _ in range(n_tensors):
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
            tensors[name] = {'dims': dims, 'dtype': dtype, 'offset': rel_offset}

        padding = offset % alignment
        if padding != 0:
            offset += alignment - padding
        data_offset = offset

        if tensor_name not in tensors:
            return None

        info = tensors[tensor_name]
        abs_offset = data_offset + info['offset']
        n_elements = 1
        for d in info['dims']:
            n_elements *= d

        if info['dtype'] == 36:
            packed_size = n_elements // 4
            packed_data = bytes(mm[abs_offset:abs_offset + packed_size])
            scale_bytes = mm[abs_offset + packed_size:abs_offset + packed_size + 4]
            scale = struct.unpack('<f', scale_bytes)[0]
            ternary = decode_i2s(packed_data, n_elements)
            return {'data': ternary, 'dims': info['dims'], 'scale': scale}

        return None


def main():
    gguf_path = "/tmp/bitnet-gguf-official/ggml-model-i2_s.gguf"
    hf_path = "/tmp/bitnet-hf/model.safetensors"

    print("=== Comparing gate_proj.weight ===\n")

    # Load GGUF
    gguf_data = load_gguf_i2s(gguf_path, "blk.0.ffn_gate.weight")
    gguf_ternary = gguf_data['data']
    gguf_scale = gguf_data['scale']

    # Load HF
    with safe_open(hf_path, framework="pt") as f:
        W_hf_packed = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy()
        hf_scale = f.get_tensor("model.layers.0.mlp.gate_proj.weight_scale").float().item()

    # Unpack HF
    packed_shape = W_hf_packed.shape
    original_row_dim = packed_shape[0] * 4
    hf_ternary_2d = np.zeros((original_row_dim, packed_shape[1]), dtype=np.int8)
    for i in range(4):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = 3 << (2 * i)
        hf_ternary_2d[start:end] = (W_hf_packed & mask) >> (2 * i)
    hf_ternary = (hf_ternary_2d - 1).flatten()

    print(f"GGUF n_elements: {len(gguf_ternary)}")
    print(f"HF n_elements: {len(hf_ternary)}")

    print(f"\nGGUF scale: {gguf_scale:.6f}")
    print(f"HF scale: {hf_scale:.6f}")

    # Compare statistics (should be identical regardless of layout)
    print("\n--- Ternary value distributions ---")
    for name, arr in [("GGUF", gguf_ternary), ("HF", hf_ternary)]:
        n_minus1 = (arr == -1).sum()
        n_zero = (arr == 0).sum()
        n_plus1 = (arr == 1).sum()
        total = len(arr)
        print(f"{name}: -1: {n_minus1} ({n_minus1/total*100:.2f}%), "
              f"0: {n_zero} ({n_zero/total*100:.2f}%), "
              f"+1: {n_plus1} ({n_plus1/total*100:.2f}%)")

    print("\n--- Per-row statistics (sorted) ---")
    gguf_2d = gguf_ternary.reshape(6912, 2560)  # (out, in)
    hf_2d = hf_ternary.reshape(6912, 2560)

    gguf_row_sums = np.sort(gguf_2d.sum(axis=1))
    hf_row_sums = np.sort(hf_2d.sum(axis=1))

    print(f"GGUF row sums (first 10): {gguf_row_sums[:10]}")
    print(f"HF row sums (first 10): {hf_row_sums[:10]}")
    print(f"\nGGUF row sums (last 10): {gguf_row_sums[-10:]}")
    print(f"HF row sums (last 10): {hf_row_sums[-10:]}")

    # Check if sorted row sums match
    if np.array_equal(gguf_row_sums, hf_row_sums):
        print("\n*** SORTED ROW SUMS MATCH! Same model, different layout. ***")
    else:
        print(f"\nSorted row sums differ! Max diff: {np.abs(gguf_row_sums - hf_row_sums).max()}")

    print("\n--- Per-column statistics (sorted) ---")
    gguf_col_sums = np.sort(gguf_2d.sum(axis=0))
    hf_col_sums = np.sort(hf_2d.sum(axis=0))

    print(f"GGUF col sums (first 10): {gguf_col_sums[:10]}")
    print(f"HF col sums (first 10): {hf_col_sums[:10]}")

    if np.array_equal(gguf_col_sums, hf_col_sums):
        print("\n*** SORTED COL SUMS MATCH! ***")
    else:
        print(f"\nSorted col sums differ! Max diff: {np.abs(gguf_col_sums - hf_col_sums).max()}")

    # Check if one is transposed version of other
    print("\n--- Checking transpose relationship ---")
    gguf_2d_T = gguf_2d.T  # Now (2560, 6912)
    # Flatten and compare sorted values
    gguf_flat_sorted = np.sort(gguf_2d.flatten())
    hf_flat_sorted = np.sort(hf_2d.flatten())
    if np.array_equal(gguf_flat_sorted, hf_flat_sorted):
        print("All individual values match (same set of ternary weights)")
    else:
        print("Values differ!")

    # Try different reshape interpretations
    print("\n--- Trying different GGUF interpretations ---")
    # GGUF dims are [2560, 6912] = [in, out]
    # Try reshaping as (in, out) then transpose
    gguf_in_out = gguf_ternary.reshape(2560, 6912)  # (in, out)
    gguf_out_in = gguf_in_out.T  # (out, in) = (6912, 2560)

    match_direct = (gguf_2d == hf_2d).mean()
    match_transposed = (gguf_out_in == hf_2d).mean()

    print(f"GGUF reshaped as (out,in) vs HF: {match_direct:.1%} match")
    print(f"GGUF reshaped as (in,out).T vs HF: {match_transposed:.1%} match")

    # Check specific positions
    print("\n--- Checking if HF rows correspond to GGUF columns ---")
    # If GGUF is stored as (in, out) and HF as (out, in), then:
    # HF[i, :] should equal GGUF[:, i]
    hf_row_0 = hf_2d[0, :]  # Shape (2560,)
    gguf_col_0 = gguf_in_out[:, 0]  # Shape (2560,)

    if np.array_equal(hf_row_0, gguf_col_0):
        print("HF row 0 == GGUF col 0: MATCH!")
    else:
        match = (hf_row_0 == gguf_col_0).mean()
        print(f"HF row 0 vs GGUF col 0: {match:.1%} match")
        print(f"HF row 0[:10]: {hf_row_0[:10]}")
        print(f"GGUF col 0[:10]: {gguf_col_0[:10]}")


if __name__ == "__main__":
    main()
