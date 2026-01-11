#!/usr/bin/env python3
"""Compare multisets of weight values between GGUF and HF to confirm same model."""

import numpy as np
from safetensors import safe_open
import struct
import mmap
from collections import Counter


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
            packed_data = np.frombuffer(mm[abs_offset:abs_offset + packed_size], dtype=np.uint8).copy()
            return {'packed': packed_data, 'dims': info['dims'], 'n_elements': n_elements}

        return None


def decode_gguf_lsb(packed, n_elements):
    """Decode GGUF I2_S with LSB-first."""
    output = np.zeros(n_elements, dtype=np.int8)
    idx = 0
    for byte in packed:
        byte = int(byte)
        for shift in [0, 2, 4, 6]:
            if idx >= n_elements:
                break
            val = (byte >> shift) & 0x03
            output[idx] = int(val) - 1
            idx += 1
    return output


def decode_gguf_msb(packed, n_elements):
    """Decode GGUF I2_S with MSB-first."""
    output = np.zeros(n_elements, dtype=np.int8)
    idx = 0
    for byte in packed:
        byte = int(byte)
        for shift in [6, 4, 2, 0]:
            if idx >= n_elements:
                break
            val = (byte >> shift) & 0x03
            output[idx] = int(val) - 1
            idx += 1
    return output


def unpack_hf(packed_2d):
    """Unpack HuggingFace packed weights."""
    packed_rows = packed_2d.shape[0]
    cols = packed_2d.shape[1]
    out_features = packed_rows * 4

    unpacked = np.zeros((out_features, cols), dtype=np.int8)
    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        mask = 3 << (2 * i)
        unpacked[start:end] = ((packed_2d & mask) >> (2 * i)).astype(np.int8) - 1

    return unpacked


def main():
    gguf_path = "/tmp/bitnet-gguf-official/ggml-model-i2_s.gguf"
    hf_path = "/tmp/bitnet-hf/model.safetensors"

    print("=== MULTISET COMPARISON ===\n")

    # Load GGUF
    gguf_data = load_gguf_i2s(gguf_path, "blk.0.ffn_gate.weight")
    gguf_lsb = decode_gguf_lsb(gguf_data['packed'], gguf_data['n_elements'])
    gguf_msb = decode_gguf_msb(gguf_data['packed'], gguf_data['n_elements'])

    # Load HF
    with safe_open(hf_path, framework="pt") as f:
        hf_packed = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy().astype(np.uint8)
    hf_unpacked = unpack_hf(hf_packed)
    hf_flat = hf_unpacked.flatten()

    print(f"GGUF dims: {gguf_data['dims']}")
    print(f"GGUF n_elements: {len(gguf_lsb)}")
    print(f"HF unpacked shape: {hf_unpacked.shape}")
    print(f"HF n_elements: {len(hf_flat)}")

    # Compare value distributions
    print("\n--- Value distributions ---")
    for name, arr in [("GGUF LSB", gguf_lsb), ("GGUF MSB", gguf_msb), ("HF", hf_flat)]:
        counts = Counter(arr)
        total = len(arr)
        print(f"{name}:")
        for v in [-1, 0, 1]:
            c = counts.get(v, 0)
            print(f"  {v:+d}: {c:>8} ({c/total*100:.2f}%)")

    # Check if multisets are identical
    print("\n--- Multiset comparison ---")
    gguf_lsb_sorted = np.sort(gguf_lsb)
    gguf_msb_sorted = np.sort(gguf_msb)
    hf_sorted = np.sort(hf_flat)

    lsb_match = np.array_equal(gguf_lsb_sorted, hf_sorted)
    msb_match = np.array_equal(gguf_msb_sorted, hf_sorted)

    print(f"GGUF LSB multiset == HF: {lsb_match}")
    print(f"GGUF MSB multiset == HF: {msb_match}")

    if lsb_match or msb_match:
        print("*** SAME WEIGHTS CONFIRMED - just different arrangement! ***")

    # Compare row sums
    print("\n--- Row sum comparison ---")

    in_dim = gguf_data['dims'][0]   # 2560
    out_dim = gguf_data['dims'][1]  # 6912

    # Use the decode that worked best in multiset
    gguf_arr = gguf_lsb

    gguf_2d = gguf_arr.reshape(out_dim, in_dim)
    gguf_row_sums = gguf_2d.sum(axis=1)
    hf_row_sums = hf_unpacked.sum(axis=1)

    gguf_sorted_sums = np.sort(gguf_row_sums)
    hf_sorted_sums = np.sort(hf_row_sums)

    row_sums_match = np.array_equal(gguf_sorted_sums, hf_sorted_sums)
    print(f"Sorted row sums match: {row_sums_match}")
    print(f"First 10 GGUF sorted row sums: {gguf_sorted_sums[:10]}")
    print(f"First 10 HF sorted row sums: {hf_sorted_sums[:10]}")

    # Try hash-based matching for a few rows
    print("\n--- Finding specific row matches (hash-based) ---")
    hf_row_hashes = {tuple(hf_unpacked[i].tolist()): i for i in range(min(100, out_dim))}

    found = 0
    for gguf_idx in range(out_dim):
        row_tuple = tuple(gguf_2d[gguf_idx].tolist())
        if row_tuple in hf_row_hashes:
            hf_idx = hf_row_hashes[row_tuple]
            found += 1
            if found <= 5:
                print(f"  GGUF row {gguf_idx} == HF row {hf_idx}")
            if found >= 20:
                break

    print(f"  Found {found} matching rows in first 100 HF rows")


if __name__ == "__main__":
    main()
