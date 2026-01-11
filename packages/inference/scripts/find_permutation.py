#!/usr/bin/env python3
"""Find the row permutation between GGUF and HF weights."""

import numpy as np
from safetensors import safe_open
import struct
import mmap
from collections import defaultdict


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


def decode_lsb(packed, n_elements):
    """Decode with LSB-first."""
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

    print("=== FINDING ROW PERMUTATION ===\n")

    # Load GGUF
    gguf_data = load_gguf_i2s(gguf_path, "blk.0.ffn_gate.weight")
    gguf_decoded = decode_lsb(gguf_data['packed'], gguf_data['n_elements'])

    in_dim = gguf_data['dims'][0]   # 2560
    out_dim = gguf_data['dims'][1]  # 6912

    gguf_2d = gguf_decoded.reshape(out_dim, in_dim)  # (6912, 2560)

    # Load HF
    with safe_open(hf_path, framework="pt") as f:
        hf_packed = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy().astype(np.uint8)
    hf_2d = unpack_hf(hf_packed)  # (6912, 2560)

    print(f"GGUF shape: {gguf_2d.shape}")
    print(f"HF shape: {hf_2d.shape}")

    # Build hash index for HF rows
    print("\nBuilding hash index for HF rows...")
    hf_hash_to_rows = defaultdict(list)
    for i in range(out_dim):
        row_hash = hash(hf_2d[i].tobytes())
        hf_hash_to_rows[row_hash].append(i)

    # Find mapping from GGUF rows to HF rows
    print("Finding row mapping...")
    gguf_to_hf = {}
    not_found = 0
    multiple_matches = 0

    for gguf_idx in range(out_dim):
        row_hash = hash(gguf_2d[gguf_idx].tobytes())
        matching_hf_rows = hf_hash_to_rows.get(row_hash, [])

        if not matching_hf_rows:
            not_found += 1
            gguf_to_hf[gguf_idx] = None
        else:
            # Verify exact match
            for hf_idx in matching_hf_rows:
                if np.array_equal(gguf_2d[gguf_idx], hf_2d[hf_idx]):
                    gguf_to_hf[gguf_idx] = hf_idx
                    break
            else:
                # Hash collision but no exact match
                not_found += 1
                gguf_to_hf[gguf_idx] = None

            if len(matching_hf_rows) > 1:
                multiple_matches += 1

    print(f"\nMapping results:")
    print(f"  Rows found: {out_dim - not_found} / {out_dim}")
    print(f"  Rows with multiple HF matches: {multiple_matches}")

    # Analyze the permutation pattern
    print("\n--- Analyzing permutation pattern ---")
    first_20 = [(i, gguf_to_hf[i]) for i in range(20)]
    print(f"First 20 GGUF->HF mappings: {first_20}")

    # Check if it's identity
    identity_count = sum(1 for i in range(out_dim) if gguf_to_hf.get(i) == i)
    print(f"\nRows where GGUF idx == HF idx: {identity_count} / {out_dim}")

    if identity_count == out_dim:
        print("*** IDENTITY MAPPING - rows are in same order! ***")
    elif identity_count > 0:
        print("*** PARTIAL MATCH - some rows match, some don't ***")

    # Check if there's a simple pattern
    mappings = [(i, gguf_to_hf[i]) for i in range(out_dim) if gguf_to_hf[i] is not None]

    if len(mappings) > 100:
        # Check for offset pattern
        diffs = [hf_idx - gguf_idx for gguf_idx, hf_idx in mappings[:100]]
        unique_diffs = set(diffs)
        if len(unique_diffs) == 1:
            print(f"*** CONSTANT OFFSET: GGUF[i] = HF[{list(unique_diffs)[0]}] ***")
        elif len(unique_diffs) <= 10:
            print(f"Limited offset patterns: {unique_diffs}")

    # Check for modular pattern (e.g., interleaving)
    print("\n--- Checking for interleaved pattern ---")
    # Maybe HF splits rows into groups and GGUF interleaves?
    # HF row 0 -> GGUF row 0
    # HF row 1728 -> GGUF row 1
    # HF row 3456 -> GGUF row 2
    # HF row 5184 -> GGUF row 3
    # etc.

    # Test: is GGUF row i == HF row (i % group_size) * stride + (i // group_size)?
    for stride in [1728, 2304, 1152]:  # 6912/4, 6912/3, 6912/6
        matches = 0
        for gguf_idx in range(min(100, out_dim)):
            group = gguf_idx // stride if stride <= out_dim else 0
            within = gguf_idx % stride
            expected_hf = within * (out_dim // stride) + group
            if expected_hf < out_dim and gguf_to_hf.get(gguf_idx) == expected_hf:
                matches += 1
        print(f"  Interleave pattern (stride={stride}): {matches}/100 matches")

    # Print some specific mappings to see pattern
    print("\n--- Sample mappings ---")
    samples = [0, 1, 2, 3, 4, 1728, 1729, 3456, 5184]
    for s in samples:
        if s < out_dim:
            print(f"  GGUF row {s} -> HF row {gguf_to_hf.get(s)}")


if __name__ == "__main__":
    main()
