#!/usr/bin/env python3
"""Verify our I2_S weight decoding matches the safetensors original."""

import sys
import struct
import mmap
from safetensors import safe_open
import torch
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


def decode_i2s_msb_first(data, n_elements, encoding):
    """Decode I2_S with MSB-first order."""
    output = []
    for byte in data:
        if len(output) >= n_elements:
            break
        for shift in [6, 4, 2, 0]:  # MSB first
            if len(output) >= n_elements:
                break
            val = (byte >> shift) & 0x03
            output.append(encoding[val])
    return np.array(output, dtype=np.int8)


def decode_i2s_lsb_first(data, n_elements, encoding):
    """Decode I2_S with LSB-first order."""
    output = []
    for byte in data:
        if len(output) >= n_elements:
            break
        for shift in [0, 2, 4, 6]:  # LSB first
            if len(output) >= n_elements:
                break
            val = (byte >> shift) & 0x03
            output.append(encoding[val])
    return np.array(output, dtype=np.int8)


def verify(gguf_path, st_path):
    print("=== VERIFYING I2_S WEIGHT DECODING ===\n")

    # Load safetensors
    with safe_open(st_path, framework='pt') as sf:
        # The safetensors stores PACKED bytes as BF16 values (0-170)
        # Shape (1728, 2560) = 4,423,680 packed bytes
        st_packed = sf.get_tensor('model.layers.0.mlp.gate_proj.weight')
        st_scale = sf.get_tensor('model.layers.0.mlp.gate_proj.weight_scale').float().item()

        print(f"Safetensors gate_proj shape: {st_packed.shape}")
        print(f"Safetensors scale: {st_scale}")

        # The packed bytes are stored as float values
        st_packed_bytes = st_packed.to(torch.uint8).numpy().flatten()
        print(f"Safetensors packed bytes: {len(st_packed_bytes)}")
        print(f"First 16 packed bytes: {[hex(b) for b in st_packed_bytes[:16]]}")

        # Decode safetensors packed bytes to ternary
        # I2_S encoding: 00=-1, 01=0, 10=+1, 11=unused
        st_encoding = {0: -1, 1: 0, 2: 1, 3: 0}
        st_ternary_lsb = decode_i2s_lsb_first(st_packed_bytes, len(st_packed_bytes) * 4, st_encoding)
        st_ternary_msb = decode_i2s_msb_first(st_packed_bytes, len(st_packed_bytes) * 4, st_encoding)

        print(f"Decoded ternary (LSB) first 20: {st_ternary_lsb[:20]}")
        print(f"Decoded ternary (MSB) first 20: {st_ternary_msb[:20]}")

    # Parse GGUF
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

        # Get gate_proj I2_S data
        gate_tensor = tensors.get('blk.0.ffn_gate.weight')
        if gate_tensor and gate_tensor['dtype'] == 36:  # I2_S
            n_elements = gate_tensor['dims'][0] * gate_tensor['dims'][1]
            packed_size = n_elements // 4
            abs_offset = data_offset + gate_tensor['offset']

            gguf_data = bytes(mm[abs_offset:abs_offset + packed_size])

            print(f"\nGGUF gate_proj dims: {gate_tensor['dims']}")
            print(f"GGUF n_elements: {n_elements}")
            print(f"GGUF packed_size: {packed_size}")
            print(f"First 16 packed bytes: {[hex(b) for b in gguf_data[:16]]}")

            # Compare packed bytes directly
            print("\n=== COMPARING PACKED BYTES ===")
            print(f"Safetensors packed bytes count: {len(st_packed_bytes)}")
            print(f"GGUF packed bytes count: {packed_size}")

            # Check if GGUF bytes are transposed/reordered version of safetensors
            # Safetensors shape: (1728, 2560) = (out/4, in)
            # GGUF shape: [2560, 6912] => (in, out) in column-major

            # Try reshaping and transposing
            st_2d = st_packed_bytes.reshape(1728, 2560)  # (out/4, in)
            print(f"\nSafetensors reshaped: {st_2d.shape}")
            print(f"Safetensors [0,0:8]: {[hex(b) for b in st_2d[0, :8]]}")
            print(f"Safetensors [0:8,0]: {[hex(b) for b in st_2d[:8, 0]]}")

            # GGUF is stored as (in_features, out_features/4) in column-major
            gguf_2d = np.frombuffer(gguf_data, dtype=np.uint8).reshape(2560, 1728).T  # Read as (in, out/4), transpose to (out/4, in)
            print(f"\nGGUF reshaped+transposed: {gguf_2d.shape}")
            print(f"GGUF [0,0:8]: {[hex(b) for b in gguf_2d[0, :8]]}")
            print(f"GGUF [0:8,0]: {[hex(b) for b in gguf_2d[:8, 0]]}")

            # Check if they match after transpose
            matches = np.sum(st_2d == gguf_2d)
            total = st_2d.size
            print(f"\nDirect match after transpose: {matches}/{total} ({100*matches/total:.1f}%)")

            # Try other reshape orders
            gguf_2d_alt = np.frombuffer(gguf_data, dtype=np.uint8).reshape(1728, 2560)  # Same as safetensors
            matches_alt = np.sum(st_2d == gguf_2d_alt)
            print(f"Match with same reshape: {matches_alt}/{total} ({100*matches_alt/total:.1f}%)")

            # Check column-by-column
            print("\n=== COLUMN-BY-COLUMN ANALYSIS ===")
            for col_idx in [0, 1, 100, 500, 1000]:
                st_col = st_2d[:, col_idx]
                gguf_col = gguf_2d_alt[:, col_idx]
                col_matches = np.sum(st_col == gguf_col)
                print(f"Column {col_idx}: {col_matches}/1728 ({100*col_matches/1728:.1f}%)")

            # Check row-by-row
            print("\n=== ROW-BY-ROW ANALYSIS ===")
            for row_idx in [0, 1, 100, 500, 1000]:
                st_row = st_2d[row_idx, :]
                gguf_row = gguf_2d_alt[row_idx, :]
                row_matches = np.sum(st_row == gguf_row)
                print(f"Row {row_idx}: {row_matches}/2560 ({100*row_matches/2560:.1f}%)")

            # Try to find the mapping
            print("\n=== FINDING BYTE MAPPING ===")
            # Check if same bytes exist (just reordered)
            st_unique = set(st_packed_bytes.tolist())
            gguf_unique = set(gguf_data)
            common = st_unique & gguf_unique
            print(f"Safetensors unique bytes: {len(st_unique)}")
            print(f"GGUF unique bytes: {len(gguf_unique)}")
            print(f"Common bytes: {len(common)}")

            # Check byte histogram
            from collections import Counter
            st_hist = Counter(st_packed_bytes.tolist())
            gguf_hist = Counter(gguf_data)
            print(f"\nTop 5 safetensors bytes: {st_hist.most_common(5)}")
            print(f"Top 5 GGUF bytes: {gguf_hist.most_common(5)}")


if __name__ == "__main__":
    verify('/tmp/bitnet-gguf/ggml-model-i2_s.gguf', '/tmp/bitnet-st/model.safetensors')
