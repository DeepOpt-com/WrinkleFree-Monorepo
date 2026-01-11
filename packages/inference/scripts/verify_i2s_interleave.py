#!/usr/bin/env python3
"""Verify I2_S uses interleaved SIMD layout."""

import struct
import mmap
from safetensors import safe_open
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


def decode_i2s_simple(packed_bytes, n_elements):
    """Decode I2_S with simple sequential layout (safetensors style).
    Each byte: [b7:b6]=elem0, [b5:b4]=elem1, [b3:b2]=elem2, [b1:b0]=elem3
    Encoding: 00=-1, 01=0, 10=+1
    """
    output = np.zeros(n_elements, dtype=np.int8)
    for i, byte in enumerate(packed_bytes):
        if i * 4 >= n_elements:
            break
        for j, shift in enumerate([6, 4, 2, 0]):
            idx = i * 4 + j
            if idx >= n_elements:
                break
            val = (byte >> shift) & 0x03
            output[idx] = [-1, 0, 1, 0][val]  # 00=-1, 01=0, 10=+1, 11=0
    return output


def decode_i2s_interleaved(packed_bytes, n_elements, group_size=128):
    """Decode I2_S with interleaved SIMD layout (GGUF style).

    For a group of 32 bytes (128 elements):
    - Byte 0: [elem0, elem32, elem64, elem96] in bits [7:6], [5:4], [3:2], [1:0]
    - Byte 1: [elem1, elem33, elem65, elem97]
    - ...
    - Byte 31: [elem31, elem63, elem95, elem127]
    """
    output = np.zeros(n_elements, dtype=np.int8)
    n_groups = (n_elements + group_size - 1) // group_size

    for g in range(n_groups):
        group_start = g * group_size
        byte_offset = g * 32  # 32 bytes per group

        for byte_idx in range(32):
            if byte_offset + byte_idx >= len(packed_bytes):
                break
            byte = packed_bytes[byte_offset + byte_idx]

            # Each byte contains 4 elements at positions: byte_idx, byte_idx+32, byte_idx+64, byte_idx+96
            for shift_idx, shift in enumerate([6, 4, 2, 0]):
                elem_offset = byte_idx + shift_idx * 32
                elem_idx = group_start + elem_offset
                if elem_idx >= n_elements:
                    break
                val = (byte >> shift) & 0x03
                output[elem_idx] = [-1, 0, 1, 0][val]

    return output


def verify(gguf_path, st_path):
    print("=== VERIFYING I2_S INTERLEAVED LAYOUT ===\n")

    # Load safetensors
    with safe_open(st_path, framework='pt') as sf:
        import torch
        st_packed = sf.get_tensor('model.layers.0.mlp.gate_proj.weight')
        st_packed_bytes = st_packed.to(torch.uint8).numpy().flatten()
        print(f"Safetensors packed bytes: {len(st_packed_bytes)}")

    # Decode safetensors with simple layout
    n_elements = len(st_packed_bytes) * 4
    st_ternary = decode_i2s_simple(st_packed_bytes, n_elements)
    print(f"Safetensors decoded ternary elements: {n_elements}")
    print(f"First 20 values: {st_ternary[:20]}")

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
            gguf_n_elements = gate_tensor['dims'][0] * gate_tensor['dims'][1]
            packed_size = gguf_n_elements // 4
            abs_offset = data_offset + gate_tensor['offset']

            gguf_data = bytes(mm[abs_offset:abs_offset + packed_size])

            print(f"\nGGUF dims: {gate_tensor['dims']}")
            print(f"GGUF packed bytes: {len(gguf_data)}")

            # Decode GGUF with interleaved layout
            gguf_ternary_interleaved = decode_i2s_interleaved(gguf_data, gguf_n_elements)
            print(f"GGUF interleaved decode first 20: {gguf_ternary_interleaved[:20]}")

            # Decode GGUF with simple layout for comparison
            gguf_ternary_simple = decode_i2s_simple(gguf_data, gguf_n_elements)
            print(f"GGUF simple decode first 20: {gguf_ternary_simple[:20]}")

            # GGUF stores as (in_features, out_features)
            # Need to check if we need to transpose
            in_features = gate_tensor['dims'][0]  # 2560
            out_features = gate_tensor['dims'][1]  # 6912

            print(f"\nGGUF shape: ({in_features}, {out_features})")
            print(f"Safetensors packed shape gives: ({len(st_packed_bytes)*4//2560}, 2560) = ({len(st_packed_bytes)*4//2560}, 2560)")

            # Reshape and compare
            # Safetensors: (out_features, in_features) = (6912, 2560)
            st_2d = st_ternary.reshape(out_features, in_features)

            # Try different GGUF interpretations
            print("\n=== COMPARISON ===")

            # 1. Direct interleaved decode, reshape same way
            try:
                gguf_2d_interleaved = gguf_ternary_interleaved.reshape(out_features, in_features)
                matches = np.sum(st_2d == gguf_2d_interleaved)
                print(f"Interleaved decode, same reshape: {matches}/{st_2d.size} ({100*matches/st_2d.size:.1f}%)")
            except:
                print("Reshape failed for interleaved")

            # 2. Direct interleaved, transposed
            try:
                gguf_2d_interleaved_t = gguf_ternary_interleaved.reshape(in_features, out_features).T
                matches = np.sum(st_2d == gguf_2d_interleaved_t)
                print(f"Interleaved decode, transposed: {matches}/{st_2d.size} ({100*matches/st_2d.size:.1f}%)")
            except:
                print("Transpose failed for interleaved")

            # 3. Simple decode, same reshape
            try:
                gguf_2d_simple = gguf_ternary_simple.reshape(out_features, in_features)
                matches = np.sum(st_2d == gguf_2d_simple)
                print(f"Simple decode, same reshape: {matches}/{st_2d.size} ({100*matches/st_2d.size:.1f}%)")
            except:
                print("Reshape failed for simple")

            # 4. Simple decode, transposed
            try:
                gguf_2d_simple_t = gguf_ternary_simple.reshape(in_features, out_features).T
                matches = np.sum(st_2d == gguf_2d_simple_t)
                print(f"Simple decode, transposed: {matches}/{st_2d.size} ({100*matches/st_2d.size:.1f}%)")
            except:
                print("Transpose failed for simple")

            # Check per-row interleave (each row of 2560 elements may be interleaved separately)
            print("\n=== CHECKING PER-ROW INTERLEAVE ===")

            # Try decoding row-by-row with interleave
            # Each row is 2560 elements = 640 bytes
            row_bytes = in_features // 4
            print(f"Bytes per row: {row_bytes}")

            gguf_per_row = np.zeros((out_features, in_features), dtype=np.int8)
            for row in range(out_features):
                row_start = row * row_bytes
                row_data = gguf_data[row_start:row_start + row_bytes]
                row_ternary = decode_i2s_interleaved(row_data, in_features, group_size=128)
                gguf_per_row[row] = row_ternary

            matches = np.sum(st_2d == gguf_per_row)
            print(f"Per-row interleaved: {matches}/{st_2d.size} ({100*matches/st_2d.size:.1f}%)")

            # Also try per-row simple
            gguf_per_row_simple = np.zeros((out_features, in_features), dtype=np.int8)
            for row in range(out_features):
                row_start = row * row_bytes
                row_data = gguf_data[row_start:row_start + row_bytes]
                row_ternary = decode_i2s_simple(row_data, in_features)
                gguf_per_row_simple[row] = row_ternary

            matches = np.sum(st_2d == gguf_per_row_simple)
            print(f"Per-row simple: {matches}/{st_2d.size} ({100*matches/st_2d.size:.1f}%)")


if __name__ == "__main__":
    verify('/tmp/bitnet-gguf/ggml-model-i2_s.gguf', '/tmp/bitnet-st/model.safetensors')
