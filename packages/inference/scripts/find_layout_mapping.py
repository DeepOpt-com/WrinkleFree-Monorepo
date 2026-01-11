#!/usr/bin/env python3
"""Find the correct layout mapping between GGUF I2_S and HuggingFace."""

import torch
import numpy as np
from safetensors import safe_open
import struct
import mmap


def load_gguf_i2s_raw(gguf_path, tensor_name):
    """Load raw I2_S data from GGUF (packed bytes only)."""
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
            scale_bytes = mm[abs_offset + packed_size:abs_offset + packed_size + 4]
            scale = struct.unpack('<f', scale_bytes)[0]
            return {'packed': packed_data, 'dims': info['dims'], 'scale': scale, 'n_elements': n_elements}

        return None


def main():
    gguf_path = "/tmp/bitnet-gguf-official/ggml-model-i2_s.gguf"
    hf_path = "/tmp/bitnet-hf/model.safetensors"

    # Load both raw packed formats
    gguf_data = load_gguf_i2s_raw(gguf_path, "blk.0.ffn_gate.weight")
    gguf_packed = gguf_data['packed']
    gguf_dims = gguf_data['dims']  # [in_features, out_features] = [2560, 6912]

    with safe_open(hf_path, framework="pt") as f:
        hf_packed = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy()

    print(f"GGUF dims: {gguf_dims}")
    print(f"GGUF packed shape: {gguf_packed.shape}")  # (4423680,)
    print(f"HF packed shape: {hf_packed.shape}")  # (1728, 2560)

    in_features = gguf_dims[0]  # 2560
    out_features = gguf_dims[1]  # 6912
    packed_rows = out_features // 4  # 1728

    print(f"\nin_features: {in_features}, out_features: {out_features}, packed_rows: {packed_rows}")

    # HF layout: (packed_rows, in_features) = (1728, 2560)
    # Each byte contains 4 values from rows [i, i+1728, i+3456, i+5184] for column j

    # GGUF I2_S layout (hypothesis): might be stored as (in_features, packed_rows)
    # Let's check...

    print("\n=== Testing different GGUF interpretations ===")

    # Hypothesis 1: GGUF is stored as (in_features, out_features/4) = (2560, 1728) transposed from HF
    try:
        gguf_2d_1 = gguf_packed.reshape(in_features, packed_rows)  # (2560, 1728)
        gguf_2d_1_T = gguf_2d_1.T  # (1728, 2560)
        match = (gguf_2d_1_T == hf_packed).mean()
        print(f"Hypothesis 1: GGUF as (in,out/4).T: {match:.1%} match")
        if match > 0.9:
            print("  *** FOUND IT! ***")
    except Exception as e:
        print(f"Hypothesis 1 failed: {e}")

    # Hypothesis 2: GGUF is stored row-major (out_features/4, in_features) same as HF
    try:
        gguf_2d_2 = gguf_packed.reshape(packed_rows, in_features)  # (1728, 2560)
        match = (gguf_2d_2 == hf_packed).mean()
        print(f"Hypothesis 2: GGUF as (out/4,in): {match:.1%} match")
        if match > 0.9:
            print("  *** FOUND IT! ***")
    except Exception as e:
        print(f"Hypothesis 2 failed: {e}")

    # Hypothesis 3: Different bit order within bytes
    # HF packs bits [0,2,4,6] for values 0-3
    # GGUF might pack bits [6,4,2,0] (reversed)
    print("\nTesting different bit orderings...")

    # Unpack HF
    hf_unpacked = np.zeros((out_features, in_features), dtype=np.int8)
    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        mask = 3 << (2 * i)
        hf_unpacked[start:end] = (hf_packed & mask) >> (2 * i)
    hf_unpacked -= 1

    # Try different unpack strategies for GGUF
    gguf_2d = gguf_packed.reshape(in_features, packed_rows)  # (2560, 1728) as (col, packed_row)

    # Strategy A: Same bit order as HF, but transposed
    gguf_unpacked_A = np.zeros((in_features, out_features), dtype=np.int8)  # (2560, 6912)
    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        mask = 3 << (2 * i)
        gguf_unpacked_A[:, start:end] = ((gguf_2d & mask) >> (2 * i)).astype(np.int8)
    gguf_unpacked_A -= 1
    gguf_unpacked_A_T = gguf_unpacked_A.T  # (6912, 2560)

    match_A = (gguf_unpacked_A_T == hf_unpacked).mean()
    print(f"Strategy A (same bits, col-major packing, transposed): {match_A:.1%} match")

    # Strategy B: Reversed bit order
    gguf_unpacked_B = np.zeros((in_features, out_features), dtype=np.int8)
    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        shift = 2 * (3 - i)  # Reversed: [6,4,2,0] instead of [0,2,4,6]
        mask = 3 << shift
        gguf_unpacked_B[:, start:end] = ((gguf_2d & mask) >> shift).astype(np.int8)
    gguf_unpacked_B -= 1
    gguf_unpacked_B_T = gguf_unpacked_B.T

    match_B = (gguf_unpacked_B_T == hf_unpacked).mean()
    print(f"Strategy B (reversed bits, col-major, transposed): {match_B:.1%} match")

    # Strategy C: Linear unpack then reshape
    gguf_linear = np.zeros(out_features * in_features, dtype=np.int8)
    idx = 0
    for byte in gguf_packed:
        for shift in [6, 4, 2, 0]:
            if idx >= len(gguf_linear):
                break
            val = (byte >> shift) & 0x03
            gguf_linear[idx] = val - 1
            idx += 1

    # Reshape and compare
    gguf_C_1 = gguf_linear.reshape(out_features, in_features)  # (6912, 2560)
    match_C1 = (gguf_C_1 == hf_unpacked).mean()
    print(f"Strategy C1 (linear MSB first, reshape out,in): {match_C1:.1%} match")

    gguf_C_2 = gguf_linear.reshape(in_features, out_features).T  # (2560, 6912).T = (6912, 2560)
    match_C2 = (gguf_C_2 == hf_unpacked).mean()
    print(f"Strategy C2 (linear MSB first, reshape in,out then .T): {match_C2:.1%} match")

    # Strategy D: LSB first
    gguf_linear_lsb = np.zeros(out_features * in_features, dtype=np.int8)
    idx = 0
    for byte in gguf_packed:
        for shift in [0, 2, 4, 6]:
            if idx >= len(gguf_linear_lsb):
                break
            val = (byte >> shift) & 0x03
            gguf_linear_lsb[idx] = val - 1
            idx += 1

    gguf_D_1 = gguf_linear_lsb.reshape(out_features, in_features)
    match_D1 = (gguf_D_1 == hf_unpacked).mean()
    print(f"Strategy D1 (linear LSB first, reshape out,in): {match_D1:.1%} match")

    gguf_D_2 = gguf_linear_lsb.reshape(in_features, out_features).T
    match_D2 = (gguf_D_2 == hf_unpacked).mean()
    print(f"Strategy D2 (linear LSB first, reshape in,out then .T): {match_D2:.1%} match")

    # Best match so far
    best_matches = [
        ("A", match_A), ("B", match_B),
        ("C1", match_C1), ("C2", match_C2),
        ("D1", match_D1), ("D2", match_D2)
    ]
    best = max(best_matches, key=lambda x: x[1])
    print(f"\nBest match: Strategy {best[0]} with {best[1]:.1%}")

    if best[1] < 0.5:
        print("\n*** No good match found! Trying more strategies... ***")

        # Maybe the GGUF stores values differently?
        # Let's check what values appear in GGUF vs HF packed bytes

        print("\nGGUF unique packed byte values (first 1000):", np.unique(gguf_packed[:1000]))
        print("HF unique packed byte values (first 1000 flattened):", np.unique(hf_packed.flatten()[:1000]))

        # The packing encoding might be different
        # GGUF: 00=-1, 01=0, 10=+1, 11=unused
        # HF: Maybe different?

        # Let's decode byte 0 from both and see
        gguf_byte_0 = gguf_packed[0]
        hf_byte_0 = hf_packed[0, 0]

        print(f"\nGGUF byte 0: {gguf_byte_0} = {bin(gguf_byte_0)}")
        print(f"HF byte 0: {hf_byte_0} = {bin(hf_byte_0)}")

        # Decode GGUF byte 0 with different mappings
        print("\nGGUF byte 0 decoded (MSB first, 00=-1,01=0,10=+1):")
        for shift in [6, 4, 2, 0]:
            val = (gguf_byte_0 >> shift) & 0x03
            ternary = val - 1
            print(f"  bits {shift},{shift+1}: {val:02b} -> {ternary}")

        print("\nHF byte 0 decoded (LSB first, 00=-1,01=0,10=+1):")
        for shift in [0, 2, 4, 6]:
            val = (hf_byte_0 >> shift) & 0x03
            ternary = val - 1
            print(f"  bits {shift},{shift+1}: {val:02b} -> {ternary}")


if __name__ == "__main__":
    main()
