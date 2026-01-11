#!/usr/bin/env python3
"""Exhaustively test I2_S layout interpretations with LSB-first decoding."""

import numpy as np
from safetensors import safe_open
import struct
import mmap


def load_gguf_i2s(gguf_path, tensor_name):
    """Load I2_S tensor from GGUF with LSB-first decoding."""
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

        if info['dtype'] == 36:  # I2_S
            packed_size = n_elements // 4
            packed_data = np.frombuffer(mm[abs_offset:abs_offset + packed_size], dtype=np.uint8).copy()
            return {'packed': packed_data, 'dims': info['dims'], 'n_elements': n_elements}

        return None


def decode_lsb_first(packed, n_elements):
    """Decode I2_S with LSB-first bit extraction (shifts 0,2,4,6)."""
    output = np.zeros(n_elements, dtype=np.int8)
    idx = 0
    for byte in packed:
        if idx >= n_elements:
            break
        byte = int(byte)  # Ensure Python int
        for shift in [0, 2, 4, 6]:  # LSB first
            if idx >= n_elements:
                break
            val = (byte >> shift) & 0x03
            output[idx] = int(val) - 1  # 00=-1, 01=0, 10=+1
            idx += 1
    return output


def decode_msb_first(packed, n_elements):
    """Decode I2_S with MSB-first bit extraction (shifts 6,4,2,0)."""
    output = np.zeros(n_elements, dtype=np.int8)
    idx = 0
    for byte in packed:
        if idx >= n_elements:
            break
        byte = int(byte)  # Ensure Python int
        for shift in [6, 4, 2, 0]:  # MSB first
            if idx >= n_elements:
                break
            val = (byte >> shift) & 0x03
            output[idx] = int(val) - 1
            idx += 1
    return output


def unpack_hf(packed_2d):
    """Unpack HuggingFace packed weights to ternary."""
    packed_rows = packed_2d.shape[0]  # 1728
    cols = packed_2d.shape[1]         # 2560
    out_features = packed_rows * 4    # 6912

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

    print("=== EXHAUSTIVE I2_S LAYOUT SEARCH ===\n")

    # Load GGUF
    gguf_data = load_gguf_i2s(gguf_path, "blk.0.ffn_gate.weight")
    packed = gguf_data['packed']
    dims = gguf_data['dims']  # [2560, 6912]
    n_elements = gguf_data['n_elements']

    print(f"GGUF dims: {dims}")
    print(f"n_elements: {n_elements}")

    in_dim = dims[0]   # 2560
    out_dim = dims[1]  # 6912

    # Load HF and unpack
    with safe_open(hf_path, framework="pt") as f:
        hf_packed = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy().astype(np.uint8)

    hf_unpacked = unpack_hf(hf_packed)
    print(f"HF unpacked shape: {hf_unpacked.shape}")  # (6912, 2560)
    print(f"HF row 0 first 10: {hf_unpacked[0, :10]}")

    # Try both bit orderings
    for bit_order_name, decode_fn in [("LSB-first", decode_lsb_first), ("MSB-first", decode_msb_first)]:
        print(f"\n{'='*60}")
        print(f"Testing {bit_order_name} decoding")
        print('='*60)

        gguf_decoded = decode_fn(packed, n_elements)

        # Try all reasonable reshape + transpose combinations
        layouts = [
            ("row-major (out, in)", lambda x: x.reshape(out_dim, in_dim)),
            ("row-major (in, out).T", lambda x: x.reshape(in_dim, out_dim).T),
            ("col-major (out, in)", lambda x: x.reshape(out_dim, in_dim, order='F')),
            ("col-major (in, out).T", lambda x: x.reshape(in_dim, out_dim, order='F').T),
        ]

        best_match = 0
        best_name = None

        for name, transform in layouts:
            try:
                W = transform(gguf_decoded)
                if W.shape != hf_unpacked.shape:
                    print(f"  {name}: shape {W.shape} != HF {hf_unpacked.shape}, skipping")
                    continue

                match = (W == hf_unpacked).mean()
                print(f"  {name}: {match:.2%} match")

                if match > best_match:
                    best_match = match
                    best_name = name

                # If high match, show first row comparison
                if match > 0.9:
                    print(f"    GGUF row 0: {W[0, :10]}")
                    print(f"    HF row 0:   {hf_unpacked[0, :10]}")
                    print(f"    *** FOUND IT! ***")

            except Exception as e:
                print(f"  {name}: error - {e}")

        if best_match > 0.5:
            print(f"\nBest: {best_name} with {best_match:.2%}")
        else:
            print(f"\nNo good match found (best: {best_match:.2%})")

    # Try one more thing: maybe the GGUF packed bytes need to be reshaped first
    print(f"\n{'='*60}")
    print("Testing 2D packed byte interpretations")
    print('='*60)

    packed_rows = out_dim // 4  # 1728
    packed_cols = in_dim        # 2560

    # Maybe GGUF packs the same way as HF?
    for byte_layout_name, byte_reshape in [
        ("(packed_out, in)", lambda p: p.reshape(packed_rows, packed_cols)),
        ("(in, packed_out).T", lambda p: p.reshape(packed_cols, packed_rows).T),
        ("(packed_out, in) F-order", lambda p: p.reshape(packed_rows, packed_cols, order='F')),
    ]:
        try:
            packed_2d = byte_reshape(packed)
            if packed_2d.shape != hf_packed.shape:
                print(f"  {byte_layout_name}: packed shape {packed_2d.shape} != HF {hf_packed.shape}")
                continue

            # Unpack using HF method
            W = unpack_hf(packed_2d)
            match = (W == hf_unpacked).mean()
            print(f"  {byte_layout_name}: {match:.2%} match after unpack")

            if match > 0.9:
                print(f"    *** FOUND IT! GGUF uses same packed format as HF! ***")
        except Exception as e:
            print(f"  {byte_layout_name}: error - {e}")

    # Final attempt: check if specific rows match (permutation)
    print(f"\n{'='*60}")
    print("Checking for row permutation")
    print('='*60)

    gguf_W = decode_lsb_first(packed, n_elements).reshape(out_dim, in_dim)

    # Find which GGUF row matches HF row 0
    hf_row_0 = hf_unpacked[0]
    for i in range(min(100, out_dim)):
        if np.array_equal(gguf_W[i], hf_row_0):
            print(f"HF row 0 found at GGUF row {i}")
            break
    else:
        # Maybe it's in a column?
        gguf_W_T = gguf_W.T
        for i in range(min(100, in_dim)):
            if np.array_equal(gguf_W_T[i], hf_row_0):
                print(f"HF row 0 found at GGUF col {i} (after transpose)")
                break
        else:
            print("HF row 0 not found in first 100 GGUF rows or cols")

            # Check row sums to see if any match
            hf_row0_sum = hf_row_0.sum()
            matching_rows = [i for i in range(out_dim) if gguf_W[i].sum() == hf_row0_sum]
            print(f"Rows with same sum as HF row 0 ({hf_row0_sum}): {len(matching_rows)}")
            if matching_rows:
                print(f"  First few: {matching_rows[:5]}")
                # Check element-wise match for these rows
                for idx in matching_rows[:5]:
                    match = (gguf_W[idx] == hf_row_0).mean()
                    print(f"    Row {idx}: {match:.2%} element match")


if __name__ == "__main__":
    main()
