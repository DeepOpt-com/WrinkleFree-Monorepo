#!/usr/bin/env python3
"""Check if GGUF stores transposed weights."""

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

    print("=== CHECKING TRANSPOSE RELATIONSHIP ===\n")

    # Load GGUF
    gguf_data = load_gguf_i2s(gguf_path, "blk.0.ffn_gate.weight")
    gguf_decoded = decode_lsb(gguf_data['packed'], gguf_data['n_elements'])

    in_dim = gguf_data['dims'][0]   # 2560
    out_dim = gguf_data['dims'][1]  # 6912

    print(f"GGUF dims: {gguf_data['dims']}")

    # Load HF
    with safe_open(hf_path, framework="pt") as f:
        hf_packed = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy().astype(np.uint8)
    hf_2d = unpack_hf(hf_packed)  # (6912, 2560)

    print(f"HF unpacked shape: {hf_2d.shape}")

    # Try different interpretations
    print("\n--- Testing interpretations ---")

    # 1. Row-major (out, in): what we've been doing
    gguf_out_in = gguf_decoded.reshape(out_dim, in_dim)  # (6912, 2560)
    match1 = (gguf_out_in == hf_2d).mean()
    print(f"1. Row-major (out={out_dim}, in={in_dim}): {match1:.2%} match")

    # 2. Row-major (in, out) then transpose: gives (out, in)
    gguf_in_out_T = gguf_decoded.reshape(in_dim, out_dim).T  # (2560, 6912).T = (6912, 2560)
    match2 = (gguf_in_out_T == hf_2d).mean()
    print(f"2. Row-major (in={in_dim}, out={out_dim}).T: {match2:.2%} match")

    # 3. Column-major Fortran order
    gguf_F = gguf_decoded.reshape((out_dim, in_dim), order='F')  # (6912, 2560)
    match3 = (gguf_F == hf_2d).mean()
    print(f"3. Column-major F (out={out_dim}, in={in_dim}): {match3:.2%} match")

    # 4. Column-major then transpose
    gguf_F_T = gguf_decoded.reshape((in_dim, out_dim), order='F').T  # (2560, 6912).T = (6912, 2560)
    match4 = (gguf_F_T == hf_2d).mean()
    print(f"4. Column-major F (in={in_dim}, out={out_dim}).T: {match4:.2%} match")

    # Check if GGUF row 0 matches any HF column
    print("\n--- Checking specific alignments ---")
    gguf_row_0 = gguf_out_in[0]  # Shape (2560,)
    hf_col_0 = hf_2d[:, 0]      # Shape (6912,)

    print(f"GGUF row 0 shape: {gguf_row_0.shape}")
    print(f"HF col 0 shape: {hf_col_0.shape}")

    # GGUF row (length in_dim=2560) can't directly match HF col (length out_dim=6912)
    # But if GGUF is (in, out) before reshape...
    gguf_in_out = gguf_decoded.reshape(in_dim, out_dim)  # (2560, 6912)
    gguf_col_0_as_in_out = gguf_in_out[:, 0]  # Column 0 of (in, out), shape (2560,)

    # Compare GGUF (in,out)[:,0] with HF (out,in)[:,0]
    hf_col_0_short = hf_2d[:2560, 0]  # Just first 2560 elements of HF col 0
    match_col = (gguf_col_0_as_in_out == hf_col_0_short).mean()
    print(f"GGUF (in,out) col 0 vs HF (out,in) col 0[:2560]: {match_col:.2%}")

    # Compare GGUF (in,out)[0,:] with HF (out,in)[0,:]
    gguf_row_0_as_in_out = gguf_in_out[0, :]  # Row 0 of (in, out), shape (6912,)
    hf_row_0 = hf_2d[0, :]  # Row 0 of HF, shape (2560,)
    # Lengths don't match, so compare first 2560 of GGUF row with HF row
    match_row = (gguf_row_0_as_in_out[:2560] == hf_row_0).mean()
    print(f"GGUF (in,out) row 0[:2560] vs HF row 0: {match_row:.2%}")

    # Maybe GGUF stores W^T?
    # If GGUF stores W^T as (in, out), and we decode it as such:
    gguf_WT = gguf_decoded.reshape(in_dim, out_dim)  # This is W^T with shape (2560, 6912)
    # Then W = gguf_WT.T = (6912, 2560)
    gguf_W = gguf_WT.T
    match_WT = (gguf_W == hf_2d).mean()
    print(f"\nGGUF interpreted as W^T.T: {match_WT:.2%} match")

    # Compare row/column sums
    print("\n--- Comparing row/column sums ---")
    gguf_W_row_sums = np.sort(gguf_W.sum(axis=1))
    hf_row_sums = np.sort(hf_2d.sum(axis=1))
    print(f"Row sums match: {np.array_equal(gguf_W_row_sums, hf_row_sums)}")

    gguf_W_col_sums = np.sort(gguf_W.sum(axis=0))
    hf_col_sums = np.sort(hf_2d.sum(axis=0))
    print(f"Col sums match: {np.array_equal(gguf_W_col_sums, hf_col_sums)}")


if __name__ == "__main__":
    main()
