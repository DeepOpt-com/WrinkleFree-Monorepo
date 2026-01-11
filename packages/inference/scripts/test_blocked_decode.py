#!/usr/bin/env python3
"""Test blocked I2_S decoding to match Microsoft's layout."""

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


def decode_simple_lsb(packed, n_elements):
    """Simple LSB-first decode (what we've been doing)."""
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


def decode_blocked(packed, out_features, in_features):
    """
    Decode GGUF I2_S blocked layout as used by Microsoft's bitnet.cpp.

    Microsoft's layout (from ggml-bitnet-mad.cpp):
    - Access pattern: y + i * 128 * 32 + j * 128 + offset
    - 32 output rows × 128 input columns per block
    - Blocks stored sequentially

    The packed bytes are organized as:
    - For each block of 32 output rows:
      - For each of 32 sub-blocks of 128/4=32 packed bytes:
        - 32 bytes × 4 values = 128 input elements per row
    """
    BLOCK_ROWS = 32
    BLOCK_COLS = 128

    # Calculate block counts
    n_row_blocks = (out_features + BLOCK_ROWS - 1) // BLOCK_ROWS
    n_col_blocks = (in_features + BLOCK_COLS - 1) // BLOCK_COLS

    output = np.zeros((out_features, in_features), dtype=np.int8)

    # Each block contains BLOCK_ROWS * BLOCK_COLS elements = 32 * 128 = 4096 elements
    # Packed as 4096 / 4 = 1024 bytes per block

    bytes_per_block = (BLOCK_ROWS * BLOCK_COLS) // 4

    for row_block in range(n_row_blocks):
        for col_block in range(n_col_blocks):
            block_idx = row_block * n_col_blocks + col_block
            block_offset = block_idx * bytes_per_block

            if block_offset >= len(packed):
                break

            # Decode elements within this block
            for local_row in range(BLOCK_ROWS):
                out_row = row_block * BLOCK_ROWS + local_row
                if out_row >= out_features:
                    continue

                for local_col_group in range(BLOCK_COLS // 4):
                    # Each byte contains 4 consecutive column values
                    byte_idx = block_offset + local_row * (BLOCK_COLS // 4) + local_col_group

                    if byte_idx >= len(packed):
                        break

                    byte = int(packed[byte_idx])
                    for bit_idx, shift in enumerate([0, 2, 4, 6]):
                        out_col = col_block * BLOCK_COLS + local_col_group * 4 + bit_idx
                        if out_col >= in_features:
                            continue
                        val = (byte >> shift) & 0x03
                        output[out_row, out_col] = int(val) - 1

    return output


def decode_blocked_v2(packed, out_features, in_features):
    """
    Alternative interpretation: blocked column-major.

    Access pattern from code: y + i * 128 * 32 + j * 128 + offset
    where i iterates blocks, j iterates within block (0 to 31)

    This suggests:
    - Stride of 4096 (128*32) between major blocks
    - Stride of 128 between sub-iterations
    - Maybe the layout is column-first within blocks?
    """
    BLOCK_ROWS = 32
    BLOCK_COLS = 128

    n_row_blocks = (out_features + BLOCK_ROWS - 1) // BLOCK_ROWS
    n_col_blocks = (in_features + BLOCK_COLS - 1) // BLOCK_COLS

    output = np.zeros((out_features, in_features), dtype=np.int8)

    bytes_per_block = (BLOCK_ROWS * BLOCK_COLS) // 4

    for row_block in range(n_row_blocks):
        for col_block in range(n_col_blocks):
            block_idx = row_block * n_col_blocks + col_block
            block_offset = block_idx * bytes_per_block

            if block_offset >= len(packed):
                break

            # Column-major within block:
            # First 32 bytes = first column of 32 rows
            # Next 32 bytes = second column (actually columns 4-7 since 4 per byte)
            for local_col_group in range(BLOCK_COLS // 4):  # 32 groups of 4 columns
                for local_row in range(BLOCK_ROWS):  # 32 rows
                    byte_idx = block_offset + local_col_group * BLOCK_ROWS + local_row

                    if byte_idx >= len(packed):
                        break

                    byte = int(packed[byte_idx])
                    for bit_idx, shift in enumerate([0, 2, 4, 6]):
                        out_row = row_block * BLOCK_ROWS + local_row
                        out_col = col_block * BLOCK_COLS + local_col_group * 4 + bit_idx
                        if out_row >= out_features or out_col >= in_features:
                            continue
                        val = (byte >> shift) & 0x03
                        output[out_row, out_col] = int(val) - 1

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


def decode_32x32_blocked(packed, out_features, in_features):
    """
    Decode using 32x32 blocking based on weight access pattern: x + i * 32 * 32 + j * 32
    """
    BLOCK_SIZE = 32
    n_row_blocks = (out_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_col_blocks = (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE

    output = np.zeros((out_features, in_features), dtype=np.int8)
    bytes_per_block = (BLOCK_SIZE * BLOCK_SIZE) // 4  # 256 bytes per 32x32 block

    for row_block in range(n_row_blocks):
        for col_block in range(n_col_blocks):
            block_idx = row_block * n_col_blocks + col_block
            block_offset = block_idx * bytes_per_block

            if block_offset >= len(packed):
                break

            for local_row in range(BLOCK_SIZE):
                for local_col_group in range(BLOCK_SIZE // 4):
                    byte_idx = block_offset + local_row * (BLOCK_SIZE // 4) + local_col_group

                    if byte_idx >= len(packed):
                        break

                    byte = int(packed[byte_idx])
                    for bit_idx, shift in enumerate([0, 2, 4, 6]):
                        out_row = row_block * BLOCK_SIZE + local_row
                        out_col = col_block * BLOCK_SIZE + local_col_group * 4 + bit_idx
                        if out_row >= out_features or out_col >= in_features:
                            continue
                        val = (byte >> shift) & 0x03
                        output[out_row, out_col] = int(val) - 1

    return output


def main():
    gguf_path = "/tmp/bitnet-gguf-official/ggml-model-i2_s.gguf"
    hf_path = "/tmp/bitnet-hf/model.safetensors"

    print("=== TESTING BLOCKED I2_S DECODE ===\n")

    # Load GGUF
    gguf_data = load_gguf_i2s(gguf_path, "blk.0.ffn_gate.weight")
    packed = gguf_data['packed']
    in_dim = gguf_data['dims'][0]   # 2560
    out_dim = gguf_data['dims'][1]  # 6912

    # Load HF
    with safe_open(hf_path, framework="pt") as f:
        hf_packed = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy().astype(np.uint8)
    hf_2d = unpack_hf(hf_packed)

    print(f"GGUF dims: [{in_dim}, {out_dim}]")
    print(f"HF unpacked shape: {hf_2d.shape}")
    print(f"HF first row[:10]: {hf_2d[0, :10]}")

    # Test simple decode
    simple = decode_simple_lsb(packed, out_dim * in_dim).reshape(out_dim, in_dim)
    simple_match = (simple == hf_2d).mean()
    print(f"\nSimple LSB decode: {simple_match:.2%} match")
    print(f"  First row[:10]: {simple[0, :10]}")

    # Test blocked 32x32
    blocked_32 = decode_32x32_blocked(packed, out_dim, in_dim)
    blocked_32_match = (blocked_32 == hf_2d).mean()
    print(f"\nBlocked 32x32 decode: {blocked_32_match:.2%} match")
    print(f"  First row[:10]: {blocked_32[0, :10]}")

    # Test blocked decode (row-major within block)
    blocked = decode_blocked(packed, out_dim, in_dim)
    blocked_match = (blocked == hf_2d).mean()
    print(f"\nBlocked decode (32x128 row-major in block): {blocked_match:.2%} match")

    # Test blocked decode v2 (column-major within block)
    blocked_v2 = decode_blocked_v2(packed, out_dim, in_dim)
    blocked_v2_match = (blocked_v2 == hf_2d).mean()
    print(f"\nBlocked decode v2 (32x128 col-major in block): {blocked_v2_match:.2%} match")

    # Check row sums
    print("\n--- Row sum comparison ---")
    for name, W in [("Simple", simple), ("Blocked 32x32", blocked_32)]:
        sorted_sums = np.sort(W.sum(axis=1))
        hf_sorted_sums = np.sort(hf_2d.sum(axis=1))
        match = np.array_equal(sorted_sums, hf_sorted_sums)
        print(f"  {name}: sorted row sums match = {match}")

    # Also try: maybe GGUF is just reordering of packed bytes, not values?
    # Let's compare if the GGUF packed matches HF packed when transposed
    print("\n--- Comparing packed byte layouts ---")
    hf_flat = hf_packed.flatten()
    print(f"  Direct: {(packed[:1000] == hf_flat[:1000]).mean():.2%}")

    # Try transposed
    hf_T = hf_packed.T.flatten()
    print(f"  HF transposed: {(packed[:1000] == hf_T[:1000]).mean():.2%}")


if __name__ == "__main__":
    main()
