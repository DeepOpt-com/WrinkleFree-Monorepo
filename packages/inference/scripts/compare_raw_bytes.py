#!/usr/bin/env python3
"""Compare raw packed bytes between GGUF I2_S and HuggingFace."""

import numpy as np
from safetensors import safe_open
import struct
import mmap


def load_gguf_raw(gguf_path, tensor_name):
    """Load raw bytes from GGUF I2_S tensor."""
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
            raw_bytes = np.frombuffer(mm[abs_offset:abs_offset + packed_size], dtype=np.uint8).copy()
            return {'bytes': raw_bytes, 'dims': info['dims'], 'n_elements': n_elements}

        return None


def main():
    gguf_path = "/tmp/bitnet-gguf-official/ggml-model-i2_s.gguf"
    hf_path = "/tmp/bitnet-hf/model.safetensors"

    print("=== COMPARING RAW PACKED BYTES ===\n")

    # Load GGUF raw bytes
    gguf_data = load_gguf_raw(gguf_path, "blk.0.ffn_gate.weight")
    gguf_bytes = gguf_data['bytes']
    gguf_dims = gguf_data['dims']

    # Load HF raw bytes
    with safe_open(hf_path, framework="pt") as f:
        hf_packed = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy().astype(np.uint8)

    print(f"GGUF dims: {gguf_dims}")  # [2560, 6912]
    print(f"GGUF bytes shape: {gguf_bytes.shape}")  # (4423680,)
    print(f"HF packed shape: {hf_packed.shape}")  # (1728, 2560)
    print(f"HF bytes count: {hf_packed.size}")  # 4423680

    # Check if byte counts match
    if gguf_bytes.size == hf_packed.size:
        print("\n✓ Byte counts match!")
    else:
        print(f"\n✗ Byte counts differ: {gguf_bytes.size} vs {hf_packed.size}")
        return

    # Compare raw bytes directly
    hf_flat = hf_packed.flatten()
    direct_match = (gguf_bytes == hf_flat).mean()
    print(f"\nDirect byte comparison: {direct_match:.1%} match")

    if direct_match > 0.9:
        print("*** GGUF is direct copy of HF packed bytes! ***")
        return

    # Try different orderings
    print("\n--- Testing byte reorderings ---")

    # Hypothesis 1: GGUF is HF transposed (before flattening)
    # HF is (1728, 2560), GGUF might be (2560, 1728).flatten()
    hf_T = hf_packed.T.flatten()
    match_T = (gguf_bytes == hf_T).mean()
    print(f"HF transposed: {match_T:.1%} match")
    if match_T > 0.9:
        print("*** FOUND: GGUF = HF.T.flatten() ***")

    # Hypothesis 2: GGUF is HF reshaped differently
    # GGUF dims [2560, 6912] -> maybe packed as (6912/4, 2560) vs HF (1728, 2560)?
    # That's the same as (1728, 2560), so no difference

    # Hypothesis 3: Maybe GGUF bytes need bit reversal within each byte?
    def reverse_bits_in_byte(b):
        # Reverse 2-bit pairs: [7:6,5:4,3:2,1:0] -> [1:0,3:2,5:4,7:6]
        return ((b & 0x03) << 6) | ((b & 0x0C) << 2) | ((b & 0x30) >> 2) | ((b & 0xC0) >> 6)

    gguf_bit_rev = np.array([reverse_bits_in_byte(b) for b in gguf_bytes], dtype=np.uint8)
    match_rev = (gguf_bit_rev == hf_flat).mean()
    print(f"GGUF bit-reversed: {match_rev:.1%} match")

    # Hypothesis 4: Maybe need both transpose AND bit reversal
    match_T_rev = (gguf_bit_rev == hf_T).mean()
    print(f"GGUF bit-reversed vs HF.T: {match_T_rev:.1%} match")

    # Let's look at the first few bytes in detail
    print("\n--- First 16 bytes comparison ---")
    print(f"GGUF: {list(gguf_bytes[:16])}")
    print(f"HF:   {list(hf_flat[:16])}")
    print(f"HF.T: {list(hf_T[:16])}")

    # Decode first few elements from both to see values
    print("\n--- Decoded values (first 16) ---")
    def decode_lsb(byte):
        return [(byte >> s) & 0x03 for s in [0, 2, 4, 6]]

    def decode_msb(byte):
        return [(byte >> s) & 0x03 for s in [6, 4, 2, 0]]

    gguf_vals_lsb = []
    hf_vals_lsb = []
    for i in range(4):
        gguf_vals_lsb.extend(decode_lsb(gguf_bytes[i]))
        hf_vals_lsb.extend(decode_lsb(hf_flat[i]))

    print(f"GGUF decoded (LSB): {[v-1 for v in gguf_vals_lsb]}")
    print(f"HF decoded (LSB):   {[v-1 for v in hf_vals_lsb]}")

    # Try MSB decoding
    gguf_vals_msb = []
    hf_vals_msb = []
    for i in range(4):
        gguf_vals_msb.extend(decode_msb(gguf_bytes[i]))
        hf_vals_msb.extend(decode_msb(hf_flat[i]))

    print(f"GGUF decoded (MSB): {[v-1 for v in gguf_vals_msb]}")
    print(f"HF decoded (MSB):   {[v-1 for v in hf_vals_msb]}")

    # Check where the bytes come from in HF
    print("\n--- Locating GGUF byte 0 in HF ---")
    matches = np.where(hf_flat == gguf_bytes[0])[0]
    if len(matches) > 0:
        print(f"GGUF byte 0 ({gguf_bytes[0]}) found at HF indices: {matches[:10]}...")
        # What's the pattern?
        if len(matches) > 1:
            diffs = np.diff(matches[:20])
            print(f"Differences between indices: {diffs}")

    # Try to find the mapping pattern
    print("\n--- Searching for mapping pattern ---")
    # Try a few specific GGUF bytes and see where they appear in HF
    for gguf_idx in [0, 1, 2, 3, 2560, 2561]:
        byte_val = gguf_bytes[gguf_idx]
        # Find exact position where this byte value should come from
        # by checking nearby matches
        matches = np.where(hf_flat == byte_val)[0]
        if len(matches) > 0:
            # Which of these matches is "correct"?
            # We can check by looking at surrounding bytes
            pass  # Too complex for quick check

    # Let's try a different approach: check if GGUF is column-major read of HF
    print("\n--- Testing column-major reading ---")
    # HF is (1728, 2560) in row-major
    # Column-major reading: hf_packed[:, 0], hf_packed[:, 1], ...
    hf_col_major = hf_packed.T.flatten()  # Same as hf_T
    # Already tested above

    # What if GGUF reshapes to (2560, 1728) first?
    # That would mean it's (in_dim, packed_out_dim)
    if gguf_bytes.size == 2560 * 1728:
        gguf_2d = gguf_bytes.reshape(2560, 1728)
        # Then read in different orders
        gguf_row_major = gguf_2d.flatten()  # same as original
        gguf_col_major = gguf_2d.T.flatten()
        match_c = (gguf_col_major == hf_flat).mean()
        print(f"GGUF reshaped (2560,1728).T.flatten vs HF: {match_c:.1%}")


if __name__ == "__main__":
    main()
