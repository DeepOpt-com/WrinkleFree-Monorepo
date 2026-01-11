#!/usr/bin/env python3
"""Compare packed bytes directly between GGUF and HF."""

import numpy as np
from safetensors import safe_open
import struct
import mmap


def load_gguf_packed(gguf_path, tensor_name):
    """Load raw packed bytes from GGUF."""
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
            return {'packed': packed_data, 'dims': info['dims'], 'n_elements': n_elements, 'packed_size': packed_size}

        return None


def main():
    gguf_path = "/tmp/bitnet-gguf-official/ggml-model-i2_s.gguf"
    hf_path = "/tmp/bitnet-hf/model.safetensors"

    print("=== DIRECT PACKED BYTE COMPARISON ===\n")

    # Load GGUF
    gguf_data = load_gguf_packed(gguf_path, "blk.0.ffn_gate.weight")
    gguf_packed = gguf_data['packed']

    in_dim = gguf_data['dims'][0]   # 2560
    out_dim = gguf_data['dims'][1]  # 6912

    # Load HF
    with safe_open(hf_path, framework="pt") as f:
        hf_packed = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy().astype(np.uint8)

    print(f"GGUF dims: {gguf_data['dims']}")
    print(f"GGUF packed size: {len(gguf_packed)}")
    print(f"HF packed shape: {hf_packed.shape}")
    print(f"HF packed size: {hf_packed.size}")

    # Direct comparison
    hf_flat = hf_packed.flatten()
    direct_match = (gguf_packed == hf_flat).mean()
    print(f"\nDirect byte match: {direct_match:.2%}")

    # Reshape GGUF packed to same shape as HF and compare
    hf_packed_rows = hf_packed.shape[0]  # 1728
    hf_packed_cols = hf_packed.shape[1]  # 2560

    print(f"\n--- Trying different GGUF packed reshapes ---")

    # Hypothesis: GGUF packs the same way but in different order
    try:
        gguf_as_hf_shape = gguf_packed.reshape(hf_packed_rows, hf_packed_cols)
        match = (gguf_as_hf_shape == hf_packed).mean()
        print(f"GGUF reshaped to ({hf_packed_rows}, {hf_packed_cols}): {match:.2%}")
    except ValueError as e:
        print(f"Reshape failed: {e}")

    # Try transpose
    try:
        gguf_as_hf_T = gguf_packed.reshape(hf_packed_cols, hf_packed_rows).T
        match_T = (gguf_as_hf_T == hf_packed).mean()
        print(f"GGUF reshaped to ({hf_packed_cols}, {hf_packed_rows}).T: {match_T:.2%}")
    except ValueError as e:
        print(f"Reshape T failed: {e}")

    # Try column-major
    try:
        gguf_F = gguf_packed.reshape((hf_packed_rows, hf_packed_cols), order='F')
        match_F = (gguf_F == hf_packed).mean()
        print(f"GGUF reshaped F-order ({hf_packed_rows}, {hf_packed_cols}): {match_F:.2%}")
    except ValueError as e:
        print(f"Reshape F failed: {e}")

    # What if HF packing is different from GGUF packing?
    # HF: each byte at (packed_row, col) contains 4 values from rows at stride 1728
    # GGUF: might pack 4 consecutive elements per byte

    print(f"\n--- Analyzing byte structure ---")
    print(f"First 8 GGUF bytes: {list(gguf_packed[:8])}")
    print(f"First 8 HF bytes: {list(hf_flat[:8])}")

    # Decode a single byte from each and compare
    def decode_byte_lsb(b):
        return [(int(b) >> s) & 0x03 for s in [0, 2, 4, 6]]

    def decode_byte_msb(b):
        return [(int(b) >> s) & 0x03 for s in [6, 4, 2, 0]]

    print(f"\nGGUF byte 0 decoded LSB: {decode_byte_lsb(gguf_packed[0])} -> ternary: {[v-1 for v in decode_byte_lsb(gguf_packed[0])]}")
    print(f"HF byte 0 decoded LSB: {decode_byte_lsb(hf_flat[0])} -> ternary: {[v-1 for v in decode_byte_lsb(hf_flat[0])]}")

    # If HF uses different bit order?
    print(f"\nHF byte 0 decoded MSB: {decode_byte_msb(hf_flat[0])} -> ternary: {[v-1 for v in decode_byte_msb(hf_flat[0])]}")

    # Let's check: does GGUF byte at position X contain the same encoded value as HF byte at position Y?
    # If they're both encoding the same weight matrix, just in different layouts...
    print(f"\n--- Searching for byte correspondence ---")

    # The multiset of DECODED values matches.
    # What about the multiset of PACKED BYTES?
    from collections import Counter
    gguf_byte_dist = Counter(gguf_packed)
    hf_byte_dist = Counter(hf_flat)

    print(f"GGUF unique bytes: {len(gguf_byte_dist)}")
    print(f"HF unique bytes: {len(hf_byte_dist)}")

    # Do the byte distributions match?
    if gguf_byte_dist == hf_byte_dist:
        print("*** BYTE DISTRIBUTIONS MATCH - same packed bytes, different order! ***")
    else:
        print("Byte distributions differ - different packing schemes")
        # Show differences
        print(f"Top 5 GGUF bytes: {gguf_byte_dist.most_common(5)}")
        print(f"Top 5 HF bytes: {hf_byte_dist.most_common(5)}")


if __name__ == "__main__":
    main()
