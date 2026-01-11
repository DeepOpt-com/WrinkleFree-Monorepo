#!/usr/bin/env python3
"""Debug GGUF tensor offsets.

Reads a GGUF file and prints all tensor offsets for comparison with our Rust reader.
"""

import sys
import struct
import mmap
from pathlib import Path


def read_string(mm, offset):
    """Read a GGUF string (length-prefixed)."""
    length = struct.unpack_from('<Q', mm, offset)[0]
    s = mm[offset + 8:offset + 8 + length].decode('utf-8')
    return s, 8 + length


def skip_value(mm, offset, value_type):
    """Skip a GGUF value and return bytes consumed."""
    type_sizes = {
        0: 1,   # UINT8
        1: 1,   # INT8
        2: 2,   # UINT16
        3: 2,   # INT16
        4: 4,   # UINT32
        5: 4,   # INT32
        6: 4,   # FLOAT32
        7: 1,   # BOOL
        10: 8,  # UINT64
        11: 8,  # INT64
        12: 8,  # FLOAT64
    }

    if value_type == 8:  # STRING
        _, consumed = read_string(mm, offset)
        return consumed
    elif value_type == 9:  # ARRAY
        arr_type = struct.unpack_from('<I', mm, offset)[0]
        arr_len = struct.unpack_from('<Q', mm, offset + 4)[0]
        consumed = 4 + 8
        if arr_type == 8:  # Array of strings
            for _ in range(arr_len):
                _, c = read_string(mm, offset + consumed)
                consumed += c
        else:
            elem_size = type_sizes.get(arr_type, 0)
            consumed += arr_len * elem_size
        return consumed
    else:
        return type_sizes.get(value_type, 0)


def parse_gguf(path):
    """Parse GGUF file and print tensor offsets."""
    print(f"Parsing: {path}")

    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Read header
        magic = mm[0:4]
        if magic != b'GGUF':
            print(f"Invalid magic: {magic}")
            return

        version = struct.unpack_from('<I', mm, 4)[0]
        n_tensors = struct.unpack_from('<Q', mm, 8)[0]
        n_kv = struct.unpack_from('<Q', mm, 16)[0]

        print(f"Version: {version}")
        print(f"N tensors: {n_tensors}")
        print(f"N KV pairs: {n_kv}")

        offset = 24  # After header

        # Skip KV pairs
        alignment = 32  # Default
        for i in range(n_kv):
            key, consumed = read_string(mm, offset)
            offset += consumed
            value_type = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            consumed = skip_value(mm, offset, value_type)
            offset += consumed

            if key == "general.alignment":
                alignment = struct.unpack_from('<I', mm, offset - consumed)[0]

        print(f"Alignment: {alignment}")
        print(f"Position after KV: {offset}")

        # Parse tensor info
        tensors = []
        for i in range(n_tensors):
            name, consumed = read_string(mm, offset)
            offset += consumed

            n_dims = struct.unpack_from('<I', mm, offset)[0]
            offset += 4

            dims = []
            for _ in range(n_dims):
                dim = struct.unpack_from('<Q', mm, offset)[0]
                dims.append(dim)
                offset += 8

            dtype = struct.unpack_from('<I', mm, offset)[0]
            offset += 4

            tensor_offset = struct.unpack_from('<Q', mm, offset)[0]
            offset += 8

            tensors.append({
                'name': name,
                'dims': dims,
                'dtype': dtype,
                'offset': tensor_offset,
            })

        print(f"Position after tensor info: {offset}")

        # Calculate data section start (aligned)
        padding = offset % alignment
        if padding != 0:
            data_offset = offset + (alignment - padding)
        else:
            data_offset = offset

        print(f"Data offset (aligned): {data_offset}")
        print()

        # Print tensors
        print("=== TENSOR OFFSETS ===")
        dtype_names = {
            0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
            8: "Q8_0", 9: "Q8_1", 34: "TQ1_0", 35: "TQ2_0", 36: "I2_S",
        }

        for t in tensors:
            if 'sub_norm' in t['name'] or 'ffn_norm' in t['name'] or t['name'].startswith('blk.0') or t['name'].startswith('blk.1.'):
                dtype_str = dtype_names.get(t['dtype'], f"unk({t['dtype']})")
                abs_offset = data_offset + t['offset']

                # Read first 16 bytes at that offset
                first_bytes = mm[abs_offset:abs_offset + 16]
                hex_str = ' '.join(f'{b:02x}' for b in first_bytes)

                print(f"{t['name']}")
                print(f"  dims: {t['dims']}, dtype: {dtype_str}")
                print(f"  relative: {t['offset']}, absolute: {abs_offset}")
                print(f"  first 16 bytes: {hex_str}")

                # For F32, interpret as floats
                if t['dtype'] == 0:  # F32
                    floats = struct.unpack_from('<4f', mm, abs_offset)
                    print(f"  as F32: {floats}")
                print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: debug_gguf_offsets.py <path-to-gguf>")
        sys.exit(1)

    parse_gguf(sys.argv[1])
