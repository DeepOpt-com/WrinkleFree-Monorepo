#!/usr/bin/env python3
"""Check model architecture and metadata from GGUF."""

import sys
import struct
import mmap


def read_string(mm, offset):
    """Read a GGUF string (length-prefixed)."""
    length = struct.unpack_from('<Q', mm, offset)[0]
    s = mm[offset + 8:offset + 8 + length].decode('utf-8')
    return s, 8 + length


def read_value(mm, offset, value_type):
    """Read a GGUF value."""
    type_sizes = {
        0: (1, '<B'),   # UINT8
        1: (1, '<b'),   # INT8
        2: (2, '<H'),   # UINT16
        3: (2, '<h'),   # INT16
        4: (4, '<I'),   # UINT32
        5: (4, '<i'),   # INT32
        6: (4, '<f'),   # FLOAT32
        7: (1, '?'),    # BOOL
        10: (8, '<Q'),  # UINT64
        11: (8, '<q'),  # INT64
        12: (8, '<d'),  # FLOAT64
    }

    if value_type == 8:  # STRING
        return read_string(mm, offset)
    elif value_type == 9:  # ARRAY
        arr_type = struct.unpack_from('<I', mm, offset)[0]
        arr_len = struct.unpack_from('<Q', mm, offset + 4)[0]
        consumed = 4 + 8
        values = []
        if arr_type == 8:  # Array of strings
            for _ in range(arr_len):
                s, c = read_string(mm, offset + consumed)
                values.append(s)
                consumed += c
        elif arr_type in type_sizes:
            elem_size, fmt = type_sizes[arr_type]
            for i in range(arr_len):
                val = struct.unpack_from(fmt, mm, offset + consumed)[0]
                values.append(val)
                consumed += elem_size
        return values, consumed
    elif value_type in type_sizes:
        size, fmt = type_sizes[value_type]
        return struct.unpack_from(fmt, mm, offset)[0], size
    else:
        return None, 0


def check_arch(path):
    """Check model architecture from GGUF."""
    print(f"=== MODEL ARCHITECTURE CHECK ===")
    print(f"File: {path}")
    print()

    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Read header
        version = struct.unpack_from('<I', mm, 4)[0]
        n_tensors = struct.unpack_from('<Q', mm, 8)[0]
        n_kv = struct.unpack_from('<Q', mm, 16)[0]

        print(f"Version: {version}")
        print(f"N tensors: {n_tensors}")
        print(f"N KV pairs: {n_kv}")
        print()

        offset = 24

        # Read and display all KV pairs
        print("=== METADATA ===")
        for i in range(n_kv):
            key, consumed = read_string(mm, offset)
            offset += consumed
            value_type = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            value, consumed = read_value(mm, offset, value_type)
            offset += consumed

            # Display key metadata
            if isinstance(value, list) and len(value) > 10:
                print(f"{key}: [{len(value)} items]")
            elif isinstance(value, str) and len(value) > 100:
                print(f"{key}: '{value[:100]}...'")
            else:
                print(f"{key}: {value}")

        print()

        # Count tensors by type
        tensor_counts = {}
        tensor_offset = offset

        for i in range(n_tensors):
            name, consumed = read_string(mm, tensor_offset)
            tensor_offset += consumed

            n_dims = struct.unpack_from('<I', mm, tensor_offset)[0]
            tensor_offset += 4

            for _ in range(n_dims):
                tensor_offset += 8

            dtype = struct.unpack_from('<I', mm, tensor_offset)[0]
            tensor_offset += 4
            tensor_offset += 8  # offset

            # Categorize tensor
            if "sub_norm" in name:
                tensor_counts["sub_norm"] = tensor_counts.get("sub_norm", 0) + 1
            elif "ffn_norm" in name:
                tensor_counts["ffn_norm"] = tensor_counts.get("ffn_norm", 0) + 1
            elif "attn_norm" in name:
                tensor_counts["attn_norm"] = tensor_counts.get("attn_norm", 0) + 1

        print("=== TENSOR COUNTS ===")
        for k, v in sorted(tensor_counts.items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_model_arch.py <path-to-gguf>")
        sys.exit(1)
    check_arch(sys.argv[1])
