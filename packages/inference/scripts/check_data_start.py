#!/usr/bin/env python3
"""Check what's at the start of the data section."""

import sys
import struct
import mmap


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


def check_data_start(path):
    print(f"=== CHECKING DATA SECTION START ===")

    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        n_tensors = struct.unpack_from('<Q', mm, 8)[0]
        n_kv = struct.unpack_from('<Q', mm, 16)[0]
        offset = 24

        # Skip KV pairs
        alignment = 32
        for _ in range(n_kv):
            key, consumed = read_string(mm, offset)
            offset += consumed
            vtype = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            offset += skip_value(mm, offset, vtype)
            if key == "general.alignment":
                alignment = struct.unpack_from('<I', mm, offset - skip_value(mm, offset, 4))[0]

        # Parse ALL tensor info and find smallest offset
        tensors = []
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
            tensors.append((name, dims, dtype, rel_offset))

        # Calculate data offset
        padding = offset % alignment
        if padding != 0:
            offset += alignment - padding
        data_offset = offset

        print(f"Data section starts at: {data_offset}")
        print()

        # Sort tensors by offset to see order in data section
        tensors_by_offset = sorted(tensors, key=lambda t: t[3])

        print("=== FIRST 20 TENSORS BY DATA OFFSET ===")
        for i, (name, dims, dtype, rel_offset) in enumerate(tensors_by_offset[:20]):
            dtype_names = {0: "F32", 1: "F16", 36: "I2_S"}
            dtype_str = dtype_names.get(dtype, f"type{dtype}")
            print(f"  [{i:3d}] offset={rel_offset:12d} {dtype_str:5s} {name}")

        print()
        print("=== LAST 20 TENSORS BY DATA OFFSET ===")
        for i, (name, dims, dtype, rel_offset) in enumerate(tensors_by_offset[-20:]):
            dtype_names = {0: "F32", 1: "F16", 36: "I2_S"}
            dtype_str = dtype_names.get(dtype, f"type{dtype}")
            print(f"  [{len(tensors_by_offset)-20+i:3d}] offset={rel_offset:12d} {dtype_str:5s} {name}")

        print()

        # Check data at offset 0
        print("=== DATA AT RELATIVE OFFSET 0 ===")
        first_bytes = mm[data_offset:data_offset + 64]
        print(f"First 64 bytes: {' '.join(f'{b:02x}' for b in first_bytes[:32])}")
        print(f"                {' '.join(f'{b:02x}' for b in first_bytes[32:])}")

        # Interpret as F32
        print("\nAs F32:")
        for i in range(8):
            val = struct.unpack_from('<f', first_bytes, i * 4)[0]
            print(f"  [{i}] {val}")

        # Find what tensor should be at offset 0
        for name, dims, dtype, rel_offset in tensors_by_offset:
            if rel_offset == 0:
                print(f"\nTensor at offset 0: {name} dims={dims}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_data_start.py <path>")
        sys.exit(1)
    check_data_start(sys.argv[1])
