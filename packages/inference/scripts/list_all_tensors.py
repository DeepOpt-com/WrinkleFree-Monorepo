#!/usr/bin/env python3
"""List all tensors with their types and verify integrity."""

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


def i2s_size(dims):
    """Calculate I2_S size (block_size=256, type_size=64)."""
    n = 1
    for d in dims:
        n *= d
    blocks = (n + 255) // 256
    return blocks * 64


def f32_size(dims):
    n = 1
    for d in dims:
        n *= d
    return n * 4


def f16_size(dims):
    n = 1
    for d in dims:
        n *= d
    return n * 2


def list_tensors(path):
    print(f"=== FULL TENSOR LIST ===")

    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        n_tensors = struct.unpack_from('<Q', mm, 8)[0]
        n_kv = struct.unpack_from('<Q', mm, 16)[0]
        offset = 24

        # Skip KV
        alignment = 32
        for _ in range(n_kv):
            _, consumed = read_string(mm, offset)
            offset += consumed
            vtype = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            offset += skip_value(mm, offset, vtype)

        # Parse ALL tensors
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
            tensors.append({
                'name': name,
                'dims': dims,
                'dtype': dtype,
                'offset': rel_offset,
            })

        # Data offset
        padding = offset % alignment
        if padding != 0:
            offset += alignment - padding
        data_offset = offset

        # Count by type
        type_counts = {}
        for t in tensors:
            dtype = t['dtype']
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        dtype_names = {0: "F32", 1: "F16", 36: "I2_S"}
        print(f"\nTensor counts by type:")
        for dtype, count in sorted(type_counts.items()):
            print(f"  {dtype_names.get(dtype, f'type{dtype}')}: {count}")

        # Sort by offset and verify continuity
        tensors_sorted = sorted(tensors, key=lambda t: t['offset'])

        print(f"\n=== VERIFYING TENSOR DATA CONTINUITY ===")

        expected_offset = 0
        mismatches = []

        for i, t in enumerate(tensors_sorted):
            # Calculate expected size
            dtype = t['dtype']
            dims = t['dims']

            if dtype == 36:  # I2_S
                size = i2s_size(dims)
            elif dtype == 0:  # F32
                size = f32_size(dims)
            elif dtype == 1:  # F16
                size = f16_size(dims)
            else:
                size = 0  # Unknown

            actual_offset = t['offset']

            if actual_offset != expected_offset:
                gap = actual_offset - expected_offset
                mismatches.append((t['name'], expected_offset, actual_offset, gap))

                # Update expected to use actual (to track cumulative drift)
                expected_offset = actual_offset

            # Move to next position
            expected_offset += size
            # Align to 32 bytes
            if expected_offset % alignment != 0:
                expected_offset = (expected_offset // alignment + 1) * alignment

        print(f"Total tensors: {len(tensors)}")
        print(f"Mismatches found: {len(mismatches)}")

        if mismatches:
            print("\nFirst 20 mismatches:")
            for i, (name, expected, actual, gap) in enumerate(mismatches[:20]):
                print(f"  {name}: expected={expected}, actual={actual}, gap={gap}")

        # Check specific problematic tensor
        print("\n=== CHECKING blk.1.ffn_sub_norm DATA ===")
        for t in tensors:
            if t['name'] == 'blk.1.ffn_sub_norm.weight':
                abs_offset = data_offset + t['offset']
                print(f"Name: {t['name']}")
                print(f"Dims: {t['dims']}")
                print(f"Type: {dtype_names.get(t['dtype'], t['dtype'])}")
                print(f"Recorded offset: {t['offset']} (abs: {abs_offset})")

                # Read data and verify
                data = mm[abs_offset:abs_offset + 64]
                print(f"First 64 bytes: {' '.join(f'{b:02x}' for b in data[:32])}")

                # Check if it looks like F32 gamma values
                valid_f32_count = 0
                for j in range(16):
                    val = struct.unpack_from('<f', data, j * 4)[0]
                    if 0.1 < abs(val) < 10.0:
                        valid_f32_count += 1

                print(f"Valid F32 values (0.1 < |x| < 10): {valid_f32_count}/16")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: list_all_tensors.py <path>")
        sys.exit(1)
    list_tensors(sys.argv[1])
