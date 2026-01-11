#!/usr/bin/env python3
"""Verify tensor offsets with corrected I2_S size formula."""

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


def tensor_size(dims, dtype):
    """Calculate tensor size with CORRECTED I2_S formula."""
    n = 1
    for d in dims:
        n *= d

    if dtype == 36:  # I2_S: n/4 + 32 (32 extra bytes per tensor)
        return n // 4 + 32
    elif dtype == 0:  # F32
        return n * 4
    elif dtype == 1:  # F16
        return n * 2
    else:
        return 0


def verify_offsets(path):
    print(f"=== VERIFY OFFSETS WITH CORRECTED I2_S SIZE ===")
    print(f"I2_S formula: n_elements / 4 + 32")
    print()

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

        # Sort by offset
        tensors_sorted = sorted(tensors, key=lambda t: t['offset'])

        print(f"=== CHECKING CONTINUITY ===")

        expected_offset = 0
        mismatches = []

        for i, t in enumerate(tensors_sorted):
            size = tensor_size(t['dims'], t['dtype'])
            actual_offset = t['offset']

            if actual_offset != expected_offset:
                gap = actual_offset - expected_offset
                mismatches.append((t['name'], expected_offset, actual_offset, gap, t['dtype']))
                expected_offset = actual_offset

            expected_offset += size
            # Align
            if expected_offset % alignment != 0:
                expected_offset = (expected_offset // alignment + 1) * alignment

        print(f"Total tensors: {len(tensors)}")
        print(f"Mismatches: {len(mismatches)}")

        if mismatches:
            print("\nFirst 20 mismatches (if any):")
            for name, expected, actual, gap, dtype in mismatches[:20]:
                dtype_names = {0: "F32", 1: "F16", 36: "I2_S"}
                print(f"  {name} ({dtype_names.get(dtype, dtype)}): expected={expected}, actual={actual}, gap={gap}")
        else:
            print("\nAll offsets match! ✓")

        # Verify specific tensor data
        print("\n=== VERIFYING blk.1.ffn_sub_norm DATA ===")
        for t in tensors:
            if t['name'] == 'blk.1.ffn_sub_norm.weight':
                abs_offset = data_offset + t['offset']
                print(f"Offset: {t['offset']} (abs: {abs_offset})")

                data = mm[abs_offset:abs_offset + 64]
                print(f"First 64 bytes: {' '.join(f'{b:02x}' for b in data[:32])}")

                print("\nAs F32:")
                for j in range(8):
                    val = struct.unpack_from('<f', data, j * 4)[0]
                    status = "✓" if 0.1 < abs(val) < 10.0 else "✗"
                    print(f"  [{j}] {val:.6f} {status}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: verify_corrected_size.py <path>")
        sys.exit(1)
    verify_offsets(sys.argv[1])
