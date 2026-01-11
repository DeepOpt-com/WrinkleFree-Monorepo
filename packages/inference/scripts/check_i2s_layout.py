#!/usr/bin/env python3
"""Check I2_S tensor internal layout (where are the 32 extra bytes?)."""

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
    n = 1
    for d in dims:
        n *= d
    if dtype == 36:  # I2_S
        return n // 4 + 32
    elif dtype == 0:  # F32
        return n * 4
    elif dtype == 1:  # F16
        return n * 2
    return 0


def check_layout(path):
    print(f"=== CHECKING I2_S TENSOR LAYOUT ===")

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

        # Look at blk.0.ffn_down (I2_S) and what comes after
        print("=== EXAMINING blk.0.ffn_down (I2_S) and surrounding tensors ===")

        for t in tensors:
            if t['name'] == 'blk.0.ffn_down.weight':
                abs_offset = data_offset + t['offset']
                n = t['dims'][0] * t['dims'][1]
                packed_size = n // 4  # Pure 2-bit packed data
                total_size = packed_size + 32  # With extra 32 bytes

                print(f"\nblk.0.ffn_down.weight:")
                print(f"  dims: {t['dims']}")
                print(f"  elements: {n}")
                print(f"  packed data size: {packed_size}")
                print(f"  total size (with +32): {total_size}")
                print(f"  starts at absolute offset: {abs_offset}")
                print(f"  ends at: {abs_offset + total_size}")

                # Check FIRST 64 bytes (might be header/scale)
                print(f"\n  FIRST 64 bytes (potential header):")
                first = mm[abs_offset:abs_offset + 64]
                print(f"    {' '.join(f'{b:02x}' for b in first[:32])}")
                print(f"    {' '.join(f'{b:02x}' for b in first[32:])}")

                # As floats (maybe scale factors?)
                print(f"  As F32 (potential scales):")
                for i in range(8):
                    val = struct.unpack_from('<f', first, i * 4)[0]
                    print(f"    [{i}] {val}")

                # Check LAST 64 bytes (if padding is at end)
                print(f"\n  LAST 64 bytes (if header at end):")
                last = mm[abs_offset + total_size - 64:abs_offset + total_size]
                print(f"    {' '.join(f'{b:02x}' for b in last[:32])}")
                print(f"    {' '.join(f'{b:02x}' for b in last[32:])}")

                # As floats
                print(f"  As F32:")
                for i in range(8):
                    val = struct.unpack_from('<f', last, i * 4)[0]
                    print(f"    [{i}] {val}")

                # Check what's right AFTER this tensor (should be ffn_sub_norm)
                print(f"\n  DATA immediately after ffn_down (at {abs_offset + total_size}):")
                after = mm[abs_offset + total_size:abs_offset + total_size + 64]
                print(f"    {' '.join(f'{b:02x}' for b in after[:32])}")
                as_f32 = [struct.unpack_from('<f', after, i * 4)[0] for i in range(8)]
                print(f"  As F32: {as_f32}")

                # Compare with recorded ffn_sub_norm offset
                for t2 in tensors:
                    if t2['name'] == 'blk.0.ffn_sub_norm.weight':
                        print(f"\n  blk.0.ffn_sub_norm recorded offset: {data_offset + t2['offset']}")
                        print(f"  Difference: {(data_offset + t2['offset']) - (abs_offset + total_size)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_i2s_layout.py <path>")
        sys.exit(1)
    check_layout(sys.argv[1])
