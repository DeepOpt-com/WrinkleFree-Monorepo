#!/usr/bin/env python3
"""Check layer 1 specifically to understand the corruption."""

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


def check_layer1(path):
    print(f"=== CHECKING LAYER 1 BOUNDARY ===")

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

        # Find layer 1 tensors
        layer1_tensors = [t for t in tensors if t['name'].startswith('blk.1.')]

        # Sort by offset
        layer1_sorted = sorted(layer1_tensors, key=lambda t: t['offset'])

        print(f"\n=== LAYER 1 TENSORS (by offset) ===")
        for t in layer1_sorted:
            dtype_names = {0: "F32", 1: "F16", 36: "I2_S"}
            dtype_str = dtype_names.get(t['dtype'], f"type{t['dtype']}")
            size = tensor_size(t['dims'], t['dtype'])
            print(f"  {t['name']}: offset={t['offset']}, size={size}, type={dtype_str}")

        # Check blk.1.ffn_down and blk.1.ffn_sub_norm boundary
        print("\n=== EXAMINING blk.1.ffn_down → blk.1.ffn_sub_norm BOUNDARY ===")

        ffn_down = next(t for t in tensors if t['name'] == 'blk.1.ffn_down.weight')
        ffn_sub_norm = next(t for t in tensors if t['name'] == 'blk.1.ffn_sub_norm.weight')

        ffn_down_abs = data_offset + ffn_down['offset']
        ffn_down_size = tensor_size(ffn_down['dims'], ffn_down['dtype'])
        ffn_down_end = ffn_down_abs + ffn_down_size

        ffn_sub_norm_abs = data_offset + ffn_sub_norm['offset']

        print(f"\nblk.1.ffn_down:")
        print(f"  starts at: {ffn_down_abs}")
        print(f"  size: {ffn_down_size} (n/4 + 32 = {ffn_down['dims'][0] * ffn_down['dims'][1] // 4} + 32)")
        print(f"  ends at: {ffn_down_end}")

        print(f"\nblk.1.ffn_sub_norm:")
        print(f"  starts at: {ffn_sub_norm_abs}")
        print(f"  gap from ffn_down end: {ffn_sub_norm_abs - ffn_down_end}")

        # Check data at various points around the boundary
        print(f"\n=== DATA AROUND BOUNDARY ===")

        print(f"\nLast 64 bytes of ffn_down (offset {ffn_down_end - 64}):")
        last = mm[ffn_down_end - 64:ffn_down_end]
        print(f"  {' '.join(f'{b:02x}' for b in last[:32])}")
        print(f"  {' '.join(f'{b:02x}' for b in last[32:])}")

        print(f"\nFirst 64 bytes of ffn_sub_norm (offset {ffn_sub_norm_abs}):")
        first = mm[ffn_sub_norm_abs:ffn_sub_norm_abs + 64]
        print(f"  {' '.join(f'{b:02x}' for b in first[:32])}")
        print(f"  {' '.join(f'{b:02x}' for b in first[32:])}")
        print(f"  As F32: {[struct.unpack_from('<f', first, i*4)[0] for i in range(8)]}")

        # Check what's at ffn_down_end (should be ffn_sub_norm start if gap is 0)
        print(f"\nData at ffn_down_end ({ffn_down_end}):")
        at_end = mm[ffn_down_end:ffn_down_end + 64]
        print(f"  {' '.join(f'{b:02x}' for b in at_end[:32])}")
        print(f"  As F32: {[struct.unpack_from('<f', at_end, i*4)[0] for i in range(8)]}")

        # Compare with layer 0
        print(f"\n=== COMPARISON WITH LAYER 0 ===")

        ffn_down0 = next(t for t in tensors if t['name'] == 'blk.0.ffn_down.weight')
        ffn_sub_norm0 = next(t for t in tensors if t['name'] == 'blk.0.ffn_sub_norm.weight')

        ffn_down0_abs = data_offset + ffn_down0['offset']
        ffn_down0_size = tensor_size(ffn_down0['dims'], ffn_down0['dtype'])
        ffn_down0_end = ffn_down0_abs + ffn_down0_size
        ffn_sub_norm0_abs = data_offset + ffn_sub_norm0['offset']

        print(f"\nblk.0.ffn_down ends at: {ffn_down0_end}")
        print(f"blk.0.ffn_sub_norm starts at: {ffn_sub_norm0_abs}")
        print(f"Gap: {ffn_sub_norm0_abs - ffn_down0_end}")

        print(f"\nFirst 64 bytes of blk.0.ffn_sub_norm:")
        first0 = mm[ffn_sub_norm0_abs:ffn_sub_norm0_abs + 64]
        print(f"  As F32: {[struct.unpack_from('<f', first0, i*4)[0] for i in range(8)]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_layer1.py <path>")
        sys.exit(1)
    check_layer1(sys.argv[1])
