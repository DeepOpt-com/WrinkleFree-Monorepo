#!/usr/bin/env python3
"""Determine if the 32 extra I2_S bytes are at the beginning or end."""

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


def check_32_bytes(path):
    print("=== CHECKING 32 EXTRA BYTES POSITION IN I2_S ===\n")

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

        print(f"Data section starts at: {data_offset}")

        # Find first I2_S tensor and the tensor after it
        i2s_tensors = [t for t in tensors if t['dtype'] == 36]
        print(f"\nFound {len(i2s_tensors)} I2_S tensors")

        # Sort all tensors by offset to find sequential pairs
        tensors_by_offset = sorted(tensors, key=lambda t: t['offset'])

        # Find an I2_S tensor followed by an F32 tensor
        for i, t in enumerate(tensors_by_offset[:-1]):
            if t['dtype'] == 36:  # I2_S
                next_t = tensors_by_offset[i + 1]
                if next_t['dtype'] == 0:  # F32
                    n_elements = 1
                    for d in t['dims']:
                        n_elements *= d

                    i2s_abs = data_offset + t['offset']
                    packed_size = n_elements // 4
                    total_size_with_32 = packed_size + 32

                    # Align
                    end_no_extra = i2s_abs + packed_size
                    end_with_extra = i2s_abs + total_size_with_32

                    aligned_end_no_extra = end_no_extra
                    if aligned_end_no_extra % 32 != 0:
                        aligned_end_no_extra = (aligned_end_no_extra // 32 + 1) * 32

                    aligned_end_with_extra = end_with_extra
                    if aligned_end_with_extra % 32 != 0:
                        aligned_end_with_extra = (aligned_end_with_extra // 32 + 1) * 32

                    next_abs = data_offset + next_t['offset']

                    print(f"\n=== I2_S -> F32 BOUNDARY ===")
                    print(f"I2_S tensor: {t['name']}")
                    print(f"  dims: {t['dims']}, elements: {n_elements}")
                    print(f"  starts at: {i2s_abs}")
                    print(f"  packed size (n/4): {packed_size}")
                    print(f"  total size (n/4+32): {total_size_with_32}")

                    print(f"\nF32 tensor: {next_t['name']}")
                    print(f"  dims: {next_t['dims']}")
                    print(f"  recorded offset: {next_abs}")

                    print(f"\nHypothesis testing:")
                    print(f"  If 32 bytes at END: I2_S ends at {i2s_abs + total_size_with_32}")
                    print(f"    aligned: {aligned_end_with_extra}")
                    print(f"    next recorded: {next_abs}")
                    print(f"    gap: {next_abs - aligned_end_with_extra}")

                    print(f"  If 32 bytes at START: packed data ends at {i2s_abs + 32 + packed_size}")
                    end_if_start = i2s_abs + 32 + packed_size
                    aligned_if_start = end_if_start
                    if aligned_if_start % 32 != 0:
                        aligned_if_start = (aligned_if_start // 32 + 1) * 32
                    print(f"    aligned: {aligned_if_start}")
                    print(f"    next recorded: {next_abs}")
                    print(f"    gap: {next_abs - aligned_if_start}")

                    # Check what the data looks like at different interpretations
                    print(f"\n=== DATA ANALYSIS ===")

                    # If 32 bytes at START, first 32 bytes should be header
                    print(f"\nFirst 32 bytes of I2_S tensor:")
                    first = mm[i2s_abs:i2s_abs + 32]
                    print(f"  Hex: {' '.join(f'{b:02x}' for b in first)}")
                    print(f"  As F32: {[struct.unpack_from('<f', first, i*4)[0] for i in range(8)]}")

                    # Last 32 bytes (if at END)
                    print(f"\nLast 32 bytes of I2_S tensor (at offset {i2s_abs + total_size_with_32 - 32}):")
                    last = mm[i2s_abs + total_size_with_32 - 32:i2s_abs + total_size_with_32]
                    print(f"  Hex: {' '.join(f'{b:02x}' for b in last)}")
                    print(f"  As F32: {[struct.unpack_from('<f', last, i*4)[0] for i in range(8)]}")

                    # Check next F32 tensor data
                    print(f"\nFirst 32 bytes of next F32 tensor (at {next_abs}):")
                    f32_data = mm[next_abs:next_abs + 32]
                    print(f"  Hex: {' '.join(f'{b:02x}' for b in f32_data)}")
                    f32_vals = [struct.unpack_from('<f', f32_data, i*4)[0] for i in range(8)]
                    print(f"  As F32: {f32_vals}")

                    # Check if values look like norm weights (around 1.0)
                    valid = sum(1 for v in f32_vals if 0.1 < abs(v) < 10.0)
                    print(f"  Valid norm weight values (0.1 < |x| < 10): {valid}/8")

                    # Only show first pair for now
                    break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_32_bytes_position.py <path>")
        sys.exit(1)
    check_32_bytes(sys.argv[1])
