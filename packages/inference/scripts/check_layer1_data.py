#!/usr/bin/env python3
"""Check layer 1 F32 tensor data specifically."""

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


def check_layer1(path):
    print("=== CHECKING LAYER 1 F32 TENSOR DATA ===\n")

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

        # Check all F32 norm tensors in layers 0-3
        for layer in range(4):
            print(f"\n=== LAYER {layer} ===")
            for tensor_name in ['attn_norm', 'attn_sub_norm', 'ffn_norm', 'ffn_sub_norm']:
                full_name = f"blk.{layer}.{tensor_name}.weight"
                t = next((t for t in tensors if t['name'] == full_name), None)
                if t:
                    abs_offset = data_offset + t['offset']
                    dtype_name = {0: "F32", 1: "F16", 36: "I2_S"}.get(t['dtype'], f"type{t['dtype']}")
                    print(f"\n{t['name']} ({dtype_name})")
                    print(f"  dims: {t['dims']}, offset: {abs_offset}")

                    data = mm[abs_offset:abs_offset + 32]
                    f32_vals = [struct.unpack_from('<f', data, i*4)[0] for i in range(8)]
                    print(f"  First 8 F32 values: {[f'{v:.4f}' for v in f32_vals]}")

                    # Check validity
                    valid = sum(1 for v in f32_vals if 0.1 < abs(v) < 10.0)
                    status = "✓ VALID" if valid >= 6 else "✗ GARBAGE"
                    print(f"  Valid values: {valid}/8 {status}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_layer1_data.py <path>")
        sys.exit(1)
    check_layer1(sys.argv[1])
