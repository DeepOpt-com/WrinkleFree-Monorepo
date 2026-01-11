#!/usr/bin/env python3
"""Compare GGUF values with original safetensors to verify correctness."""

import sys
import struct
import mmap
from safetensors import safe_open
import numpy as np


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


def get_gguf_tensor_data(mm, data_offset, tensors, name):
    """Get tensor data from GGUF."""
    t = next((t for t in tensors if t['name'] == name), None)
    if t is None:
        return None
    abs_offset = data_offset + t['offset']
    n_elements = 1
    for d in t['dims']:
        n_elements *= d

    if t['dtype'] == 0:  # F32
        n_bytes = n_elements * 4
    else:
        return None  # Only handle F32 for now

    data = mm[abs_offset:abs_offset + n_bytes]
    return np.frombuffer(data, dtype=np.float32)


def compare(gguf_path, safetensors_path):
    print("=== COMPARING GGUF WITH SAFETENSORS ===\n")

    # Load GGUF
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

        padding = offset % alignment
        if padding != 0:
            offset += alignment - padding
        data_offset = offset

        # Load safetensors
        with safe_open(safetensors_path, framework="numpy") as f:
            print("Safetensors tensors:", list(f.keys())[:20], "...\n")

            # Map GGUF names to safetensors names
            # GGUF: blk.0.attn_norm.weight -> model.layers.0.input_layernorm.weight
            # GGUF: blk.0.ffn_norm.weight -> model.layers.0.post_attention_layernorm.weight
            # etc.

            mappings = [
                ("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"),
                ("blk.0.ffn_norm.weight", "model.layers.0.post_attention_layernorm.weight"),
                ("blk.0.attn_sub_norm.weight", "model.layers.0.self_attn.attn_sub_norm.weight"),
                ("blk.0.ffn_sub_norm.weight", "model.layers.0.mlp.ffn_sub_norm.weight"),
            ]

            for gguf_name, st_name in mappings:
                print(f"=== {gguf_name} ===")

                gguf_data = get_gguf_tensor_data(mm, data_offset, tensors, gguf_name)
                if gguf_data is None:
                    print(f"  Not found in GGUF")
                    continue

                try:
                    st_data = f.get_tensor(st_name).astype(np.float32)
                except:
                    print(f"  Not found in safetensors: {st_name}")
                    continue

                print(f"  GGUF first 8: {gguf_data[:8]}")
                print(f"  ST first 8:   {st_data[:8]}")

                # Check if they match
                if np.allclose(gguf_data, st_data, rtol=1e-3):
                    print(f"  ✓ MATCH")
                else:
                    # Check correlation
                    corr = np.corrcoef(gguf_data.flatten(), st_data.flatten())[0,1]
                    print(f"  ✗ MISMATCH (correlation: {corr:.4f})")
                print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: compare_with_safetensors.py <gguf_path> <safetensors_path>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
