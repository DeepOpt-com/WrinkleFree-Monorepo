#!/usr/bin/env python3
"""Check if per-tensor scales are in the I2_S extra 32 bytes."""

import sys
import struct
import mmap
from safetensors import safe_open
import torch


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


def check_scales(gguf_path, st_path):
    print("=== CHECKING SCALES IN I2_S EXTRA BYTES ===\n")

    # Load safetensors scales
    scales = {}
    with safe_open(st_path, framework='pt') as f:
        for k in f.keys():
            if 'weight_scale' in k:
                scales[k] = f.get_tensor(k).float().item()

    print("Safetensors scales:")
    for k, v in list(scales.items())[:10]:
        print(f"  {k}: {v}")

    # Parse GGUF
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

        # Map GGUF names to safetensors scale names
        name_map = {
            'blk.0.attn_q.weight': 'model.layers.0.self_attn.q_proj.weight_scale',
            'blk.0.attn_k.weight': 'model.layers.0.self_attn.k_proj.weight_scale',
            'blk.0.attn_v.weight': 'model.layers.0.self_attn.v_proj.weight_scale',
            'blk.0.attn_output.weight': 'model.layers.0.self_attn.o_proj.weight_scale',
            'blk.0.ffn_gate.weight': 'model.layers.0.mlp.gate_proj.weight_scale',
            'blk.0.ffn_up.weight': 'model.layers.0.mlp.up_proj.weight_scale',
            'blk.0.ffn_down.weight': 'model.layers.0.mlp.down_proj.weight_scale',
        }

        print("\n=== I2_S TENSOR EXTRA 32 BYTES ===")
        for gguf_name, st_name in name_map.items():
            t = next((t for t in tensors if t['name'] == gguf_name), None)
            if t and t['dtype'] == 36:  # I2_S
                n_elements = 1
                for d in t['dims']:
                    n_elements *= d

                abs_offset = data_offset + t['offset']
                packed_size = n_elements // 4
                total_size = packed_size + 32

                # Read the last 32 bytes (extra data)
                extra_start = abs_offset + packed_size
                extra = mm[extra_start:extra_start + 32]

                # Try interpreting as F32
                f32_vals = [struct.unpack_from('<f', extra, i*4)[0] for i in range(8)]

                # Also try first 4 bytes as scale
                first_f32 = f32_vals[0]

                expected = scales.get(st_name, 'N/A')

                print(f"\n{gguf_name}:")
                print(f"  Expected scale (safetensors): {expected}")
                print(f"  Extra 32 bytes as F32: {[f'{v:.4f}' for v in f32_vals]}")

                # Check if any F32 matches
                match = False
                for i, v in enumerate(f32_vals):
                    if abs(v - expected) < 0.01:
                        print(f"  ✓ MATCH at position {i}!")
                        match = True

                if not match:
                    # Try BF16
                    bf16_vals = []
                    for i in range(16):
                        u16 = struct.unpack_from('<H', extra, i*2)[0]
                        f32 = struct.unpack('<f', struct.pack('<I', u16 << 16))[0]
                        bf16_vals.append(f32)
                    print(f"  Extra as BF16: {[f'{v:.4f}' for v in bf16_vals[:8]]}")

                    for i, v in enumerate(bf16_vals):
                        if abs(v - expected) < 0.1:
                            print(f"  ✓ MATCH as BF16 at position {i}!")
                            match = True


if __name__ == "__main__":
    check_scales('/tmp/bitnet-gguf/ggml-model-i2_s.gguf', '/tmp/bitnet-st/model.safetensors')
