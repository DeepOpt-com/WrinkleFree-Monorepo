#!/usr/bin/env python3
"""Check HuggingFace weight_scale vs GGUF weight_scale."""

import torch
import numpy as np
from safetensors import safe_open
import struct
import mmap

def get_gguf_scale(gguf_path, tensor_name):
    """Get scale from I2_S tensor."""
    def read_string(mm, offset):
        length = struct.unpack_from('<Q', mm, offset)[0]
        s = mm[offset + 8:offset + 8 + length].decode('utf-8')
        return s, 8 + length

    def skip_value(mm, offset, value_type):
        type_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
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

        tensors = {}
        for _ in range(n_tensors):
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
            tensors[name] = {'dims': dims, 'dtype': dtype, 'offset': rel_offset}

        padding = offset % alignment
        if padding != 0:
            offset += alignment - padding
        data_offset = offset

        if tensor_name not in tensors:
            return None

        info = tensors[tensor_name]
        abs_offset = data_offset + info['offset']
        n_elements = 1
        for d in info['dims']:
            n_elements *= d

        if info['dtype'] == 36:  # I2_S
            packed_size = n_elements // 4
            # Scale is in the extra 32 bytes
            scale_bytes = mm[abs_offset + packed_size:abs_offset + packed_size + 4]
            scale = struct.unpack('<f', scale_bytes)[0]
            return scale

        return None


def main():
    gguf_path = "/tmp/bitnet-gguf/ggml-model-i2_s.gguf"
    hf_path = "/tmp/bitnet-hf/model.safetensors"

    print("=== GGUF I2_S Weight Scales ===")
    tensors_to_check = [
        ("blk.0.ffn_gate.weight", "gate_proj"),
        ("blk.0.ffn_up.weight", "up_proj"),
        ("blk.0.ffn_down.weight", "down_proj"),
        ("blk.0.attn_q.weight", "q_proj"),
        ("blk.0.attn_k.weight", "k_proj"),
        ("blk.0.attn_v.weight", "v_proj"),
        ("blk.0.attn_output.weight", "o_proj"),
    ]

    gguf_scales = {}
    for gguf_name, nice_name in tensors_to_check:
        scale = get_gguf_scale(gguf_path, gguf_name)
        print(f"  {nice_name}: {scale:.6f}")
        gguf_scales[nice_name] = scale

    print("\n=== HuggingFace Weight Scales ===")
    with safe_open(hf_path, framework="pt") as f:
        keys = list(f.keys())
        # Check for weight_scale tensors
        scale_keys = [k for k in keys if "weight_scale" in k and "layers.0" in k]
        print(f"  Scale keys found: {scale_keys}")

        if scale_keys:
            for key in scale_keys:
                scale = f.get_tensor(key)
                print(f"  {key}: {scale.item():.6f}")
        else:
            print("  No weight_scale tensors found in safetensors!")
            print("  This suggests weight_scale is computed at runtime or has default value 1.0")

    # The key insight: check what HF BitLinear actually computes
    print("\n=== BitLinear Computation Analysis ===")

    # HuggingFace BitLinear.forward():
    # 1. input_scale = 127 / max_abs(input)
    # 2. input_quant = round(input * input_scale).clamp(-128, 127)
    # 3. y = input_quant @ ternary_weights
    # 4. y = y / (input_scale * weight_scale)
    #
    # Simplifying:
    #   y = (input * 127/max_abs) @ W / ((127/max_abs) * weight_scale)
    #   y = input @ W / weight_scale
    #
    # So HF DIVIDES by weight_scale!

    # My implementation:
    # 1. input_scale = max_abs / 127
    # 2. input_quant = round(input / input_scale).clamp(-127, 127)
    # 3. y = input_quant @ ternary_weights
    # 4. y = y * input_scale * weight_scale
    #
    # Simplifying:
    #   y = (input / (max_abs/127)) @ W * (max_abs/127) * weight_scale
    #   y = input @ W * weight_scale
    #
    # So I MULTIPLY by weight_scale!

    print("HuggingFace: output = matmul / (input_scale * weight_scale) = matmul / weight_scale (simplified)")
    print("My impl:     output = matmul * input_scale * weight_scale = matmul * weight_scale (simplified)")
    print("")
    print("If GGUF weight_scale = 1.5512:")
    print(f"  HF output = matmul / 1.5512 = matmul * {1/1.5512:.4f}")
    print(f"  My output = matmul * 1.5512 = matmul * 1.5512")
    print(f"  Ratio (my/HF) = 1.5512 * 1.5512 = {1.5512 * 1.5512:.4f}")

    print("\n=== THE BUG ===")
    print("My implementation MULTIPLIES by weight_scale")
    print("HuggingFace implementation DIVIDES by weight_scale")
    print("I need to DIVIDE by weight_scale, not multiply!")


if __name__ == "__main__":
    main()
