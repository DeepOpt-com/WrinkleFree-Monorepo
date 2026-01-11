#!/usr/bin/env python3
"""Compare gate_proj forward pass: HF AutoBitLinear vs my implementation."""

import torch
import numpy as np
from safetensors import safe_open
import struct
import mmap


def load_gguf_i2s_tensor(gguf_path, tensor_name):
    """Load a single I2_S tensor from GGUF."""
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

    def decode_i2s(data, n_elements):
        output = np.zeros(n_elements, dtype=np.int8)
        idx = 0
        for byte in data:
            if idx >= n_elements:
                break
            for shift in [6, 4, 2, 0]:
                if idx >= n_elements:
                    break
                val = (byte >> shift) & 0x03
                output[idx] = val - 1
                idx += 1
        return output

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
            packed_data = bytes(mm[abs_offset:abs_offset + packed_size])
            scale_bytes = mm[abs_offset + packed_size:abs_offset + packed_size + 4]
            scale = struct.unpack('<f', scale_bytes)[0]
            ternary = decode_i2s(packed_data, n_elements)
            return {'data': ternary, 'dims': info['dims'], 'scale': scale}

        return None


def load_f32_tensor(gguf_path, tensor_name):
    """Load a F32 tensor from GGUF."""
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
        if info['dtype'] != 0:  # F32
            return None

        abs_offset = data_offset + info['offset']
        n_elements = 1
        for d in info['dims']:
            n_elements *= d
        n_bytes = n_elements * 4
        data = np.frombuffer(mm[abs_offset:abs_offset + n_bytes], dtype=np.float32).copy()
        return data


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization."""
    variance = (x ** 2).mean(-1, keepdims=True)
    rms = np.sqrt(variance + eps)
    return (x / rms) * weight


def main():
    gguf_path = "/tmp/bitnet-gguf/ggml-model-i2_s.gguf"
    hf_path = "/tmp/bitnet-hf/model.safetensors"

    # Create identical test input (embedding for "The")
    # Load embedding from GGUF
    print("Loading embeddings from HF safetensors...")
    with safe_open(hf_path, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float().numpy()

    # Token ID for "The" is 791
    the_token_id = 791
    embedding = embeddings[the_token_id].astype(np.float32)
    print(f"Embedding for 'The' (token {the_token_id}):")
    print(f"  First 8: {embedding[:8]}")
    print(f"  Range: [{embedding.min():.4f}, {embedding.max():.4f}]")

    # Load attn_norm weights from GGUF
    attn_norm_weight = load_f32_tensor(gguf_path, "blk.0.attn_norm.weight")
    print(f"\nAttn norm weight first 8: {attn_norm_weight[:8]}")

    # Apply attention norm
    normed_input = rms_norm(embedding.reshape(1, -1), attn_norm_weight)[0]
    print(f"\nNormed input (after attn_norm):")
    print(f"  First 8: {normed_input[:8]}")
    print(f"  Range: [{normed_input.min():.4f}, {normed_input.max():.4f}]")

    # Load gate_proj weights from GGUF
    gate_gguf = load_gguf_i2s_tensor(gguf_path, "blk.0.ffn_gate.weight")
    in_features = gate_gguf['dims'][0]
    out_features = gate_gguf['dims'][1]
    W_gguf = gate_gguf['data'].reshape(out_features, in_features)
    weight_scale = gate_gguf['scale']
    print(f"\nGGUF gate_proj: ({out_features}, {in_features}), scale={weight_scale:.6f}")

    # Load gate_proj weights from HF
    with safe_open(hf_path, framework="pt") as f:
        W_hf_packed = f.get_tensor("model.layers.0.mlp.gate_proj.weight")
        if W_hf_packed.dtype == torch.uint8:
            W_hf_packed = W_hf_packed.numpy()
        else:
            W_hf_packed = W_hf_packed.float().numpy()
        hf_weight_scale = f.get_tensor("model.layers.0.mlp.gate_proj.weight_scale").float().item()
    print(f"HF gate_proj packed shape: {W_hf_packed.shape}")
    print(f"HF weight_scale: {hf_weight_scale:.6f}")

    # Unpack HF weights (same as HF unpack_weights)
    packed_shape = W_hf_packed.shape
    original_row_dim = packed_shape[0] * 4
    unpacked_shape = (original_row_dim, *packed_shape[1:])
    W_hf = np.zeros(unpacked_shape, dtype=np.uint8)
    for i in range(4):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = 3 << (2 * i)
        W_hf[start:end] = (W_hf_packed & mask) >> (2 * i)
    W_hf = W_hf.astype(np.float32) - 1  # Convert to ternary (-1, 0, 1)
    print(f"HF unpacked shape: {W_hf.shape}")

    # Compare ternary weights
    W_gguf_f = W_gguf.astype(np.float32)
    print(f"\nWeight comparison:")
    print(f"  GGUF first row[:10]: {W_gguf_f[0, :10]}")
    print(f"  HF first row[:10]: {W_hf[0, :10]}")
    match_rate = (W_gguf_f == W_hf).mean()
    print(f"  Match rate: {match_rate:.1%}")

    # === HF AutoBitLinear forward ===
    print("\n" + "="*50)
    print("=== HuggingFace AutoBitLinear Forward ===")
    # ActQuant fake quantization
    hf_input = normed_input.copy()
    scale_hf = 127 / np.abs(hf_input).max()
    quant_hf = np.round(hf_input * scale_hf).clip(-128, 127)
    dequant_hf = quant_hf / scale_hf
    print(f"HF fake quant input range: [{dequant_hf.min():.4f}, {dequant_hf.max():.4f}]")

    # Matrix multiply (float)
    hf_output_raw = W_hf @ dequant_hf
    print(f"HF matmul output range: [{hf_output_raw.min():.2f}, {hf_output_raw.max():.2f}]")

    # Apply weight_scale
    hf_output = hf_output_raw * hf_weight_scale
    print(f"HF final output range: [{hf_output.min():.2f}, {hf_output.max():.2f}]")
    print(f"HF output first 8: {hf_output[:8]}")

    # === My int8 implementation ===
    print("\n" + "="*50)
    print("=== My Int8 Implementation ===")
    my_input = normed_input.copy()
    max_abs = np.abs(my_input).max()
    input_scale = max_abs / 127.0
    input_quant = np.round(my_input / input_scale).clip(-127, 127).astype(np.int8)
    print(f"My input_scale: {input_scale:.6f}")
    print(f"My quant input range: [{input_quant.min()}, {input_quant.max()}]")

    # Integer matrix multiply
    output_int = W_gguf.astype(np.int32) @ input_quant.astype(np.int32)
    print(f"My int output range: [{output_int.min()}, {output_int.max()}]")

    # Apply combined scale (multiply by both scales)
    combined_scale = input_scale * weight_scale
    my_output = output_int.astype(np.float32) * combined_scale
    print(f"My combined_scale: {combined_scale:.6f}")
    print(f"My final output range: [{my_output.min():.2f}, {my_output.max():.2f}]")
    print(f"My output first 8: {my_output[:8]}")

    # === Comparison ===
    print("\n" + "="*50)
    print("=== Comparison ===")
    ratio = np.abs(my_output).max() / np.abs(hf_output).max()
    print(f"Magnitude ratio (my/HF): {ratio:.4f}x")

    # Check if outputs are proportional
    correlation = np.corrcoef(my_output, hf_output)[0, 1]
    print(f"Correlation: {correlation:.6f}")

    # Element-wise difference
    diff = np.abs(my_output - hf_output)
    print(f"Max absolute diff: {diff.max():.4f}")
    print(f"Mean absolute diff: {diff.mean():.4f}")
    print(f"Relative error: {(diff / (np.abs(hf_output) + 1e-6)).mean():.2%}")

    if ratio > 1.5 or ratio < 0.67:
        print(f"\n*** WARNING: Outputs differ by {ratio:.2f}x! ***")
        print("Investigating...")

        # Check if weight_scales match
        print(f"\n  Weight scale comparison:")
        print(f"    GGUF: {weight_scale:.6f}")
        print(f"    HF: {hf_weight_scale:.6f}")
        print(f"    Ratio: {weight_scale / hf_weight_scale:.6f}")


if __name__ == "__main__":
    main()
