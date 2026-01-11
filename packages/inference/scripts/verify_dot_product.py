#!/usr/bin/env python3
"""Verify raw integer dot product for Q projection row 0."""

import torch
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
import struct
import mmap

def load_gguf_tensors(gguf_path):
    """Load tensors from GGUF file."""
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

        result = {}
        for name, info in tensors.items():
            abs_offset = data_offset + info['offset']
            if info['dtype'] == 0:  # F32
                n_elements = 1
                for d in info['dims']:
                    n_elements *= d
                n_bytes = n_elements * 4
                data = np.frombuffer(mm[abs_offset:abs_offset + n_bytes], dtype=np.float32).copy()
                result[name] = {'data': data, 'dims': info['dims'], 'dtype': 'f32'}
            elif info['dtype'] == 1:  # F16
                n_elements = 1
                for d in info['dims']:
                    n_elements *= d
                n_bytes = n_elements * 2
                data = np.frombuffer(mm[abs_offset:abs_offset + n_bytes], dtype=np.float16).copy().astype(np.float32)
                result[name] = {'data': data, 'dims': info['dims'], 'dtype': 'f16'}
            elif info['dtype'] == 36:  # I2_S
                n_elements = 1
                for d in info['dims']:
                    n_elements *= d
                packed_size = n_elements // 4
                packed_data = bytes(mm[abs_offset:abs_offset + packed_size])
                scale_bytes = mm[abs_offset + packed_size:abs_offset + packed_size + 4]
                scale = struct.unpack('<f', scale_bytes)[0]
                ternary = decode_i2s(packed_data, n_elements)
                result[name] = {'data': ternary, 'dims': info['dims'], 'dtype': 'i2_s', 'scale': scale}

        return result, mm


def decode_i2s(data, n_elements):
    """Decode I2_S sequential layout."""
    output = np.zeros(n_elements, dtype=np.int8)
    idx = 0
    for byte in data:
        if idx >= n_elements:
            break
        for shift in [6, 4, 2, 0]:
            if idx >= n_elements:
                break
            val = (byte >> shift) & 0x03
            output[idx] = val - 1  # 00=-1, 01=0, 10=+1
            idx += 1
    return output


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization."""
    variance = (x ** 2).mean(-1, keepdims=True)
    rms = np.sqrt(variance + eps)
    return (x / rms) * weight, rms


def main():
    gguf_path = "/tmp/bitnet-gguf/ggml-model-i2_s.gguf"

    print("Loading GGUF tensors...")
    tensors, mm = load_gguf_tensors(gguf_path)

    # Load embedding
    emb_tensor = tensors.get('token_embd.weight')
    hidden_size = emb_tensor['dims'][0]
    vocab_size = emb_tensor['dims'][1]
    embeddings = emb_tensor['data'].reshape(vocab_size, hidden_size)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BitNet-b1.58-2B-4T")
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt)
    print(f"Token IDs: {input_ids}")

    # Get embedding for position 0
    emb_0 = embeddings[input_ids[0], :]  # Position 0
    print(f"\nEmbedding pos 0, first 8: {emb_0[:8]}")

    # Apply attention norm
    attn_norm_weight = tensors['blk.0.attn_norm.weight']['data']
    normed, rms = rms_norm(emb_0.reshape(1, -1), attn_norm_weight)
    normed = normed[0]  # Shape (2560,)
    print(f"After norm, first 8: {normed[:8]}")
    print(f"RMS: {rms[0, 0]}")

    # Quantize to INT8 (absmax)
    max_abs = np.abs(normed).max()
    input_scale = max_abs / 127.0
    input_quant = np.round(normed / input_scale).clip(-127, 127).astype(np.int8)
    print(f"\nInput scale: {input_scale:.6f}")
    print(f"Input quant first 16: {list(input_quant[:16])}")
    print(f"Sum of quantized activations: {np.sum(input_quant.astype(np.int32))}")

    # Get Q projection weights
    q_proj = tensors['blk.0.attn_q.weight']
    q_ternary = q_proj['data']
    q_scale = q_proj['scale']
    in_features = q_proj['dims'][0]
    out_features = q_proj['dims'][1]
    print(f"\nQ projection: in={in_features}, out={out_features}, scale={q_scale}")
    print(f"Q weight first 32: {list(q_ternary[:32])}")

    # Reshape weights: (out_features, in_features)
    W = q_ternary.reshape(out_features, in_features)
    print(f"W shape: {W.shape}")
    print(f"W[0, :32]: {list(W[0, :32])}")

    # Compute integer dot product for row 0
    row0_weights = W[0, :]  # First output row
    int_dot = np.sum(row0_weights.astype(np.int32) * input_quant.astype(np.int32))
    print(f"\n=== INTEGER DOT PRODUCT (row 0) ===")
    print(f"  dot = {int_dot}")
    print(f"  combined_scale = {input_scale} * {q_scale} = {input_scale * q_scale}")
    print(f"  scaled_result = {int_dot * input_scale * q_scale}")

    # Compare with Rust output
    rust_int_dot = -419
    rust_combined_scale = 0.000797
    print(f"\n=== COMPARISON WITH RUST ===")
    print(f"  Rust int_dot: {rust_int_dot}")
    print(f"  Python int_dot: {int_dot}")
    print(f"  Match: {int_dot == rust_int_dot}")

    if int_dot != rust_int_dot:
        print("\n=== DEBUGGING MISMATCH ===")
        # Check weight sums
        row0_sum = np.sum(row0_weights)
        print(f"  Row 0 weight sum: {row0_sum}")
        print(f"  Row 0 +1 count: {np.sum(row0_weights == 1)}")
        print(f"  Row 0 0 count: {np.sum(row0_weights == 0)}")
        print(f"  Row 0 -1 count: {np.sum(row0_weights == -1)}")

        # Check first few contributions
        print("\n  First 8 contributions (weight * activation):")
        for i in range(8):
            contrib = int(row0_weights[i]) * int(input_quant[i])
            print(f"    [{i}]: w={row0_weights[i]:2d} * a={input_quant[i]:4d} = {contrib:5d}")


def bitlinear_forward_v2(input_vec, ternary_weights, weight_scale, in_features, out_features):
    """Manual BitLinear forward pass with absmax quantization."""
    max_abs = np.abs(input_vec).max()
    input_scale = max_abs / 127.0
    input_quant = np.round(input_vec / input_scale).clip(-127, 127).astype(np.int8)
    W = ternary_weights.reshape(out_features, in_features)
    output_int = W.astype(np.int32) @ input_quant.astype(np.int32)
    return output_int.astype(np.float32) * input_scale * weight_scale


def check_layer0_ffn():
    """Check FFN output magnitudes in layer 0."""
    gguf_path = "/tmp/bitnet-gguf/ggml-model-i2_s.gguf"

    print("Loading GGUF tensors...")
    tensors, mm = load_gguf_tensors(gguf_path)

    # Load embedding
    emb_tensor = tensors.get('token_embd.weight')
    hidden_size = emb_tensor['dims'][0]
    vocab_size = emb_tensor['dims'][1]
    embeddings = emb_tensor['data'].reshape(vocab_size, hidden_size)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BitNet-b1.58-2B-4T")
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt)

    # Get embedding for position 0
    emb_0 = embeddings[input_ids[0], :]
    print(f"\n=== EMBEDDING (pos 0) ===")
    print(f"  Min: {emb_0.min():.4f}, Max: {emb_0.max():.4f}")

    # Apply attention norm
    attn_norm_weight = tensors['blk.0.attn_norm.weight']['data']
    normed, rms = rms_norm(emb_0.reshape(1, -1), attn_norm_weight)
    normed = normed[0]
    print(f"\n=== AFTER ATTN NORM ===")
    print(f"  Min: {normed.min():.4f}, Max: {normed.max():.4f}")

    # Simulate attention output (just use zeros for simplicity)
    # In reality we'd compute attention, but for testing FFN we can skip
    # Use embedding as hidden after attention residual
    hidden_after_attn = emb_0  # Approximate, ignoring attention contribution

    # Apply FFN norm
    ffn_norm_weight = tensors['blk.0.ffn_norm.weight']['data']
    ffn_input, rms = rms_norm(hidden_after_attn.reshape(1, -1), ffn_norm_weight)
    ffn_input = ffn_input[0]
    print(f"\n=== FFN INPUT (after norm) ===")
    print(f"  Min: {ffn_input.min():.4f}, Max: {ffn_input.max():.4f}")

    # Gate projection
    gate_proj = tensors['blk.0.ffn_gate.weight']
    gate_out = bitlinear_forward_v2(
        ffn_input, gate_proj['data'], gate_proj['scale'],
        gate_proj['dims'][0], gate_proj['dims'][1]
    )
    print(f"\n=== GATE OUTPUT ===")
    print(f"  Min: {gate_out.min():.4f}, Max: {gate_out.max():.4f}")

    # Up projection
    up_proj = tensors['blk.0.ffn_up.weight']
    up_out = bitlinear_forward_v2(
        ffn_input, up_proj['data'], up_proj['scale'],
        up_proj['dims'][0], up_proj['dims'][1]
    )
    print(f"\n=== UP OUTPUT ===")
    print(f"  Min: {up_out.min():.4f}, Max: {up_out.max():.4f}")

    # Squared ReLU
    gate_sq = np.maximum(gate_out, 0) ** 2
    print(f"\n=== GATE SQUARED RELU ===")
    print(f"  Min: {gate_sq.min():.4f}, Max: {gate_sq.max():.4f}")

    # Multiply
    intermediate = gate_sq * up_out
    print(f"\n=== INTERMEDIATE (before SubLN) ===")
    print(f"  Min: {intermediate.min():.4f}, Max: {intermediate.max():.4f}")

    # SubLN
    ffn_sub_norm = tensors['blk.0.ffn_sub_norm.weight']['data']
    intermediate_normed, _ = rms_norm(intermediate.reshape(1, -1), ffn_sub_norm)
    intermediate_normed = intermediate_normed[0]
    print(f"\n=== INTERMEDIATE (after SubLN) ===")
    print(f"  Min: {intermediate_normed.min():.4f}, Max: {intermediate_normed.max():.4f}")

    # Down projection
    down_proj = tensors['blk.0.ffn_down.weight']
    ffn_out = bitlinear_forward_v2(
        intermediate_normed, down_proj['data'], down_proj['scale'],
        down_proj['dims'][0], down_proj['dims'][1]
    )
    print(f"\n=== FFN OUTPUT ===")
    print(f"  Min: {ffn_out.min():.4f}, Max: {ffn_out.max():.4f}")

    # Layer output (residual)
    layer_out = hidden_after_attn + ffn_out
    print(f"\n=== LAYER 0 OUTPUT (after residual) ===")
    print(f"  Min: {layer_out.min():.4f}, Max: {layer_out.max():.4f}")


if __name__ == "__main__":
    check_layer0_ffn()
