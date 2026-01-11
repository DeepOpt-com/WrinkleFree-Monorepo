#!/usr/bin/env python3
"""Trace layer 0 values manually using safetensors weights."""

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
                # Extract scale from extra 32 bytes
                scale_bytes = mm[abs_offset + packed_size:abs_offset + packed_size + 4]
                scale = struct.unpack('<f', scale_bytes)[0]
                # Decode ternary
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


def bitlinear_forward(input_vec, ternary_weights, weight_scale, in_features, out_features):
    """Manual BitLinear forward pass with absmax quantization."""
    # Quantize input to int8 using absmax
    max_abs = np.abs(input_vec).max()
    input_scale = max_abs / 127.0
    input_quant = np.round(input_vec / input_scale).clip(-127, 127).astype(np.int8)

    # Reshape weights: GGUF stores as (in_features, out_features)
    # For matmul: output = W @ x, we need W to be (out_features, in_features)
    W = ternary_weights.reshape(out_features, in_features)

    # Integer GEMV
    output_int = W.astype(np.int32) @ input_quant.astype(np.int32)

    # Dequantize
    output = output_int.astype(np.float32) * input_scale * weight_scale

    return output


def main():
    gguf_path = "/tmp/bitnet-gguf/ggml-model-i2_s.gguf"

    print("Loading GGUF tensors...")
    tensors, mm = load_gguf_tensors(gguf_path)

    # Print available tensors
    print(f"\nLoaded {len(tensors)} tensors")
    for name, info in list(tensors.items())[:10]:
        print(f"  {name}: dims={info['dims']}, dtype={info['dtype']}")

    # Load embedding
    emb_tensor = tensors.get('token_embd.weight')
    if emb_tensor and emb_tensor['dtype'] in ('f32', 'f16'):
        # GGUF dims: [hidden_size, vocab_size]
        hidden_size = emb_tensor['dims'][0]
        vocab_size = emb_tensor['dims'][1]
        embeddings = emb_tensor['data'].reshape(vocab_size, hidden_size)
        print(f"\nEmbedding: ({vocab_size}, {hidden_size})")
        print(f"Embedding dtype: {emb_tensor['dtype']}")
    else:
        print(f"Embedding not found or wrong type: {emb_tensor}")
        return

    # Get token IDs for the test prompt - use the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BitNet-b1.58-2B-4T")
    # Use same prompt as the Rust hardcoded tokens
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt)
    print(f"\n=== TOKENIZATION ===")
    print(f"Prompt: '{prompt}'")
    print(f"Token IDs: {input_ids}")
    print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids]}")

    # Get embeddings for tokens
    token_embeddings = embeddings[input_ids, :]
    print(f"\n=== EMBEDDINGS ===")
    print(f"Shape: {token_embeddings.shape}")
    print(f"Position 0 first 8: {token_embeddings[0, :8]}")
    print(f"Min: {token_embeddings.min():.6f}, Max: {token_embeddings.max():.6f}")

    # Attention norm weights
    attn_norm = tensors.get('blk.0.attn_norm.weight')
    if attn_norm:
        attn_norm_weight = attn_norm['data']
        print(f"\n=== ATTN NORM WEIGHT ===")
        print(f"First 8: {attn_norm_weight[:8]}")
        print(f"Min: {attn_norm_weight.min():.6f}, Max: {attn_norm_weight.max():.6f}")

        # Apply norm to position 0
        normed, rms = rms_norm(token_embeddings[0:1, :], attn_norm_weight)
        print(f"\n=== AFTER ATTN NORM (pos 0) ===")
        print(f"RMS: {rms[0, 0]:.6f}")
        print(f"First 8: {normed[0, :8]}")
        print(f"Min: {normed.min():.6f}, Max: {normed.max():.6f}")

    # FFN norm weights
    ffn_norm = tensors.get('blk.0.ffn_norm.weight')
    if ffn_norm:
        ffn_norm_weight = ffn_norm['data']
        print(f"\n=== FFN NORM WEIGHT ===")
        print(f"First 8: {ffn_norm_weight[:8]}")
        print(f"Min: {ffn_norm_weight.min():.6f}, Max: {ffn_norm_weight.max():.6f}")

    # Check Q projection
    q_proj = tensors.get('blk.0.attn_q.weight')
    if q_proj and q_proj['dtype'] == 'i2_s':
        in_features = q_proj['dims'][0]
        out_features = q_proj['dims'][1]
        print(f"\n=== Q PROJECTION ===")
        print(f"Shape: ({in_features}, {out_features})")
        print(f"Scale: {q_proj['scale']:.6f}")
        ternary = q_proj['data']
        print(f"Ternary distribution: -1:{np.sum(ternary==-1)}, 0:{np.sum(ternary==0)}, +1:{np.sum(ternary==1)}")

        # Test Q projection on normed input
        q_output = bitlinear_forward(normed[0], ternary, q_proj['scale'], in_features, out_features)
        print(f"Q output first 8: {q_output[:8]}")
        print(f"Q output min: {q_output.min():.4f}, max: {q_output.max():.4f}")

    # Check gate projection
    gate_proj = tensors.get('blk.0.ffn_gate.weight')
    if gate_proj and gate_proj['dtype'] == 'i2_s':
        in_features = gate_proj['dims'][0]
        out_features = gate_proj['dims'][1]
        print(f"\n=== GATE PROJECTION ===")
        print(f"Shape: ({in_features}, {out_features})")
        print(f"Scale: {gate_proj['scale']:.6f}")

        # For FFN, the input would be after attention + residual
        # For now, test with normed embedding
        gate_output = bitlinear_forward(normed[0], gate_proj['data'], gate_proj['scale'], in_features, out_features)
        print(f"Gate output first 8: {gate_output[:8]}")
        print(f"Gate output min: {gate_output.min():.4f}, max: {gate_output.max():.4f}")

    # Check up projection
    up_proj = tensors.get('blk.0.ffn_up.weight')
    if up_proj and up_proj['dtype'] == 'i2_s':
        in_features = up_proj['dims'][0]
        out_features = up_proj['dims'][1]
        print(f"\n=== UP PROJECTION ===")
        print(f"Shape: ({in_features}, {out_features})")
        print(f"Scale: {up_proj['scale']:.6f}")

        up_output = bitlinear_forward(normed[0], up_proj['data'], up_proj['scale'], in_features, out_features)
        print(f"Up output first 8: {up_output[:8]}")
        print(f"Up output min: {up_output.min():.4f}, max: {up_output.max():.4f}")

        # SqReLU activation on gate, then multiply with up
        if gate_proj:
            gate_output = bitlinear_forward(normed[0], gate_proj['data'], gate_proj['scale'],
                                            gate_proj['dims'][0], gate_proj['dims'][1])
            squared_relu = np.maximum(gate_output, 0) ** 2
            print(f"\n=== AFTER SQRELU ===")
            print(f"First 8: {squared_relu[:8]}")
            print(f"Min: {squared_relu.min():.4f}, Max: {squared_relu.max():.4f}")

            intermediate = squared_relu * up_output
            print(f"\n=== INTERMEDIATE (gate_sqrelu * up) ===")
            print(f"First 8: {intermediate[:8]}")
            print(f"Min: {intermediate.min():.4f}, Max: {intermediate.max():.4f}")

    # Check down projection
    down_proj = tensors.get('blk.0.ffn_down.weight')
    if down_proj and down_proj['dtype'] == 'i2_s':
        in_features = down_proj['dims'][0]
        out_features = down_proj['dims'][1]
        print(f"\n=== DOWN PROJECTION ===")
        print(f"Shape: ({in_features}, {out_features})")
        print(f"Scale: {down_proj['scale']:.6f}")

        if gate_proj and up_proj:
            # Compute full FFN
            gate_output = bitlinear_forward(normed[0], gate_proj['data'], gate_proj['scale'],
                                            gate_proj['dims'][0], gate_proj['dims'][1])
            up_output = bitlinear_forward(normed[0], up_proj['data'], up_proj['scale'],
                                          up_proj['dims'][0], up_proj['dims'][1])
            squared_relu = np.maximum(gate_output, 0) ** 2
            intermediate = squared_relu * up_output

            # Down projection
            down_output = bitlinear_forward(intermediate, down_proj['data'], down_proj['scale'],
                                            in_features, out_features)
            print(f"Down output first 8: {down_output[:8]}")
            print(f"Down output min: {down_output.min():.4f}, max: {down_output.max():.4f}")


if __name__ == "__main__":
    main()
