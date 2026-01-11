#!/usr/bin/env python3
"""Full layer 0 forward pass with all intermediate values."""

import torch
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
import struct
import mmap
import math


def load_gguf_tensors(gguf_path):
    """Load all tensors from GGUF file."""
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

        tensors_info = {}
        for _ in range(n_tensors):
            name, consumed = read_string(mm, offset)
            offset += consumed
            n_dims = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack_from('<Q', mm, offset)[0]
                            )
                offset += 8
            dtype = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            rel_offset = struct.unpack_from('<Q', mm, offset)[0]
            offset += 8
            tensors_info[name] = {'dims': dims, 'dtype': dtype, 'offset': rel_offset}

        padding = offset % alignment
        if padding != 0:
            offset += alignment - padding
        data_offset = offset

        result = {}
        for name, info in tensors_info.items():
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
            output[idx] = val - 1
            idx += 1
    return output


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization."""
    variance = (x ** 2).mean(-1, keepdims=True)
    rms = np.sqrt(variance + eps)
    return (x / rms) * weight


def bitlinear_forward(input_vec, ternary_weights, weight_scale, in_features, out_features):
    """BitLinear forward: absmax quantize input, int GEMV, dequantize."""
    max_abs = np.abs(input_vec).max()
    if max_abs == 0:
        max_abs = 1e-8
    input_scale = max_abs / 127.0
    input_quant = np.round(input_vec / input_scale).clip(-127, 127).astype(np.int8)

    # GGUF stores as (in_features, out_features), reshape to (out_features, in_features) for matmul
    W = ternary_weights.reshape(out_features, in_features)
    output_int = W.astype(np.int32) @ input_quant.astype(np.int32)
    output = output_int.astype(np.float32) * input_scale * weight_scale

    return output, input_scale, input_quant


def apply_rope(q, k, seq_len, num_q_heads, num_kv_heads, head_dim, base=10000.0):
    """Apply Rotary Position Embedding."""
    # q shape: (seq_len, num_q_heads * head_dim)
    # k shape: (seq_len, num_kv_heads * head_dim)
    q = q.reshape(seq_len, num_q_heads, head_dim)
    k = k.reshape(seq_len, num_kv_heads, head_dim)

    # Create position indices
    positions = np.arange(seq_len)

    # Create frequency bands (for half of head_dim)
    dim_indices = np.arange(head_dim // 2)
    freqs = 1.0 / (base ** (2.0 * dim_indices / head_dim))

    # Create rotation angles
    angles = np.outer(positions, freqs)  # (seq_len, head_dim // 2)

    cos = np.cos(angles)
    sin = np.sin(angles)

    # Apply rotation
    q_rot = np.zeros_like(q)
    k_rot = np.zeros_like(k)

    for pos in range(seq_len):
        for head in range(num_q_heads):
            for i in range(head_dim // 2):
                q0 = q[pos, head, 2*i]
                q1 = q[pos, head, 2*i + 1]
                c = cos[pos, i]
                s = sin[pos, i]
                q_rot[pos, head, 2*i] = q0 * c - q1 * s
                q_rot[pos, head, 2*i + 1] = q0 * s + q1 * c

        for head in range(num_kv_heads):
            for i in range(head_dim // 2):
                k0 = k[pos, head, 2*i]
                k1 = k[pos, head, 2*i + 1]
                c = cos[pos, i]
                s = sin[pos, i]
                k_rot[pos, head, 2*i] = k0 * c - k1 * s
                k_rot[pos, head, 2*i + 1] = k0 * s + k1 * c

    return q_rot.reshape(seq_len, -1), k_rot.reshape(seq_len, -1)


def main():
    gguf_path = "/tmp/bitnet-gguf/ggml-model-i2_s.gguf"

    print("Loading GGUF tensors...")
    tensors, mm = load_gguf_tensors(gguf_path)

    # Model config (BitNet-2B)
    hidden_size = 2560
    num_heads = 20
    num_kv_heads = 5
    head_dim = hidden_size // num_heads  # 128
    intermediate_size = 6912
    eps = 1e-5

    # Load embedding
    emb = tensors['token_embd.weight']
    vocab_size = emb['dims'][1]
    embeddings = emb['data'].reshape(vocab_size, hidden_size)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BitNet-b1.58-2B-4T")
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt)
    seq_len = len(input_ids)

    print(f"\n=== INPUT ===")
    print(f"Prompt: '{prompt}'")
    print(f"Token IDs: {input_ids}")
    print(f"Seq len: {seq_len}")

    # Get embeddings
    hidden = embeddings[input_ids, :]  # (seq_len, hidden_size)
    print(f"\n=== EMBEDDINGS ===")
    print(f"Shape: {hidden.shape}")
    print(f"Position 0 first 8: {hidden[0, :8]}")
    print(f"Min: {hidden.min():.6f}, Max: {hidden.max():.6f}")

    # === LAYER 0 ===
    print(f"\n{'='*50}")
    print("LAYER 0")
    print(f"{'='*50}")

    # Load layer 0 weights
    attn_norm = tensors['blk.0.attn_norm.weight']['data']
    ffn_norm = tensors['blk.0.ffn_norm.weight']['data']
    attn_sub_norm = tensors['blk.0.attn_sub_norm.weight']['data']
    ffn_sub_norm = tensors['blk.0.ffn_sub_norm.weight']['data']

    print(f"\n--- SUBNORM WEIGHTS ---")
    print(f"Attn SubLN: shape={attn_sub_norm.shape}, min={attn_sub_norm.min():.6f}, max={attn_sub_norm.max():.6f}")
    print(f"  First 8: {attn_sub_norm[:8]}")
    print(f"FFN SubLN: shape={ffn_sub_norm.shape}, min={ffn_sub_norm.min():.6f}, max={ffn_sub_norm.max():.6f}")
    print(f"  First 8: {ffn_sub_norm[:8]}")

    q_proj = tensors['blk.0.attn_q.weight']
    k_proj = tensors['blk.0.attn_k.weight']
    v_proj = tensors['blk.0.attn_v.weight']
    o_proj = tensors['blk.0.attn_output.weight']

    gate_proj = tensors['blk.0.ffn_gate.weight']
    up_proj = tensors['blk.0.ffn_up.weight']
    down_proj = tensors['blk.0.ffn_down.weight']

    # --- ATTENTION ---
    print(f"\n--- ATTENTION ---")

    # Apply attention norm
    normed_for_attn = rms_norm(hidden, attn_norm, eps)
    print(f"\nAfter attn_norm:")
    print(f"  Position 0 first 8: {normed_for_attn[0, :8]}")
    print(f"  Min: {normed_for_attn.min():.6f}, Max: {normed_for_attn.max():.6f}")

    # Q, K, V projections (for all positions)
    q_all = []
    k_all = []
    v_all = []
    for pos in range(seq_len):
        q_out, _, _ = bitlinear_forward(normed_for_attn[pos], q_proj['data'], q_proj['scale'],
                                        q_proj['dims'][0], q_proj['dims'][1])
        k_out, _, _ = bitlinear_forward(normed_for_attn[pos], k_proj['data'], k_proj['scale'],
                                        k_proj['dims'][0], k_proj['dims'][1])
        v_out, _, _ = bitlinear_forward(normed_for_attn[pos], v_proj['data'], v_proj['scale'],
                                        v_proj['dims'][0], v_proj['dims'][1])
        q_all.append(q_out)
        k_all.append(k_out)
        v_all.append(v_out)

    q = np.stack(q_all)  # (seq_len, hidden_size)
    k = np.stack(k_all)  # (seq_len, kv_dim)
    v = np.stack(v_all)  # (seq_len, kv_dim)

    print(f"\nQ projection:")
    print(f"  Shape: {q.shape}")
    print(f"  Position 0 first 8: {q[0, :8]}")
    print(f"  Min: {q.min():.4f}, Max: {q.max():.4f}")

    print(f"\nK projection:")
    print(f"  Shape: {k.shape}")
    print(f"  Min: {k.min():.4f}, Max: {k.max():.4f}")

    # Print Q and K at position 5 (last) BEFORE RoPE
    print(f"\n=== Q BEFORE ROPE (pos 5, first 16) ===")
    print(f"  Q[5, :16]: {q[5, :16]}")
    print(f"\n=== K BEFORE ROPE (pos 5, first 16) ===")
    print(f"  K[5, :16]: {k[5, :16]}")

    print(f"\nV projection:")
    print(f"  Shape: {v.shape}")
    print(f"  Min: {v.min():.4f}, Max: {v.max():.4f}")

    # Apply RoPE (BitNet uses theta=500000, not default 10000)
    kv_dim = num_kv_heads * head_dim  # 640
    q_rope, k_rope = apply_rope(q, k, seq_len, num_heads, num_kv_heads, head_dim, base=500000.0)
    print(f"\nAfter RoPE (Q):")
    print(f"  First 16 Q values: {q_rope[0, :16]}")
    print(f"  Min: {q_rope.min():.4f}, Max: {q_rope.max():.4f}")

    print(f"\nAfter RoPE (K):")
    print(f"  First 16 K values: {k_rope[0, :16]}")
    print(f"  Min: {k_rope.min():.4f}, Max: {k_rope.max():.4f}")

    # Reshape for attention
    q_heads = q_rope.reshape(seq_len, num_heads, head_dim)
    k_heads = k_rope.reshape(seq_len, num_kv_heads, head_dim)
    v_heads = v.reshape(seq_len, num_kv_heads, head_dim)

    print(f"\nAfter RoPE reshape:")
    print(f"  Q heads shape: {q_heads.shape}")
    print(f"  K heads shape: {k_heads.shape}")
    print(f"  V heads shape: {v_heads.shape}")

    # GQA: repeat KV heads
    k_expanded = np.repeat(k_heads, num_heads // num_kv_heads, axis=1)
    v_expanded = np.repeat(v_heads, num_heads // num_kv_heads, axis=1)

    # Compute attention scores
    scale = 1.0 / math.sqrt(head_dim)
    scores = np.zeros((num_heads, seq_len, seq_len))
    for h in range(num_heads):
        scores[h] = q_heads[:, h, :] @ k_expanded[:, h, :].T * scale

    # Print Q values for last position, K values for positions 0 and 1
    print(f"\n=== Q VALUES (head 0, last pos) ===")
    print(f"  Q first 16: {q_heads[-1, 0, :16]}")

    print(f"\n=== K VALUES (head 0, pos 0) ===")
    print(f"  K first 16: {k_heads[0, 0, :16]}")

    print(f"\n=== K VALUES (head 0, pos 1) ===")
    print(f"  K first 16: {k_heads[1, 0, :16]}")

    print(f"\nRaw attention scores (head 0, last position, before causal mask):")
    print(f"  scale: {scale:.6f}")
    print(f"  scores: {scores[0, -1, :]}")

    # Apply causal mask
    causal_mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    scores = scores + causal_mask

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attn_weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

    print(f"\nAttention weights (head 0, last position):")
    print(f"  {attn_weights[0, -1, :]}")

    # Apply attention to values
    attn_output = np.zeros((seq_len, num_heads, head_dim))
    for h in range(num_heads):
        attn_output[:, h, :] = attn_weights[h] @ v_expanded[:, h, :]

    attn_output = attn_output.reshape(seq_len, hidden_size)
    print(f"\nAttention output (before O proj):")
    print(f"  Shape: {attn_output.shape}")
    print(f"  Position 0 first 8: {attn_output[0, :8]}")
    print(f"  Min: {attn_output.min():.4f}, Max: {attn_output.max():.4f}")

    # O projection
    o_outputs = []
    for pos in range(seq_len):
        o_out, _, _ = bitlinear_forward(attn_output[pos], o_proj['data'], o_proj['scale'],
                                        o_proj['dims'][0], o_proj['dims'][1])
        o_outputs.append(o_out)
    o_output = np.stack(o_outputs)

    print(f"\nO projection output:")
    print(f"  Position 0 first 8: {o_output[0, :8]}")
    print(f"  Min: {o_output.min():.4f}, Max: {o_output.max():.4f}")

    # Apply SubLN after O projection
    o_normed = rms_norm(o_output, attn_sub_norm, eps)
    print(f"\nAfter attn SubLN:")
    print(f"  Min: {o_normed.min():.4f}, Max: {o_normed.max():.4f}")

    # Add residual
    hidden_after_attn = hidden + o_normed
    print(f"\nAfter attention residual:")
    print(f"  Position 0 first 8: {hidden_after_attn[0, :8]}")
    print(f"  Min: {hidden_after_attn.min():.4f}, Max: {hidden_after_attn.max():.4f}")

    # --- FFN ---
    print(f"\n--- FFN ---")

    # Apply FFN norm
    normed_for_ffn = rms_norm(hidden_after_attn, ffn_norm, eps)
    print(f"\nAfter ffn_norm:")
    print(f"  Position 0 first 8: {normed_for_ffn[0, :8]}")
    print(f"  Min: {normed_for_ffn.min():.6f}, Max: {normed_for_ffn.max():.6f}")

    # Gate and Up projections (for position 0 to compare with Rust)
    gate_out, gate_scale, _ = bitlinear_forward(normed_for_ffn[0], gate_proj['data'], gate_proj['scale'],
                                                 gate_proj['dims'][0], gate_proj['dims'][1])
    up_out, up_scale, _ = bitlinear_forward(normed_for_ffn[0], up_proj['data'], up_proj['scale'],
                                            up_proj['dims'][0], up_proj['dims'][1])

    print(f"\nGate projection (pos 0):")
    print(f"  First 8: {gate_out[:8]}")
    print(f"  Min: {gate_out.min():.4f}, Max: {gate_out.max():.4f}")

    print(f"\nUp projection (pos 0):")
    print(f"  First 8: {up_out[:8]}")
    print(f"  Min: {up_out.min():.4f}, Max: {up_out.max():.4f}")

    # SqReLU activation
    sqrelu = np.maximum(gate_out, 0) ** 2
    print(f"\nAfter SqReLU:")
    print(f"  First 8: {sqrelu[:8]}")
    print(f"  Min: {sqrelu.min():.4f}, Max: {sqrelu.max():.4f}")

    # Element-wise multiply
    intermediate = sqrelu * up_out
    print(f"\nIntermediate (sqrelu * up):")
    print(f"  First 8: {intermediate[:8]}")
    print(f"  Min: {intermediate.min():.4f}, Max: {intermediate.max():.4f}")

    # Apply SubLN before down projection
    intermediate_normed = rms_norm(intermediate, ffn_sub_norm, eps)
    print(f"\nAfter FFN SubLN:")
    print(f"  First 8: {intermediate_normed[:8]}")
    print(f"  Min: {intermediate_normed.min():.4f}, Max: {intermediate_normed.max():.4f}")

    # Down projection
    down_out, down_scale, _ = bitlinear_forward(intermediate_normed, down_proj['data'], down_proj['scale'],
                                                 down_proj['dims'][0], down_proj['dims'][1])
    print(f"\nDown projection (pos 0):")
    print(f"  First 8: {down_out[:8]}")
    print(f"  Min: {down_out.min():.4f}, Max: {down_out.max():.4f}")

    # Final residual for layer 0
    hidden_after_ffn = hidden_after_attn + down_out
    print(f"\n=== LAYER 0 OUTPUT ===")
    print(f"Position 0 first 8: {hidden_after_ffn[0, :8]}")
    print(f"Min: {hidden_after_ffn.min():.4f}, Max: {hidden_after_ffn.max():.4f}")


if __name__ == "__main__":
    main()
