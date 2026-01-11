#!/usr/bin/env python3
"""Test inference using HuggingFace weights directly."""

import torch
import numpy as np
from safetensors import safe_open
import struct


def load_hf_weights(hf_path):
    """Load HuggingFace weights and unpack ternary."""
    with safe_open(hf_path, framework="pt") as f:
        tensors = {}
        for name in f.keys():
            tensor = f.get_tensor(name)
            # Convert bfloat16 to float32 for numpy compatibility
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            tensors[name] = tensor.numpy()
    return tensors


def unpack_ternary(packed, out_features):
    """Unpack HF-style packed ternary weights."""
    packed_rows = packed.shape[0]
    in_features = packed.shape[1]
    assert out_features == packed_rows * 4

    unpacked = np.zeros((out_features, in_features), dtype=np.int8)
    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        mask = 3 << (2 * i)
        unpacked[start:end] = ((packed & mask) >> (2 * i)).astype(np.int8) - 1

    return unpacked


def apply_rms_norm(x, weight, eps=1e-5):
    """Apply RMS normalization."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def quantize_activation(x, eps=1e-5):
    """Quantize activations to int8 using absmax."""
    scale = np.abs(x).max() + eps
    x_int8 = np.clip(np.round(x * 127 / scale), -128, 127).astype(np.int8)
    return x_int8, scale


def matmul_ternary(W_ternary, x_int8, input_scale, weight_scale):
    """Perform ternary matmul like AutoBitLinear."""
    # W_ternary: (out_features, in_features), values in {-1, 0, +1}
    # x_int8: (in_features,), values in [-128, 127]

    # Integer matmul
    result_int = W_ternary.astype(np.int32) @ x_int8.astype(np.int32)

    # Dequantize (AutoBitLinear multiplies by weight_scale)
    result_float = (result_int.astype(np.float32) / 127.0) * input_scale * weight_scale

    return result_float


def relu2(x):
    """Squared ReLU."""
    return np.maximum(x, 0) ** 2


def main():
    hf_path = "/tmp/bitnet-hf/model.safetensors"

    print("Loading HuggingFace weights...")
    tensors = load_hf_weights(hf_path)

    print(f"Loaded {len(tensors)} tensors")
    print("\nKey tensors:")
    for name in sorted(tensors.keys())[:15]:
        print(f"  {name}: {tensors[name].shape}")

    # Get token embeddings
    embed = tensors["model.embed_tokens.weight"]  # (vocab_size, hidden_size)
    print(f"\nEmbedding shape: {embed.shape}")

    # Test with "The capital of France is"
    # Token IDs (assuming same as before): [464, 6864, 315, 9822, 374]
    token_ids = [791, 6864, 315, 9822, 374]  # "The capital of France is"
    print(f"\nToken IDs: {token_ids}")

    # Get input embeddings
    hidden = embed[token_ids]  # (seq_len, hidden_size)
    print(f"Hidden states shape: {hidden.shape}")
    print(f"Hidden states[0, :10]: {hidden[0, :10]}")

    # Apply input layernorm
    ln_weight = tensors["model.layers.0.input_layernorm.weight"]
    hidden_norm = apply_rms_norm(hidden, ln_weight)
    print(f"\nAfter input_layernorm, hidden[:, :10]: {hidden_norm[-1, :10]}")

    # Test gate_proj
    gate_packed = tensors["model.layers.0.mlp.gate_proj.weight"]
    gate_scale = tensors["model.layers.0.mlp.gate_proj.weight_scale"][0]
    print(f"\ngate_proj packed shape: {gate_packed.shape}")
    print(f"gate_proj weight_scale: {gate_scale}")

    # Unpack gate weights
    gate_ternary = unpack_ternary(gate_packed, gate_packed.shape[0] * 4)
    print(f"gate_ternary shape: {gate_ternary.shape}")
    print(f"gate_ternary first row[:10]: {gate_ternary[0, :10]}")

    # Quantize activation (last token)
    x = hidden_norm[-1]  # Last token hidden state
    x_int8, x_scale = quantize_activation(x)
    print(f"\nActivation scale: {x_scale}")
    print(f"x_int8 first 10: {x_int8[:10]}")

    # Matmul with gate_proj
    gate_out = matmul_ternary(gate_ternary, x_int8, x_scale, gate_scale)
    print(f"\ngate_out shape: {gate_out.shape}")
    print(f"gate_out first 10: {gate_out[:10]}")
    print(f"gate_out range: [{gate_out.min():.4f}, {gate_out.max():.4f}]")

    # Similarly for up_proj
    up_packed = tensors["model.layers.0.mlp.up_proj.weight"]
    up_scale = tensors["model.layers.0.mlp.up_proj.weight_scale"][0]
    up_ternary = unpack_ternary(up_packed, up_packed.shape[0] * 4)
    up_out = matmul_ternary(up_ternary, x_int8, x_scale, up_scale)
    print(f"\nup_out shape: {up_out.shape}")
    print(f"up_out range: [{up_out.min():.4f}, {up_out.max():.4f}]")

    # FFN: relu2(gate) * up
    ffn_hidden = relu2(gate_out) * up_out
    print(f"\nffn_hidden (after relu2*up) range: [{ffn_hidden.min():.4f}, {ffn_hidden.max():.4f}]")

    # Apply FFN sub-norm
    ffn_sub_norm = tensors["model.layers.0.mlp.ffn_sub_norm.weight"]
    ffn_norm = apply_rms_norm(ffn_hidden, ffn_sub_norm)
    print(f"ffn_norm range: [{ffn_norm.min():.4f}, {ffn_norm.max():.4f}]")

    # Quantize for down_proj
    ffn_int8, ffn_scale = quantize_activation(ffn_norm)

    # down_proj
    down_packed = tensors["model.layers.0.mlp.down_proj.weight"]
    down_scale = tensors["model.layers.0.mlp.down_proj.weight_scale"][0]
    down_ternary = unpack_ternary(down_packed, down_packed.shape[0] * 4)
    down_out = matmul_ternary(down_ternary, ffn_int8, ffn_scale, down_scale)
    print(f"\ndown_out shape: {down_out.shape}")
    print(f"down_out range: [{down_out.min():.4f}, {down_out.max():.4f}]")

    # This should be similar to what our Rust code computes
    print(f"\nLayer 0 FFN output (down_out) first 10: {down_out[:10]}")


if __name__ == "__main__":
    main()
