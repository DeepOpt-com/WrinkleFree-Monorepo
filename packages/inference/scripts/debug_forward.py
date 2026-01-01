#!/usr/bin/env python3
"""Debug script to trace forward pass and find where zeros are introduced."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import math

MAGIC = b"SGLBITNT"
QK_I2_S = 128


@dataclass
class BitNetConfig:
    vocab_size: int = 128256
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    head_dim: int = 128
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    @classmethod
    def from_dict(cls, data: dict) -> "BitNetConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_sglkernel_binary(path: Path) -> Tuple[BitNetConfig, dict]:
    """Load model from sgl-kernel binary format."""
    tensors = {}

    with open(path, 'rb') as f:
        magic = f.read(8)
        assert magic == MAGIC, f"Invalid magic: {magic}"

        version = struct.unpack('<I', f.read(4))[0]
        assert version == 1

        config_len = struct.unpack('<I', f.read(4))[0]
        config_json = json.loads(f.read(config_len).decode('utf-8'))
        config = BitNetConfig.from_dict(config_json)

        num_tensors = struct.unpack('<I', f.read(4))[0]
        print(f"Loading {num_tensors} tensors...")

        for i in range(num_tensors):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')

            dtype_id = struct.unpack('<I', f.read(4))[0]
            dtype_map = {0: torch.uint8, 1: torch.float32, 2: torch.float16, 3: torch.bfloat16}
            dtype = dtype_map[dtype_id]

            ndims = struct.unpack('<I', f.read(4))[0]
            shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndims)]

            has_scale = struct.unpack('<I', f.read(4))[0]
            scale_bytes = f.read(4)  # Always read 4 bytes
            scale = struct.unpack('<f', scale_bytes)[0] if has_scale else None

            data_size = struct.unpack('<Q', f.read(8))[0]
            data = f.read(data_size)

            tensor = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)
            tensors[name] = {'tensor': tensor.clone(), 'scale': scale}

    print(f"Loaded {len(tensors)} tensors")
    return config, tensors


def unpack_weights_simd(packed: torch.Tensor) -> torch.Tensor:
    """Unpack SIMD block-interleaved 2-bit weights to ternary tensor."""
    M, K_packed = packed.shape
    K = K_packed * 4

    out = torch.zeros(M, K, dtype=torch.float32)

    num_blocks = K // QK_I2_S
    for block_idx in range(num_blocks):
        base_w = block_idx * QK_I2_S
        base_p = block_idx * 32

        for j in range(32):
            byte_val = packed[:, base_p + j]
            out[:, base_w + j + 0] = ((byte_val >> 6) & 0x03).float() - 1.0
            out[:, base_w + j + 32] = ((byte_val >> 4) & 0x03).float() - 1.0
            out[:, base_w + j + 64] = ((byte_val >> 2) & 0x03).float() - 1.0
            out[:, base_w + j + 96] = ((byte_val >> 0) & 0x03).float() - 1.0

    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(x.dtype)


def main():
    model_path = Path("/home/lev/models/dlm-bitnet-2b.bin")

    print("=" * 60)
    print("Loading model...")
    config, tensors = load_sglkernel_binary(model_path)
    print(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")

    # Test input
    input_ids = torch.tensor([[128000, 2]])  # BOS token + "a"

    # 1. Embedding
    embed_weight = tensors['model.embed_tokens.weight']['tensor'].float()
    print(f"\n1. EMBEDDING")
    print(f"   embed_weight shape: {embed_weight.shape}, dtype: {embed_weight.dtype}")
    print(f"   embed_weight range: [{embed_weight.min():.4f}, {embed_weight.max():.4f}]")

    hidden = embed_weight[input_ids]  # [1, 2, hidden_size]
    print(f"   hidden after embed: shape={hidden.shape}")
    print(f"   hidden range: [{hidden.min():.4f}, {hidden.max():.4f}]")

    # 2. First layer input_layernorm
    norm_weight = tensors['model.layers.0.input_layernorm.weight']['tensor'].float()
    print(f"\n2. LAYER 0 INPUT LAYERNORM")
    print(f"   norm_weight range: [{norm_weight.min():.4f}, {norm_weight.max():.4f}]")

    hidden_norm = rms_norm(hidden, norm_weight, config.rms_norm_eps)
    print(f"   hidden after norm: range=[{hidden_norm.min():.4f}, {hidden_norm.max():.4f}]")

    # 3. Q projection
    q_proj_packed = tensors['model.layers.0.self_attn.q_proj.weight']['tensor']
    q_proj_scale = tensors['model.layers.0.self_attn.q_proj.weight']['scale']
    print(f"\n3. Q_PROJ")
    print(f"   packed weight shape: {q_proj_packed.shape}, scale: {q_proj_scale:.4f}")

    # Unpack weights
    q_proj_weight = unpack_weights_simd(q_proj_packed) * q_proj_scale
    print(f"   unpacked weight shape: {q_proj_weight.shape}")
    print(f"   unpacked weight range: [{q_proj_weight.min():.4f}, {q_proj_weight.max():.4f}]")

    # Check ternary values
    unique_before_scale = torch.unique(unpack_weights_simd(q_proj_packed))
    print(f"   ternary values (before scale): {unique_before_scale.tolist()}")

    # Linear forward
    q = F.linear(hidden_norm, q_proj_weight)
    print(f"   q output: shape={q.shape}, range=[{q.min():.4f}, {q.max():.4f}]")

    # 4. Check final norm and embedding for lm_head
    print(f"\n4. LM_HEAD (tied embeddings)")
    print(f"   Will use embed_weight for logits computation")

    # Simulate final layer output
    final_norm_weight = tensors['model.norm.weight']['tensor'].float()
    print(f"   final norm weight range: [{final_norm_weight.min():.4f}, {final_norm_weight.max():.4f}]")

    # Test logits computation directly with first hidden state
    test_hidden = hidden[:, 0:1, :]  # [1, 1, hidden_size]
    test_hidden_norm = rms_norm(test_hidden, final_norm_weight, config.rms_norm_eps)
    print(f"   test hidden after final norm: range=[{test_hidden_norm.min():.4f}, {test_hidden_norm.max():.4f}]")

    # Compute logits
    logits = F.linear(test_hidden_norm, embed_weight)
    print(f"\n5. LOGITS")
    print(f"   logits shape: {logits.shape}")
    print(f"   logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"   logits[0,0,:10]: {logits[0, 0, :10].tolist()}")

    # Check for all zeros
    if logits.abs().max() < 1e-6:
        print("\n   *** PROBLEM: Logits are all zeros! ***")
        print(f"   test_hidden_norm.abs().max() = {test_hidden_norm.abs().max():.6f}")
        print(f"   embed_weight.abs().max() = {embed_weight.abs().max():.6f}")
    else:
        print("\n   *** Logits look OK ***")
        top_token = logits.argmax(dim=-1)
        print(f"   top token: {top_token.item()}")


if __name__ == "__main__":
    main()
