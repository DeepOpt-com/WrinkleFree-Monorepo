#!/usr/bin/env python
"""Direct BitNet inference bypassing sglang HTTP server.

This script loads the BitNet model and runs inference directly,
measuring the actual model throughput without HTTP/scheduler overhead.
"""

import sys
sys.path.insert(0, "/home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine/extern/sglang-bitnet/python")

import torch
import time
from transformers import AutoTokenizer

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

print("Loading BitNet model for direct inference...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")

# Load model using sglang's model class
from sglang.srt.models.bitnet import BitNetForCausalLM, _unpack_ternary_weights, _pack_ternary_weights
from sglang.srt.configs.model_config import ModelConfig

# Simple model loading
from safetensors.torch import load_file
import json

model_path = "/home/lev/.cache/huggingface/hub/models--microsoft--BitNet-b1.58-2B-4T/snapshots"

# Find the latest snapshot
import os
snapshots = os.listdir(model_path)
latest = sorted(snapshots)[-1]
model_dir = os.path.join(model_path, latest)

print(f"Loading from: {model_dir}")

# Load config
with open(os.path.join(model_dir, "config.json")) as f:
    config = json.load(f)

print(f"Model config: {config['num_hidden_layers']} layers, {config['hidden_size']} hidden, {config['num_attention_heads']} heads")

# Try to load model weights and create model
print("\nLoading model weights...")

# Load safetensors
weights = load_file(os.path.join(model_dir, "model.safetensors"))
print(f"Loaded {len(weights)} tensors")

# Check weight format
# HuggingFace stores weights as [out/4, in] uint8
# Our kernel expects [out, in/4] uint8

print("\n" + "="*60)
print("Repacking HuggingFace weights to kernel format")
print("="*60)

def repack_hf_weight(hf_weight):
    """Convert HF weight format [out/4, in] to kernel format [out, in/4]."""
    packed_rows, in_features = hf_weight.shape
    out_features = packed_rows * 4
    print(f"  HF format: {hf_weight.shape} -> Unpacking to [{out_features}, {in_features}]")

    # Unpack HF format to float
    unpacked = _unpack_ternary_weights(hf_weight)
    print(f"  Unpacked: {unpacked.shape}")

    # Repack to kernel format [out, in/4]
    packed, _ = _pack_ternary_weights(unpacked)
    print(f"  Kernel format: {packed.shape}")
    return packed

# Repack Q projection weight
print("\nRepacking Q projection...")
q_proj_hf = weights["model.layers.0.self_attn.q_proj.weight"]
q_proj_weight = repack_hf_weight(q_proj_hf)
q_proj_scale = weights["model.layers.0.self_attn.q_proj.weight_scale"].item()

hidden_dim = config["hidden_size"]  # 2560
n_tokens = 1

print("\n" + "="*60)
print("Benchmarking BitNet GEMV with actual model weights")
print("="*60)

from sgl_kernel.quantization.bitnet import bitnet_gemv

# Create test input
x = torch.randn(n_tokens, hidden_dim, dtype=torch.bfloat16)

# Quantize to int8
act_scale = x.abs().max().item() / 127.0
x_int8 = (x / act_scale).to(torch.int8).squeeze(0)

print(f"Q projection weight: shape {q_proj_weight.shape}, dtype {q_proj_weight.dtype}")
print(f"Input: {x.shape}, quantized to int8: {x_int8.shape}")

# Time the GEMV operation
n_runs = 1000

# Warmup
for _ in range(50):
    _ = bitnet_gemv(q_proj_weight, x_int8, q_proj_scale)

start = time.perf_counter()
for _ in range(n_runs):
    _ = bitnet_gemv(q_proj_weight, x_int8, q_proj_scale)
gemv_time = (time.perf_counter() - start) / n_runs * 1000

print(f"\nBitNet GEMV time: {gemv_time:.4f}ms per call")

# Calculate expected throughput
# BitNet-2B has 30 layers (from config), each with ~7 GEMV calls
n_layers = config["num_hidden_layers"]  # 30
gemv_per_layer = 7  # q, k, v, o, gate, up, down
total_gemv = n_layers * gemv_per_layer

expected_per_token = gemv_time * total_gemv
expected_tps = 1000 / expected_per_token

print(f"\nWith {n_layers} layers × {gemv_per_layer} GEMV = {total_gemv} GEMV per token:")
print(f"  Expected GEMV time: {expected_per_token:.2f}ms/token")
print(f"  Expected throughput: {expected_tps:.1f} tok/s (from GEMV alone)")

# Now let's also time attention
print("\n" + "="*60)
print("Benchmarking Attention (SDPA)")
print("="*60)

import torch.nn.functional as F

num_heads = config["num_attention_heads"]
head_dim = hidden_dim // num_heads
seq_len = 50

# Create test tensors
q = torch.randn(1, num_heads, 1, head_dim, dtype=torch.bfloat16)
k_cache = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
v_cache = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

# Warmup
for _ in range(50):
    _ = F.scaled_dot_product_attention(q, k_cache, v_cache)

start = time.perf_counter()
for _ in range(n_runs):
    _ = F.scaled_dot_product_attention(q, k_cache, v_cache)
sdpa_time = (time.perf_counter() - start) / n_runs * 1000

print(f"SDPA time (seq_len={seq_len}): {sdpa_time:.4f}ms per layer")
print(f"For {n_layers} layers: {sdpa_time * n_layers:.2f}ms per token")

# Total theoretical time
total_theoretical = expected_per_token + sdpa_time * n_layers
theoretical_tps = 1000 / total_theoretical

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"GEMV time:     {expected_per_token:.2f}ms/token")
print(f"SDPA time:     {sdpa_time * n_layers:.2f}ms/token")
print(f"Total kernel:  {total_theoretical:.2f}ms/token")
print(f"Theoretical:   {theoretical_tps:.1f} tok/s")
print(f"\nActual sglang: ~16 tok/s = ~62ms/token")
print(f"Framework overhead: ~{62 - total_theoretical:.0f}ms/token")
