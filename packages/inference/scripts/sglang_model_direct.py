#!/usr/bin/env python
"""Use sglang's BitNet model class directly without HTTP server."""

import sys
sys.path.insert(0, "/home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine/extern/sglang-bitnet/python")

import torch
import torch.nn.functional as F
import time
import json
import os

from transformers import AutoTokenizer
from safetensors.torch import load_file

import warnings
warnings.filterwarnings("ignore")

# Import sglang BitNet linear layer directly
from sglang.srt.models.bitnet import BitNetLinear, _pack_ternary_weights, _unpack_ternary_weights

print("="*70)
print("SGLANG BITNETLINEAR DIRECT TEST")
print("="*70)

# Find model path
model_base = "/home/lev/.cache/huggingface/hub/models--microsoft--BitNet-b1.58-2B-4T/snapshots"
snapshots = os.listdir(model_base)
model_path = os.path.join(model_base, sorted(snapshots)[-1])

# Load config
with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f)

print(f"Model: {config['num_hidden_layers']} layers, {config['hidden_size']} hidden")

# Load weights
print("Loading weights...")
weights = load_file(os.path.join(model_path, "model.safetensors"))

# Create BitNetLinear layer for Q projection
hidden_size = config["hidden_size"]
print(f"\nCreating BitNetLinear layer ({hidden_size} -> {hidden_size})...")

q_proj = BitNetLinear(hidden_size, hidden_size, bias=False)

# Load the actual Q projection weights
print("Loading Q projection weights...")
q_weight_hf = weights["model.layers.0.self_attn.q_proj.weight"]
q_scale = weights["model.layers.0.self_attn.q_proj.weight_scale"]

# Convert HF weight format to kernel format
# HF: [out/4, in] -> kernel: [out, in/4]
unpacked = _unpack_ternary_weights(q_weight_hf)
packed, _ = _pack_ternary_weights(unpacked)

q_proj.qweight.data = packed
q_proj.weight_scale.data = q_scale.view(1)
q_proj.eval()

# Create gate projection (larger)
intermediate_size = config["intermediate_size"]
print(f"Creating gate projection ({hidden_size} -> {intermediate_size})...")

gate_proj = BitNetLinear(hidden_size, intermediate_size, bias=False)
gate_weight_hf = weights["model.layers.0.mlp.gate_proj.weight"]
gate_scale = weights["model.layers.0.mlp.gate_proj.weight_scale"]
unpacked = _unpack_ternary_weights(gate_weight_hf)
packed, _ = _pack_ternary_weights(unpacked)
gate_proj.qweight.data = packed
gate_proj.weight_scale.data = gate_scale.view(1)
gate_proj.eval()

# Create down projection
print(f"Creating down projection ({intermediate_size} -> {hidden_size})...")
down_proj = BitNetLinear(intermediate_size, hidden_size, bias=False)
down_weight_hf = weights["model.layers.0.mlp.down_proj.weight"]
down_scale = weights["model.layers.0.mlp.down_proj.weight_scale"]
unpacked = _unpack_ternary_weights(down_weight_hf)
packed, _ = _pack_ternary_weights(unpacked)
down_proj.qweight.data = packed
down_proj.weight_scale.data = down_scale.view(1)
down_proj.eval()

# Benchmark
print("\nBenchmarking sglang BitNetLinear...")
n_runs = 100
x = torch.randn(1, config["hidden_size"], dtype=torch.bfloat16)

# Warmup
for _ in range(10):
    _ = q_proj(x)

start = time.perf_counter()
for _ in range(n_runs):
    _ = q_proj(x)
q_time = (time.perf_counter() - start) / n_runs * 1000

# Time MLP gate
for _ in range(10):
    _ = gate_proj(x)

start = time.perf_counter()
for _ in range(n_runs):
    _ = gate_proj(x)
gate_time = (time.perf_counter() - start) / n_runs * 1000

# Time MLP down (larger input)
mlp_x = torch.randn(1, config["intermediate_size"], dtype=torch.bfloat16)
for _ in range(10):
    _ = down_proj(mlp_x)

start = time.perf_counter()
for _ in range(n_runs):
    _ = down_proj(mlp_x)
down_time = (time.perf_counter() - start) / n_runs * 1000

# Estimate per-token
gemv_per_layer = 7  # q, k, v, o, gate, up, down
avg_gemv = (q_time + gate_time + down_time) / 3
estimated_per_token = avg_gemv * gemv_per_layer * config["num_hidden_layers"]

print(f"\nPer-operation timing (sglang BitNetLinear):")
print(f"  Q projection: {q_time:.3f}ms")
print(f"  Gate proj:    {gate_time:.3f}ms")
print(f"  Down proj:    {down_time:.3f}ms")
print(f"\nEstimated per-token (GEMV only):")
print(f"  Avg GEMV: {avg_gemv:.3f}ms")
print(f"  {gemv_per_layer} GEMVs × {config['num_hidden_layers']} layers = {estimated_per_token:.1f}ms")
print(f"  Theoretical: {1000/estimated_per_token:.1f} tok/s (GEMV only)")

# Measure SDPA timing
print(f"\n{'='*70}")
print("SDPA TIMING")
print("="*70)

num_heads = config["num_attention_heads"]
num_kv_heads = config.get("num_key_value_heads", num_heads)
head_dim = hidden_size // num_heads
seq_len = 50  # Typical context

q = torch.randn(1, num_heads, 1, head_dim, dtype=torch.bfloat16)
k = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
v = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)

for _ in range(10):
    _ = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

start = time.perf_counter()
for _ in range(n_runs):
    _ = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
sdpa_time = (time.perf_counter() - start) / n_runs * 1000

print(f"SDPA time (seq={seq_len}): {sdpa_time:.3f}ms per layer")
print(f"30 layers: {sdpa_time * 30:.1f}ms per token")

# Measure RMS norm
print(f"\n{'='*70}")
print("RMS NORM TIMING")
print("="*70)

rms_weight = torch.ones(hidden_size, dtype=torch.bfloat16)
rms_x = torch.randn(1, hidden_size, dtype=torch.bfloat16)

def rms_norm(x, weight, eps=1e-5):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps).to(x.dtype)) * weight

for _ in range(10):
    _ = rms_norm(rms_x, rms_weight)

start = time.perf_counter()
for _ in range(n_runs):
    _ = rms_norm(rms_x, rms_weight)
rms_time = (time.perf_counter() - start) / n_runs * 1000

print(f"RMS norm time: {rms_time:.4f}ms")
print(f"Per layer (2 norms): {rms_time * 2:.4f}ms")
print(f"30 layers: {rms_time * 2 * 30:.2f}ms per token")

# Total estimate
total_estimated = estimated_per_token + sdpa_time * 30 + rms_time * 2 * 30
print(f"\n{'='*70}")
print("TOTAL ESTIMATED")
print("="*70)
print(f"GEMV:      {estimated_per_token:.1f}ms")
print(f"SDPA:      {sdpa_time * 30:.1f}ms")
print(f"RMS norm:  {rms_time * 2 * 30:.1f}ms")
print(f"Total:     {total_estimated:.1f}ms → {1000/total_estimated:.0f} tok/s")
print(f"\nMy direct: 53ms → 19 tok/s")
print(f"Gap:       {53 - total_estimated:.1f}ms (Python overhead)")

print(f"\n{'='*70}")
print("SUMMARY")
print("="*70)
print(f"sglang HTTP server:       ~16 tok/s (~62ms/token)")
print(f"sglang BitNetLinear:      ~{1000/estimated_per_token:.0f} tok/s (GEMV only)")
print(f"My direct inference:      ~19 tok/s (~53ms/token)")
print(f"Overhead per token:       ~{62 - estimated_per_token:.0f}ms")
