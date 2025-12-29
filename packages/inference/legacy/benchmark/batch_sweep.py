"""Batch size sweep for 7B model throughput optimization."""
import torch
import time
import warnings
import os
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

print("=" * 70)
print(" CPU Optimization: Batch Size Sweep")
print("=" * 70)

HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 11008
NUM_LAYERS = 32

from wrinklefree_inference.sglang_backend.bitnet_quantization import (
    quantize_to_bitnet, dequantize_bitnet
)

def create_weight(out_dim, in_dim, dtype):
    w = torch.randn(out_dim, in_dim)
    packed, scale = quantize_to_bitnet(w)
    unpacked = dequantize_bitnet(packed, scale, out_dim, in_dim)
    return unpacked.to(dtype)

dtype = torch.bfloat16
W_q = create_weight(HIDDEN_DIM, HIDDEN_DIM, dtype)
W_k = create_weight(HIDDEN_DIM, HIDDEN_DIM, dtype)
W_v = create_weight(HIDDEN_DIM, HIDDEN_DIM, dtype)
W_o = create_weight(HIDDEN_DIM, HIDDEN_DIM, dtype)
W_gate = create_weight(INTERMEDIATE_DIM, HIDDEN_DIM, dtype)
W_up = create_weight(INTERMEDIATE_DIM, HIDDEN_DIM, dtype)
W_down = create_weight(HIDDEN_DIM, INTERMEDIATE_DIM, dtype)

def forward(x):
    q = torch.matmul(x, W_q.T)
    k = torch.matmul(x, W_k.T)
    v = torch.matmul(x, W_v.T)
    o = torch.matmul(x, W_o.T)
    gate = torch.matmul(x, W_gate.T)
    up = torch.matmul(x, W_up.T)
    hidden = gate * up
    down = torch.matmul(hidden, W_down.T)
    return down

print("\nBF16 Batch Size Sweep:")
print("Batch  | Layer_ms | Model_ms |  Tok_s")
print("-" * 45)

best_toks = 0
best_batch = 1

for batch_size in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256]:
    x = torch.randn(batch_size, HIDDEN_DIM, dtype=dtype)

    for _ in range(5): forward(x)

    iters = max(10, 100 // batch_size)
    start = time.perf_counter()
    for _ in range(iters): forward(x)
    layer_time = (time.perf_counter() - start) / iters * 1000

    model_time = layer_time * NUM_LAYERS
    tok_s = batch_size * 1000 / model_time

    if tok_s > best_toks:
        best_toks = tok_s
        best_batch = batch_size

    print(f"{batch_size:>5}  | {layer_time:>8.2f} | {model_time:>8.1f} | {tok_s:>7.1f}")

print(f"\nBest: batch={best_batch}, {best_toks:.1f} tok/s")

# Test FP16 for comparison
print("\n--- FP16 Comparison ---")
dtype_fp16 = torch.float16
W_q_fp16 = W_q.to(dtype_fp16)
W_k_fp16 = W_k.to(dtype_fp16)
W_v_fp16 = W_v.to(dtype_fp16)
W_o_fp16 = W_o.to(dtype_fp16)
W_gate_fp16 = W_gate.to(dtype_fp16)
W_up_fp16 = W_up.to(dtype_fp16)
W_down_fp16 = W_down.to(dtype_fp16)

def forward_fp16(x):
    q = torch.matmul(x, W_q_fp16.T)
    k = torch.matmul(x, W_k_fp16.T)
    v = torch.matmul(x, W_v_fp16.T)
    o = torch.matmul(x, W_o_fp16.T)
    gate = torch.matmul(x, W_gate_fp16.T)
    up = torch.matmul(x, W_up_fp16.T)
    hidden = gate * up
    down = torch.matmul(hidden, W_down_fp16.T)
    return down

for batch_size in [1, 32]:
    x_fp16 = torch.randn(batch_size, HIDDEN_DIM, dtype=dtype_fp16)
    for _ in range(5): forward_fp16(x_fp16)

    iters = max(10, 100 // batch_size)
    start = time.perf_counter()
    for _ in range(iters): forward_fp16(x_fp16)
    layer_time = (time.perf_counter() - start) / iters * 1000

    model_time = layer_time * NUM_LAYERS
    tok_s = batch_size * 1000 / model_time
    print(f"FP16 batch={batch_size}: {layer_time:.2f}ms layer, {tok_s:.1f} tok/s")
