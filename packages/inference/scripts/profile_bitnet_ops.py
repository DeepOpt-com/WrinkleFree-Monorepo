#!/usr/bin/env python
"""Profile core BitNet operations to understand overhead."""

import torch
import time
import torch.nn.functional as F

from sgl_kernel.quantization.bitnet import bitnet_gemv

print("Profiling core BitNet operations...\n")

# Setup - BitNet-2B parameters
hidden_dim = 2560
num_heads = 20
head_dim = 128
mlp_hidden = 6912
seq_len = 50

# Weights (packed ternary)
q_weight = torch.randint(0, 256, (hidden_dim, hidden_dim // 4), dtype=torch.uint8)
k_weight = torch.randint(0, 256, (512, hidden_dim // 4), dtype=torch.uint8)  # GQA
v_weight = torch.randint(0, 256, (512, hidden_dim // 4), dtype=torch.uint8)
o_weight = torch.randint(0, 256, (hidden_dim, hidden_dim // 4), dtype=torch.uint8)
gate_weight = torch.randint(0, 256, (mlp_hidden, hidden_dim // 4), dtype=torch.uint8)
up_weight = torch.randint(0, 256, (mlp_hidden, hidden_dim // 4), dtype=torch.uint8)
down_weight = torch.randint(0, 256, (hidden_dim, mlp_hidden // 4), dtype=torch.uint8)

# Activations
x = torch.randn(1, hidden_dim, dtype=torch.bfloat16)
x_int8 = torch.randint(-127, 128, (hidden_dim,), dtype=torch.int8)
mlp_int8 = torch.randint(-127, 128, (mlp_hidden,), dtype=torch.int8)  # For down projection

# KV cache
k_cache = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
v_cache = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)


def time_op(name, op_fn, n=100):
    """Time an operation."""
    # Warmup
    for _ in range(10):
        op_fn()
    start = time.perf_counter()
    for _ in range(n):
        op_fn()
    elapsed = (time.perf_counter() - start) / n * 1000
    return elapsed


print("Raw timing for individual operations:\n")

times = {}
times["rms_norm"] = time_op("rms_norm", lambda: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5))
times["quantize"] = time_op("quantize", lambda: (x / x.abs().max() * 127).to(torch.int8))
times["q_gemv"] = time_op("q_gemv", lambda: bitnet_gemv(q_weight, x_int8, 1.0))
times["k_gemv"] = time_op("k_gemv", lambda: bitnet_gemv(k_weight, x_int8, 1.0))
times["v_gemv"] = time_op("v_gemv", lambda: bitnet_gemv(v_weight, x_int8, 1.0))
times["sdpa"] = time_op("sdpa", lambda: F.scaled_dot_product_attention(
    torch.randn(1, num_heads, 1, head_dim, dtype=torch.bfloat16), k_cache, v_cache))
times["o_gemv"] = time_op("o_gemv", lambda: bitnet_gemv(o_weight, x_int8, 1.0))
times["gate_gemv"] = time_op("gate_gemv", lambda: bitnet_gemv(gate_weight, x_int8, 1.0))
times["up_gemv"] = time_op("up_gemv", lambda: bitnet_gemv(up_weight, x_int8, 1.0))
times["down_gemv"] = time_op("down_gemv", lambda: bitnet_gemv(down_weight, mlp_int8, 1.0))
times["silu_mul"] = time_op("silu_mul", lambda: F.silu(torch.randn(mlp_hidden, dtype=torch.bfloat16)) * torch.randn(mlp_hidden, dtype=torch.bfloat16))
times["residual"] = time_op("residual", lambda: x + torch.randn_like(x))

total = sum(times.values())
print(f"{'Operation':<15} {'Time (ms)':<12} {'% Total':<10}")
print("-" * 40)
for name, t in sorted(times.items(), key=lambda x: -x[1]):
    print(f"{name:<15} {t:.4f}       {t/total*100:.1f}%")
print("-" * 40)
print(f"{'TOTAL':<15} {total:.4f}ms per layer")
print(f"\n28 layers: {total*28:.2f}ms = {1000/(total*28):.1f} tok/s theoretical")

# Now measure with Python function call overhead
print("\n\nMeasuring Python function call overhead...\n")

def full_layer_ops():
    """Simulate full layer with all operations."""
    # RMS Norm
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    # Quantize
    scale = normed.abs().max().item() / 127.0
    x_q = (normed / scale).to(torch.int8).squeeze(0)

    # QKV
    q = bitnet_gemv(q_weight, x_q, scale)
    k = bitnet_gemv(k_weight, x_q, scale)
    v = bitnet_gemv(v_weight, x_q, scale)

    # SDPA
    q_r = q.view(1, num_heads, 1, head_dim).to(torch.bfloat16)
    attn = F.scaled_dot_product_attention(q_r, k_cache, v_cache)

    # O proj
    attn_flat = attn.view(1, -1)
    attn_scale = attn_flat.abs().max().item() / 127.0
    attn_q = (attn_flat / attn_scale).to(torch.int8).squeeze(0)
    o = bitnet_gemv(o_weight, attn_q, attn_scale)

    # Residual
    x2 = x + o.view(1, -1).to(torch.bfloat16)

    # MLP RMS Norm
    mlp_normed = x2 * torch.rsqrt(x2.pow(2).mean(-1, keepdim=True) + 1e-5)

    # MLP quantize
    mlp_scale = mlp_normed.abs().max().item() / 127.0
    mlp_q = (mlp_normed / mlp_scale).to(torch.int8).squeeze(0)

    # Gate, Up
    gate = bitnet_gemv(gate_weight, mlp_q, mlp_scale)
    up = bitnet_gemv(up_weight, mlp_q, mlp_scale)

    # SiLU * up
    hidden = F.silu(gate.to(torch.bfloat16)) * up.to(torch.bfloat16)

    # Down
    hidden_scale = hidden.abs().max().item() / 127.0
    hidden_q = (hidden / hidden_scale).to(torch.int8)
    down = bitnet_gemv(down_weight, hidden_q, hidden_scale)

    # Final residual
    out = x2 + down.view(1, -1).to(torch.bfloat16)
    return out


# Time full layer
n_runs = 100
for _ in range(10):  # Warmup
    _ = full_layer_ops()

start = time.perf_counter()
for _ in range(n_runs):
    _ = full_layer_ops()
full_layer_time = (time.perf_counter() - start) / n_runs * 1000

print(f"Full layer (with Python overhead): {full_layer_time:.4f}ms")
print(f"Sum of individual ops:             {total:.4f}ms")
print(f"Python overhead per layer:         {full_layer_time - total:.4f}ms")
print(f"\n28 layers with overhead: {full_layer_time*28:.2f}ms = {1000/(full_layer_time*28):.1f} tok/s")

# Compare with actual sglang throughput
actual_ms_per_token = 1000 / 16  # ~16 tok/s from benchmarks
actual_layer_ms = actual_ms_per_token / 28

print(f"\nActual sglang performance:")
print(f"  ~16 tok/s = {actual_ms_per_token:.1f}ms/token = {actual_layer_ms:.2f}ms/layer")
print(f"  Framework overhead: {actual_layer_ms - full_layer_time:.2f}ms/layer")
print(f"  Total framework overhead: {(actual_layer_ms - full_layer_time)*28:.1f}ms/token")
