"""Quick 7B Model Benchmark - reduced iterations."""
import torch
import time
import warnings
import os
import sys
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

# Flush output immediately
sys.stdout.reconfigure(line_buffering=True)

print("=" * 70, flush=True)
print(" BitNet 7B Model - Quick Benchmark", flush=True)
print("=" * 70, flush=True)

HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 11008
NUM_LAYERS = 32

from wrinklefree_inference.sglang_backend.bitnet_quantization import (
    quantize_to_bitnet, BitNetLinearMethod
)

# Create weights
layer_weights = {
    "q_proj": (HIDDEN_DIM, HIDDEN_DIM),
    "k_proj": (HIDDEN_DIM, HIDDEN_DIM),
    "v_proj": (HIDDEN_DIM, HIDDEN_DIM),
    "o_proj": (HIDDEN_DIM, HIDDEN_DIM),
    "gate_proj": (INTERMEDIATE_DIM, HIDDEN_DIM),
    "up_proj": (INTERMEDIATE_DIM, HIDDEN_DIM),
    "down_proj": (HIDDEN_DIM, INTERMEDIATE_DIM),
}

packed_weights = {}
scales = {}

print("Quantizing weights...", flush=True)
for name, (out_dim, in_dim) in layer_weights.items():
    w = torch.randn(out_dim, in_dim)
    packed, scale = quantize_to_bitnet(w)
    packed_weights[name] = packed
    scales[name] = scale
print("Done.", flush=True)

def benchmark_method(method, x, label, iterations=10):
    def forward():
        q = method.apply(packed_weights["q_proj"], scales["q_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        k = method.apply(packed_weights["k_proj"], scales["k_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        v = method.apply(packed_weights["v_proj"], scales["v_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        o = method.apply(packed_weights["o_proj"], scales["o_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        gate = method.apply(packed_weights["gate_proj"], scales["gate_proj"], x, INTERMEDIATE_DIM, HIDDEN_DIM)
        up = method.apply(packed_weights["up_proj"], scales["up_proj"], x, INTERMEDIATE_DIM, HIDDEN_DIM)
        hidden = gate * up
        down = method.apply(packed_weights["down_proj"], scales["down_proj"], hidden, HIDDEN_DIM, INTERMEDIATE_DIM)
        return down

    for _ in range(3): forward()  # warmup

    start = time.perf_counter()
    for _ in range(iterations): forward()
    layer_time = (time.perf_counter() - start) / iterations * 1000

    batch_size = x.shape[0]
    model_time = layer_time * NUM_LAYERS
    tok_s = batch_size * 1000 / model_time
    print(f"{label}: layer={layer_time:.2f}ms, model={model_time:.0f}ms, {tok_s:.1f} tok/s", flush=True)
    return tok_s

print("\n--- Single Token (batch=1) ---", flush=True)
x1 = torch.randn(1, HIDDEN_DIM, dtype=torch.float32)

bf16 = BitNetLinearMethod(compute_dtype=torch.bfloat16, pretranspose=False)
bf16_t = BitNetLinearMethod(compute_dtype=torch.bfloat16, pretranspose=True)
benchmark_method(bf16, x1, "BF16")
benchmark_method(bf16_t, x1, "BF16 + pretrans")

print("\n--- Batch=32 ---", flush=True)
x32 = torch.randn(32, HIDDEN_DIM, dtype=torch.float32)
bf16_32 = BitNetLinearMethod(compute_dtype=torch.bfloat16, pretranspose=True)
benchmark_method(bf16_32, x32, "BF16 + pretrans")

print("\n--- Batch=128 ---", flush=True)
x128 = torch.randn(128, HIDDEN_DIM, dtype=torch.float32)
bf16_128 = BitNetLinearMethod(compute_dtype=torch.bfloat16, pretranspose=True)
benchmark_method(bf16_128, x128, "BF16 + pretrans")

print("\n--- Batch=256 ---", flush=True)
x256 = torch.randn(256, HIDDEN_DIM, dtype=torch.float32)
bf16_256 = BitNetLinearMethod(compute_dtype=torch.bfloat16, pretranspose=True)
best_toks = benchmark_method(bf16_256, x256, "BF16 + pretrans")

print("\n" + "=" * 70, flush=True)
print(f"Best: {best_toks:.1f} tok/s (batch=256)", flush=True)
print("=" * 70, flush=True)
