"""7B Model Throughput Benchmark with all optimizations."""
import torch
import time
import warnings
import os
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

print("=" * 70)
print(" BitNet 7B Model - Throughput Optimization Results")
print("=" * 70)

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

for name, (out_dim, in_dim) in layer_weights.items():
    w = torch.randn(out_dim, in_dim)
    packed, scale = quantize_to_bitnet(w)
    packed_weights[name] = packed
    scales[name] = scale

def benchmark_method(method, x, label, iterations=100):
    """Benchmark a single layer forward pass."""
    def forward():
        # Note: This correctly handles intermediate dimensions
        q = method.apply(packed_weights["q_proj"], scales["q_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        k = method.apply(packed_weights["k_proj"], scales["k_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        v = method.apply(packed_weights["v_proj"], scales["v_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        o = method.apply(packed_weights["o_proj"], scales["o_proj"], x, HIDDEN_DIM, HIDDEN_DIM)
        gate = method.apply(packed_weights["gate_proj"], scales["gate_proj"], x, INTERMEDIATE_DIM, HIDDEN_DIM)
        up = method.apply(packed_weights["up_proj"], scales["up_proj"], x, INTERMEDIATE_DIM, HIDDEN_DIM)
        hidden = gate * up
        down = method.apply(packed_weights["down_proj"], scales["down_proj"], hidden, HIDDEN_DIM, INTERMEDIATE_DIM)
        return down

    # Warmup
    for _ in range(10):
        _ = forward()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = forward()
    layer_time = (time.perf_counter() - start) / iterations * 1000

    batch_size = x.shape[0]
    model_time = layer_time * NUM_LAYERS
    tok_s = batch_size * 1000 / model_time

    print(f"{label}: layer={layer_time:.2f}ms, model={model_time:.0f}ms, {tok_s:.1f} tok/s")
    return tok_s

print("\n--- Configuration Comparison ---")

# Test different configurations
configs = [
    ("BF16 + cache", BitNetLinearMethod(compute_dtype=torch.bfloat16, pretranspose=False)),
    ("BF16 + cache + pretranspose", BitNetLinearMethod(compute_dtype=torch.bfloat16, pretranspose=True)),
    ("FP16 + cache", BitNetLinearMethod(compute_dtype=torch.float16, pretranspose=False)),
    ("FP16 + cache + pretranspose", BitNetLinearMethod(compute_dtype=torch.float16, pretranspose=True)),
]

print("\nSingle Token (batch=1):")
x1 = torch.randn(1, HIDDEN_DIM, dtype=torch.float32)
for label, _ in configs:
    # Create fresh method for each test
    if "FP16" in label:
        method = BitNetLinearMethod(compute_dtype=torch.float16, pretranspose="pretranspose" in label)
    else:
        method = BitNetLinearMethod(compute_dtype=torch.bfloat16, pretranspose="pretranspose" in label)
    benchmark_method(method, x1, label)

print("\nBatched (batch=32):")
x32 = torch.randn(32, HIDDEN_DIM, dtype=torch.float32)
for label, _ in configs:
    if "FP16" in label:
        method = BitNetLinearMethod(compute_dtype=torch.float16, pretranspose="pretranspose" in label)
    else:
        method = BitNetLinearMethod(compute_dtype=torch.bfloat16, pretranspose="pretranspose" in label)
    benchmark_method(method, x32, label, iterations=50)

print("\nLarge Batch (batch=128):")
x128 = torch.randn(128, HIDDEN_DIM, dtype=torch.float32)
for label, _ in configs:
    if "FP16" in label:
        method = BitNetLinearMethod(compute_dtype=torch.float16, pretranspose="pretranspose" in label)
    else:
        method = BitNetLinearMethod(compute_dtype=torch.bfloat16, pretranspose="pretranspose" in label)
    benchmark_method(method, x128, label, iterations=20)

print("\nMax Batch (batch=256):")
x256 = torch.randn(256, HIDDEN_DIM, dtype=torch.float32)
best_method = BitNetLinearMethod(compute_dtype=torch.bfloat16, pretranspose=True)
best_toks = benchmark_method(best_method, x256, "BF16 + pretranspose", iterations=10)

print("\n" + "=" * 70)
print(" SUMMARY")
print("=" * 70)
print(f"Best throughput: {best_toks:.1f} tok/s (batch=256, BF16)")
print(f"Model: 7B params, ~1.54 GB packed")
print(f"CPU: {os.cpu_count()} cores, 8 threads used")
print("=" * 70)
