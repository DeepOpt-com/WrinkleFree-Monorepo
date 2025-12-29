"""Benchmark native BitNet GEMV kernel vs Python baseline."""

import torch
import time
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrinklefree_inference.sglang_backend.bitnet_quantization import (
    quantize_to_bitnet, BitNetLinearMethod
)

def benchmark_python_baseline(packed_weights, scales, x, layer_weights, iterations=50):
    """Benchmark Python implementation."""
    method = BitNetLinearMethod(compute_dtype=torch.bfloat16)

    HIDDEN_DIM = 4096
    INTERMEDIATE_DIM = 11008

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

    # Warmup
    for _ in range(5):
        forward()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        forward()
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000  # ms per layer


def benchmark_native(packed_weights, scales, x, layer_weights, iterations=50):
    """Benchmark native C++ implementation."""
    try:
        import bitnet_native
    except ImportError:
        print("Native extension not available. Building...")
        from wrinklefree_inference.native import build_native
        if not build_native():
            return None
        import bitnet_native

    HIDDEN_DIM = 4096
    INTERMEDIATE_DIM = 11008

    # Convert to float32 for native kernel
    x_f32 = x.float().contiguous()

    def forward():
        q = bitnet_native.gemv(packed_weights["q_proj"], x_f32, scales["q_proj"])
        k = bitnet_native.gemv(packed_weights["k_proj"], x_f32, scales["k_proj"])
        v = bitnet_native.gemv(packed_weights["v_proj"], x_f32, scales["v_proj"])
        o = bitnet_native.gemv(packed_weights["o_proj"], x_f32, scales["o_proj"])
        gate = bitnet_native.gemv(packed_weights["gate_proj"], x_f32, scales["gate_proj"])
        up = bitnet_native.gemv(packed_weights["up_proj"], x_f32, scales["up_proj"])
        hidden = gate * up
        down = bitnet_native.gemv(packed_weights["down_proj"], hidden.contiguous(), scales["down_proj"])
        return down

    # Warmup
    for _ in range(5):
        forward()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        forward()
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000  # ms per layer


def main():
    print("=" * 70)
    print(" Native BitNet GEMV Kernel Benchmark")
    print("=" * 70)

    os.environ["OMP_NUM_THREADS"] = "8"
    torch.set_num_threads(8)

    HIDDEN_DIM = 4096
    INTERMEDIATE_DIM = 11008
    NUM_LAYERS = 32

    layer_weights = {
        "q_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "k_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "v_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "o_proj": (HIDDEN_DIM, HIDDEN_DIM),
        "gate_proj": (INTERMEDIATE_DIM, HIDDEN_DIM),
        "up_proj": (INTERMEDIATE_DIM, HIDDEN_DIM),
        "down_proj": (HIDDEN_DIM, INTERMEDIATE_DIM),
    }

    print("Quantizing weights...")
    packed_weights = {}
    scales = {}
    for name, (out_dim, in_dim) in layer_weights.items():
        w = torch.randn(out_dim, in_dim)
        packed, scale = quantize_to_bitnet(w)
        packed_weights[name] = packed
        scales[name] = scale

    x = torch.randn(1, HIDDEN_DIM, dtype=torch.float32)

    print("\nBenchmarking Python baseline (BF16 + torch.compile)...")
    py_time = benchmark_python_baseline(packed_weights, scales, x, layer_weights)
    py_model_time = py_time * NUM_LAYERS
    py_tok_s = 1000 / py_model_time
    print(f"  Python: {py_time:.2f}ms/layer, {py_model_time:.0f}ms/model, {py_tok_s:.2f} tok/s")

    print("\nBenchmarking native C++ kernel...")
    native_time = benchmark_native(packed_weights, scales, x, layer_weights)
    if native_time:
        native_model_time = native_time * NUM_LAYERS
        native_tok_s = 1000 / native_model_time
        speedup = py_time / native_time
        print(f"  Native: {native_time:.2f}ms/layer, {native_model_time:.0f}ms/model, {native_tok_s:.2f} tok/s")
        print(f"  Speedup: {speedup:.2f}x")
    else:
        print("  Native kernel not available")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
