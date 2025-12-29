"""Thread count sweep for native kernel."""

import torch
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrinklefree_inference.sglang_backend.bitnet_quantization import quantize_to_bitnet

# Add native path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'src/wrinklefree_inference/native'))


def benchmark_threads():
    print("=" * 60)
    print(" Thread Count Sweep")
    print("=" * 60)

    HIDDEN_DIM = 4096
    INTERMEDIATE_DIM = 11008
    NUM_LAYERS = 32
    ITERATIONS = 50

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

    print("Quantizing weights...")
    packed_weights = {}
    scales = {}
    for name, (out_dim, in_dim) in layer_weights.items():
        w = torch.randn(out_dim, in_dim)
        packed, scale = quantize_to_bitnet(w)
        packed_weights[name] = packed
        scales[name] = scale

    x = torch.randn(1, HIDDEN_DIM, dtype=torch.float32)

    import bitnet_native

    def forward():
        x_f32 = x.float().contiguous()
        q = bitnet_native.gemv(packed_weights["q_proj"], x_f32, scales["q_proj"])
        k = bitnet_native.gemv(packed_weights["k_proj"], x_f32, scales["k_proj"])
        v = bitnet_native.gemv(packed_weights["v_proj"], x_f32, scales["v_proj"])
        o = bitnet_native.gemv(packed_weights["o_proj"], x_f32, scales["o_proj"])
        gate = bitnet_native.gemv(packed_weights["gate_proj"], x_f32, scales["gate_proj"])
        up = bitnet_native.gemv(packed_weights["up_proj"], x_f32, scales["up_proj"])
        hidden = gate * up
        down = bitnet_native.gemv(packed_weights["down_proj"], hidden.contiguous(), scales["down_proj"])
        return down

    print("\nThreads | ms/layer | tok/s")
    print("-" * 35)

    for num_threads in [1, 2, 4, 8, 12, 16]:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        torch.set_num_threads(num_threads)

        # Warmup
        for _ in range(5):
            forward()

        # Benchmark
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            forward()
        elapsed = time.perf_counter() - start

        ms_per_layer = elapsed / ITERATIONS * 1000
        ms_per_model = ms_per_layer * NUM_LAYERS
        tok_s = 1000 / ms_per_model

        print(f"   {num_threads:2d}   |  {ms_per_layer:.2f}   | {tok_s:.2f}")


if __name__ == "__main__":
    benchmark_threads()
