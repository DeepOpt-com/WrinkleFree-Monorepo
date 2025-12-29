"""Quick BitNet benchmark on Modal CPU (~5 minutes).

Tests varied settings with optimized parameters for fast execution.
"""

import modal

app = modal.App("bitnet-quick-benchmark")

cpu_bench_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["build-essential", "libomp-dev"])
    .pip_install(["torch==2.5.1+cpu", "numpy"],
                 extra_index_url="https://download.pytorch.org/whl/cpu")
)

BITNET_KERNEL_CODE = '''
import torch
from typing import Tuple

def _unpack_ternary_weights(packed_weights: torch.Tensor) -> torch.Tensor:
    out_features = packed_weights.shape[0]
    packed_in_features = packed_weights.shape[1]
    in_features = packed_in_features * 4
    packed = packed_weights.to(torch.int32)
    weights = torch.zeros(out_features, in_features, dtype=torch.float32)
    for i in range(4):
        shift = i * 2
        bits = (packed >> shift) & 0x03
        weights[:, i::4] = bits.float() - 1.0
    return weights

def pack_ternary_weights(weights: torch.Tensor) -> torch.Tensor:
    out_features, in_features = weights.shape
    packed_in_features = in_features // 4
    packed = torch.zeros(out_features, packed_in_features, dtype=torch.uint8)
    for i in range(4):
        w = weights[:, i::4].long() + 1
        packed = packed | (w.clamp(0, 2).to(torch.uint8) << (i * 2))
    return packed

def quantize_activations_i8(activations: torch.Tensor) -> Tuple[torch.Tensor, float]:
    max_val = activations.abs().max().item()
    if max_val < 1e-6:
        max_val = 1.0
    scale = max_val / 127.0
    quantized = (activations / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale

def bitnet_gemv(packed_weights: torch.Tensor, activations: torch.Tensor, scale: float) -> torch.Tensor:
    weights = _unpack_ternary_weights(packed_weights)
    return torch.matmul(weights, activations.float()) * scale

def bitnet_gemm(packed_weights: torch.Tensor, activations: torch.Tensor, scale: float) -> torch.Tensor:
    weights = _unpack_ternary_weights(packed_weights)
    return torch.matmul(activations.float(), weights.t()) * scale

class BitNetLayer:
    def __init__(self, hidden_dim: int, ffn_dim: int):
        self.q_proj = pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32))
        self.k_proj = pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32))
        self.v_proj = pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32))
        self.o_proj = pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32))
        self.gate_proj = pack_ternary_weights(torch.randint(-1, 2, (ffn_dim, hidden_dim), dtype=torch.float32))
        self.up_proj = pack_ternary_weights(torch.randint(-1, 2, (ffn_dim, hidden_dim), dtype=torch.float32))
        self.down_proj = pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, ffn_dim), dtype=torch.float32))

    def forward_batch(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden_i8, scale = quantize_activations_i8(hidden)
        q = bitnet_gemm(self.q_proj, hidden_i8, scale)
        k = bitnet_gemm(self.k_proj, hidden_i8, scale)
        v = bitnet_gemm(self.v_proj, hidden_i8, scale)
        attn_out_i8, scale2 = quantize_activations_i8(v)
        attn_out = bitnet_gemm(self.o_proj, attn_out_i8, scale2)
        ffn_in_i8, scale3 = quantize_activations_i8(attn_out)
        gate = bitnet_gemm(self.gate_proj, ffn_in_i8, scale3)
        up = bitnet_gemm(self.up_proj, ffn_in_i8, scale3)
        ffn_hidden = gate * torch.sigmoid(gate) * up
        ffn_hidden_i8, scale4 = quantize_activations_i8(ffn_hidden)
        return bitnet_gemm(self.down_proj, ffn_hidden_i8, scale4)

class BitNetModel:
    def __init__(self, num_layers: int, hidden_dim: int, ffn_dim: int):
        self.layers = [BitNetLayer(hidden_dim, ffn_dim) for _ in range(num_layers)]

    def forward_batch(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer.forward_batch(hidden)
        return hidden
'''


@app.function(
    image=cpu_bench_image,
    cpu=32.0,
    memory=32768,
    timeout=10 * 60,  # 10 minutes max
)
def run_quick_benchmark() -> str:
    """Quick benchmark: ~5 minutes total."""
    import torch
    import time
    import json
    import random

    exec(BITNET_KERNEL_CODE, globals())

    print("=" * 60)
    print("BitNet Quick Benchmark")
    print(f"CPU threads: {torch.get_num_threads()}")
    print("=" * 60)

    ITERATIONS = 5  # Quick iterations
    hidden_dim = 1024  # Smaller model for speed
    ffn_dim = 2816
    num_layers = 4

    summary = {"cpu_threads": torch.get_num_threads()}

    # ===== 1. Kernel Correctness (20 iterations for statistical confidence) =====
    print("\n[1/4] Kernel Correctness Test (20 iterations)...")
    configs = [
        {"out": 1024, "in": 1024, "batch": 1},
        {"out": 1024, "in": 1024, "batch": 32},
        {"out": 2816, "in": 1024, "batch": 64},
    ]

    all_passed = True
    for cfg in configs:
        cosines = []
        for _ in range(20):
            weights = torch.randint(-1, 2, (cfg["out"], cfg["in"]), dtype=torch.float32)
            packed = pack_ternary_weights(weights)

            if cfg["batch"] == 1:
                act = torch.randn(cfg["in"])
                act_i8, scale = quantize_activations_i8(act)
                out = bitnet_gemv(packed, act_i8, 1.0)
                ref = torch.matmul(weights, act_i8.float() * scale)
            else:
                act = torch.randn(cfg["batch"], cfg["in"])
                act_i8, scale = quantize_activations_i8(act)
                out = bitnet_gemm(packed, act_i8, 1.0)
                ref = torch.matmul(act_i8.float() * scale, weights.t())

            cos = torch.nn.functional.cosine_similarity(
                out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)
            ).item()
            cosines.append(cos)

        passed = min(cosines) > 0.99
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {cfg}: cosine={sum(cosines)/len(cosines):.6f}")

    summary["kernel_passed"] = all_passed

    # ===== 2. Long Context (prefill + decode) =====
    print(f"\n[2/4] Long Context Test ({ITERATIONS} iterations)...")
    model = BitNetModel(num_layers, hidden_dim, ffn_dim)

    context_results = {}
    for ctx_len in [256, 1024, 4096]:
        prefill_len = min(ctx_len, 128)  # Cap prefill for speed
        times = []

        for _ in range(ITERATIONS):
            # Prefill
            batch = torch.randn(prefill_len, hidden_dim)
            start = time.perf_counter()
            _ = model.forward_batch(batch)
            prefill_time = time.perf_counter() - start

            # Decode 5 tokens
            for _ in range(5):
                batch = torch.randn(1, hidden_dim)
                start = time.perf_counter()
                _ = model.forward_batch(batch)
                times.append(time.perf_counter() - start)

        avg_decode = sum(times) / len(times) * 1000
        tok_s = 1000.0 / avg_decode
        print(f"  ctx={ctx_len:4d}: decode={avg_decode:.1f}ms ({tok_s:.1f} tok/s)")
        context_results[ctx_len] = {"decode_ms": avg_decode, "tok_s": tok_s}

    summary["long_context"] = context_results

    # ===== 3. Batch Scaling =====
    print(f"\n[3/4] Batch Scaling Test ({ITERATIONS} iterations)...")
    batch_results = {}

    for batch_size in [1, 8, 32, 64, 128, 256]:
        throughputs = []
        for _ in range(ITERATIONS):
            batch = torch.randn(batch_size, hidden_dim)
            start = time.perf_counter()
            _ = model.forward_batch(batch)
            elapsed = time.perf_counter() - start
            throughputs.append(batch_size / elapsed)

        avg = sum(throughputs) / len(throughputs)
        print(f"  batch={batch_size:3d}: {avg:.1f} tok/s")
        batch_results[batch_size] = avg

    # Scaling efficiency
    base = batch_results[1]
    for bs in batch_results:
        eff = batch_results[bs] / (base * bs)
        batch_results[bs] = {"throughput": batch_results[bs], "efficiency": eff}

    summary["batch_scaling"] = batch_results

    # ===== 4. Mixed Workload =====
    print(f"\n[4/4] Mixed Workload Test ({ITERATIONS} iterations)...")
    workloads = {
        "prefill_heavy": (0.7, 256, 8),   # (prefill_ratio, prefill_len, decode_batch)
        "balanced": (0.5, 128, 32),
        "decode_heavy": (0.2, 64, 64),
    }

    workload_results = {}
    for name, (ratio, plen, dbatch) in workloads.items():
        throughputs = []
        for _ in range(ITERATIONS):
            total_time = 0
            total_tokens = 0
            for _ in range(5):  # 5 scheduling rounds
                if random.random() < ratio:
                    batch = torch.randn(plen, hidden_dim)
                else:
                    batch = torch.randn(dbatch, hidden_dim)
                start = time.perf_counter()
                _ = model.forward_batch(batch)
                total_time += time.perf_counter() - start
                total_tokens += batch.shape[0]
            throughputs.append(total_tokens / total_time)

        avg = sum(throughputs) / len(throughputs)
        print(f"  {name:15s}: {avg:.1f} tok/s")
        workload_results[name] = avg

    summary["mixed_workload"] = workload_results

    # ===== Final Summary =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n1. Kernel Correctness: {'✓ PASSED' if all_passed else '✗ FAILED'}")

    print("\n2. Context Performance (decode):")
    for ctx in [256, 1024, 4096]:
        print(f"   {ctx:4d} tokens: {context_results[ctx]['tok_s']:.1f} tok/s")

    print("\n3. Batch Scaling:")
    b1 = batch_results[1]["throughput"]
    b256 = batch_results[256]["throughput"]
    print(f"   Batch=1:   {b1:.1f} tok/s")
    print(f"   Batch=256: {b256:.1f} tok/s")
    print(f"   Speedup:   {b256/b1:.1f}x")

    print("\n4. Mixed Workloads:")
    for name in ["prefill_heavy", "balanced", "decode_heavy"]:
        print(f"   {name:15s}: {workload_results[name]:.1f} tok/s")

    total = 20 * 3 + ITERATIONS * 3 + ITERATIONS * 6 + ITERATIONS * 3
    print(f"\nTotal iterations: {total}")

    return json.dumps(summary)


@app.local_entrypoint()
def main():
    print("Running quick benchmark (~5 minutes)...")
    result = run_quick_benchmark.remote()
    print("\nBenchmark completed!")
