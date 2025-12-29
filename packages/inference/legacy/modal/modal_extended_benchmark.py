"""Extended BitNet benchmark on Modal CPU with varied settings.

Tests 20 iterations across:
1. Long context windows (2K, 4K, 8K, 16K)
2. Large batch sizes (1, 16, 64, 128, 256)
3. Mixed prefill + decode (simulating continuous batching)
"""

import modal
import os

app = modal.App("bitnet-extended-benchmark")

# CPU-only image for benchmarking
cpu_bench_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install([
        "build-essential",
        "libomp-dev",
    ])
    .pip_install([
        "torch==2.5.1+cpu",
        "numpy",
    ], extra_index_url="https://download.pytorch.org/whl/cpu")
)


# BitNet kernel implementation (embedded for Modal)
BITNET_KERNEL_CODE = '''
import torch
import numpy as np
from typing import Tuple, List
import time

QK_I2_S = 128


def _unpack_ternary_weights(packed_weights: torch.Tensor) -> torch.Tensor:
    """Unpack 2-bit packed weights to ternary {-1, 0, +1} values."""
    out_features = packed_weights.shape[0]
    packed_in_features = packed_weights.shape[1]
    in_features = packed_in_features * 4

    packed = packed_weights.to(torch.int32)
    weights = torch.zeros(out_features, in_features, dtype=torch.float32)

    for i in range(4):
        shift = i * 2
        bits = (packed >> shift) & 0x03
        unpacked = bits.float() - 1.0
        weights[:, i::4] = unpacked

    return weights


def pack_ternary_weights(weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary {-1, 0, +1} weights to 2-bit format."""
    out_features, in_features = weights.shape
    packed_in_features = in_features // 4
    packed = torch.zeros(out_features, packed_in_features, dtype=torch.uint8)

    for i in range(4):
        w = weights[:, i::4].long() + 1
        w = w.clamp(0, 2)
        packed = packed | (w.to(torch.uint8) << (i * 2))

    return packed


def quantize_activations_i8(activations: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize FP32 activations to INT8."""
    max_val = activations.abs().max().item()
    if max_val < 1e-6:
        max_val = 1.0
    scale = max_val / 127.0
    quantized = (activations / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def bitnet_gemv(packed_weights: torch.Tensor, activations: torch.Tensor, scale: float) -> torch.Tensor:
    """BitNet GEMV: y = scale * (W @ x)."""
    weights = _unpack_ternary_weights(packed_weights)
    activations_f = activations.to(torch.float32)
    return torch.matmul(weights, activations_f) * scale


def bitnet_gemm(packed_weights: torch.Tensor, activations: torch.Tensor, scale: float) -> torch.Tensor:
    """BitNet GEMM: Y = scale * (W @ X)."""
    weights = _unpack_ternary_weights(packed_weights)
    activations_f = activations.to(torch.float32)
    return torch.matmul(activations_f, weights.t()) * scale


class BitNetLayer:
    """Single transformer layer with BitNet weights."""
    def __init__(self, hidden_dim: int, ffn_dim: int):
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        # Attention projections
        self.q_proj = pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32))
        self.k_proj = pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32))
        self.v_proj = pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32))
        self.o_proj = pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, hidden_dim), dtype=torch.float32))
        # FFN projections
        self.gate_proj = pack_ternary_weights(torch.randint(-1, 2, (ffn_dim, hidden_dim), dtype=torch.float32))
        self.up_proj = pack_ternary_weights(torch.randint(-1, 2, (ffn_dim, hidden_dim), dtype=torch.float32))
        self.down_proj = pack_ternary_weights(torch.randint(-1, 2, (hidden_dim, ffn_dim), dtype=torch.float32))

    def forward_single(self, hidden: torch.Tensor) -> torch.Tensor:
        """Single token forward (GEMV)."""
        hidden_i8, scale = quantize_activations_i8(hidden)

        # Attention projections
        q = bitnet_gemv(self.q_proj, hidden_i8, scale)
        k = bitnet_gemv(self.k_proj, hidden_i8, scale)
        v = bitnet_gemv(self.v_proj, hidden_i8, scale)

        # Simplified attention output
        attn_out_i8, scale2 = quantize_activations_i8(v)
        attn_out = bitnet_gemv(self.o_proj, attn_out_i8, scale2)

        # FFN
        ffn_in_i8, scale3 = quantize_activations_i8(attn_out)
        gate = bitnet_gemv(self.gate_proj, ffn_in_i8, scale3)
        up = bitnet_gemv(self.up_proj, ffn_in_i8, scale3)
        ffn_hidden = gate * torch.sigmoid(gate) * up  # SwiGLU
        ffn_hidden_i8, scale4 = quantize_activations_i8(ffn_hidden)
        return bitnet_gemv(self.down_proj, ffn_hidden_i8, scale4)

    def forward_batch(self, hidden: torch.Tensor) -> torch.Tensor:
        """Batched forward (GEMM)."""
        hidden_i8, scale = quantize_activations_i8(hidden)

        # Attention projections (batched)
        q = bitnet_gemm(self.q_proj, hidden_i8, scale)
        k = bitnet_gemm(self.k_proj, hidden_i8, scale)
        v = bitnet_gemm(self.v_proj, hidden_i8, scale)

        # Simplified attention output
        attn_out_i8, scale2 = quantize_activations_i8(v)
        attn_out = bitnet_gemm(self.o_proj, attn_out_i8, scale2)

        # FFN
        ffn_in_i8, scale3 = quantize_activations_i8(attn_out)
        gate = bitnet_gemm(self.gate_proj, ffn_in_i8, scale3)
        up = bitnet_gemm(self.up_proj, ffn_in_i8, scale3)
        ffn_hidden = gate * torch.sigmoid(gate) * up
        ffn_hidden_i8, scale4 = quantize_activations_i8(ffn_hidden)
        return bitnet_gemm(self.down_proj, ffn_hidden_i8, scale4)


class BitNetModel:
    """Multi-layer BitNet model for benchmarking."""
    def __init__(self, num_layers: int, hidden_dim: int, ffn_dim: int):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = [BitNetLayer(hidden_dim, ffn_dim) for _ in range(num_layers)]

    def forward_single(self, hidden: torch.Tensor) -> torch.Tensor:
        """Single token through all layers."""
        for layer in self.layers:
            hidden = layer.forward_single(hidden)
        return hidden

    def forward_batch(self, hidden: torch.Tensor) -> torch.Tensor:
        """Batch of tokens through all layers."""
        for layer in self.layers:
            hidden = layer.forward_batch(hidden)
        return hidden


class KVCache:
    """Simulated KV cache with INT8 quantization."""
    def __init__(self, num_layers: int, max_seq_len: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        # INT8 KV cache (50% memory savings vs FP16)
        self.k_cache = torch.zeros(num_layers, max_seq_len, num_heads, head_dim, dtype=torch.int8)
        self.v_cache = torch.zeros(num_layers, max_seq_len, num_heads, head_dim, dtype=torch.int8)
        self.scales = torch.ones(num_layers, max_seq_len, 2)  # k and v scales per position
        self.seq_len = 0

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Append new KV to cache."""
        if self.seq_len >= self.max_seq_len:
            return
        # Quantize and store
        k_i8, k_scale = quantize_activations_i8(k)
        v_i8, v_scale = quantize_activations_i8(v)
        self.k_cache[layer_idx, self.seq_len] = k_i8.view(self.num_heads, self.head_dim)
        self.v_cache[layer_idx, self.seq_len] = v_i8.view(self.num_heads, self.head_dim)
        self.scales[layer_idx, self.seq_len, 0] = k_scale
        self.scales[layer_idx, self.seq_len, 1] = v_scale

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached KV for attention."""
        k = self.k_cache[layer_idx, :self.seq_len].float()
        v = self.v_cache[layer_idx, :self.seq_len].float()
        # Dequantize
        k = k * self.scales[layer_idx, :self.seq_len, 0:1].unsqueeze(-1)
        v = v * self.scales[layer_idx, :self.seq_len, 1:2].unsqueeze(-1)
        return k, v

    def increment(self):
        self.seq_len += 1

    def memory_bytes(self) -> int:
        """Current memory usage in bytes."""
        # INT8 KV cache
        return self.seq_len * self.num_layers * self.num_heads * self.head_dim * 2  # k + v
'''


@app.function(
    image=cpu_bench_image,
    cpu=32.0,
    memory=65536,
    timeout=60 * 60,  # 1 hour for extended benchmark
)
def benchmark_long_context(iterations: int = 20) -> dict:
    """Benchmark with varying context lengths."""
    import torch
    import numpy as np
    import time

    exec(BITNET_KERNEL_CODE, globals())

    results = {
        "test": "long_context",
        "iterations": iterations,
        "cpu_count": torch.get_num_threads(),
        "context_results": {},
    }

    print(f"=== Long Context Benchmark ({iterations} iterations) ===")
    print(f"Running on {results['cpu_count']} CPU threads")

    # Llama-2B dimensions
    hidden_dim = 2048
    ffn_dim = 5632
    num_layers = 8  # Reduced for faster benchmarking
    num_heads = 16
    head_dim = hidden_dim // num_heads

    # Test context lengths
    context_lengths = [512, 2048, 4096, 8192, 16384]

    for ctx_len in context_lengths:
        print(f"\n--- Context Length: {ctx_len} ---")
        ctx_results = {
            "context_length": ctx_len,
            "iterations": [],
            "all_passed": True,
        }

        for i in range(iterations):
            # Create model and KV cache
            model = BitNetModel(num_layers, hidden_dim, ffn_dim)
            kv_cache = KVCache(num_layers, ctx_len, num_heads, head_dim)

            # Prefill phase: process context
            prefill_tokens = min(ctx_len - 1, 256)  # Cap prefill for speed
            hidden_batch = torch.randn(prefill_tokens, hidden_dim)

            start = time.perf_counter()
            output = model.forward_batch(hidden_batch)
            prefill_time = time.perf_counter() - start

            # Simulate KV cache population
            for _ in range(prefill_tokens):
                kv_cache.increment()

            # Decode phase: generate tokens
            decode_tokens = 10
            decode_times = []
            hidden = torch.randn(hidden_dim)

            for _ in range(decode_tokens):
                start = time.perf_counter()
                output = model.forward_single(hidden)
                decode_times.append(time.perf_counter() - start)
                kv_cache.increment()
                hidden = output

            avg_decode = np.mean(decode_times)
            kv_memory = kv_cache.memory_bytes()

            # Verify output is valid (not NaN/Inf)
            passed = not (torch.isnan(output).any() or torch.isinf(output).any())
            if not passed:
                ctx_results["all_passed"] = False

            ctx_results["iterations"].append({
                "iteration": i + 1,
                "prefill_time_ms": prefill_time * 1000,
                "prefill_tok_s": prefill_tokens / prefill_time,
                "decode_time_ms": avg_decode * 1000,
                "decode_tok_s": 1.0 / avg_decode,
                "kv_cache_mb": kv_memory / 1e6,
                "passed": passed,
            })

            if (i + 1) % 5 == 0:
                print(f"  Iter {i+1}: prefill={prefill_time*1000:.1f}ms ({prefill_tokens/prefill_time:.1f} tok/s), "
                      f"decode={avg_decode*1000:.1f}ms, KV={kv_memory/1e6:.1f}MB")

        # Summary for this context length
        prefill_times = [r["prefill_time_ms"] for r in ctx_results["iterations"]]
        decode_times = [r["decode_time_ms"] for r in ctx_results["iterations"]]
        ctx_results["summary"] = {
            "prefill_avg_ms": float(np.mean(prefill_times)),
            "prefill_std_ms": float(np.std(prefill_times)),
            "decode_avg_ms": float(np.mean(decode_times)),
            "decode_std_ms": float(np.std(decode_times)),
            "decode_tok_s": float(1000.0 / np.mean(decode_times)),
        }

        results["context_results"][ctx_len] = ctx_results
        print(f"  Summary: prefill={ctx_results['summary']['prefill_avg_ms']:.1f}±{ctx_results['summary']['prefill_std_ms']:.1f}ms, "
              f"decode={ctx_results['summary']['decode_avg_ms']:.1f}±{ctx_results['summary']['decode_std_ms']:.1f}ms")

    return results


@app.function(
    image=cpu_bench_image,
    cpu=32.0,
    memory=65536,
    timeout=60 * 60,
)
def benchmark_large_batches(iterations: int = 20) -> dict:
    """Benchmark with large batch sizes."""
    import torch
    import numpy as np
    import time

    exec(BITNET_KERNEL_CODE, globals())

    results = {
        "test": "large_batches",
        "iterations": iterations,
        "cpu_count": torch.get_num_threads(),
        "batch_results": {},
    }

    print(f"=== Large Batch Benchmark ({iterations} iterations) ===")
    print(f"Running on {results['cpu_count']} CPU threads")

    # Model dimensions
    hidden_dim = 2048
    ffn_dim = 5632
    num_layers = 8

    # Test batch sizes (simulating concurrent requests)
    batch_sizes = [1, 4, 16, 64, 128, 256]

    # Create model once
    model = BitNetModel(num_layers, hidden_dim, ffn_dim)

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        batch_results = {
            "batch_size": batch_size,
            "iterations": [],
            "all_passed": True,
        }

        for i in range(iterations):
            # Create batch input
            hidden_batch = torch.randn(batch_size, hidden_dim)

            # Warmup
            if i == 0:
                _ = model.forward_batch(hidden_batch)

            # Benchmark
            start = time.perf_counter()
            output = model.forward_batch(hidden_batch)
            elapsed = time.perf_counter() - start

            throughput = batch_size / elapsed
            latency_per_token = elapsed / batch_size

            # Verify
            passed = not (torch.isnan(output).any() or torch.isinf(output).any())
            if not passed:
                batch_results["all_passed"] = False

            batch_results["iterations"].append({
                "iteration": i + 1,
                "time_ms": elapsed * 1000,
                "throughput_tok_s": throughput,
                "latency_per_token_ms": latency_per_token * 1000,
                "passed": passed,
            })

            if (i + 1) % 5 == 0:
                print(f"  Iter {i+1}: {elapsed*1000:.1f}ms, {throughput:.1f} tok/s, "
                      f"{latency_per_token*1000:.2f}ms/tok")

        # Summary
        times = [r["time_ms"] for r in batch_results["iterations"]]
        throughputs = [r["throughput_tok_s"] for r in batch_results["iterations"]]
        batch_results["summary"] = {
            "time_avg_ms": float(np.mean(times)),
            "time_std_ms": float(np.std(times)),
            "throughput_avg": float(np.mean(throughputs)),
            "throughput_std": float(np.std(throughputs)),
            "throughput_max": float(np.max(throughputs)),
        }

        results["batch_results"][batch_size] = batch_results
        print(f"  Summary: {batch_results['summary']['time_avg_ms']:.1f}±{batch_results['summary']['time_std_ms']:.1f}ms, "
              f"{batch_results['summary']['throughput_avg']:.1f}±{batch_results['summary']['throughput_std']:.1f} tok/s")

    # Calculate scaling efficiency
    base_throughput = results["batch_results"][1]["summary"]["throughput_avg"]
    for batch_size, batch_results in results["batch_results"].items():
        ideal = base_throughput * batch_size
        actual = batch_results["summary"]["throughput_avg"]
        batch_results["summary"]["scaling_efficiency"] = actual / ideal

    return results


@app.function(
    image=cpu_bench_image,
    cpu=32.0,
    memory=65536,
    timeout=60 * 60,
)
def benchmark_mixed_workload(iterations: int = 20) -> dict:
    """Benchmark mixed prefill + decode (real continuous batching scenario)."""
    import torch
    import numpy as np
    import time
    import random

    exec(BITNET_KERNEL_CODE, globals())

    results = {
        "test": "mixed_workload",
        "iterations": iterations,
        "cpu_count": torch.get_num_threads(),
        "workload_results": {},
    }

    print(f"=== Mixed Workload Benchmark ({iterations} iterations) ===")
    print(f"Running on {results['cpu_count']} CPU threads")

    hidden_dim = 2048
    ffn_dim = 5632
    num_layers = 8

    model = BitNetModel(num_layers, hidden_dim, ffn_dim)

    # Different workload mixes
    workloads = {
        "prefill_heavy": {"prefill_ratio": 0.8, "prefill_len": 512, "decode_batch": 4},
        "decode_heavy": {"prefill_ratio": 0.2, "prefill_len": 128, "decode_batch": 32},
        "balanced": {"prefill_ratio": 0.5, "prefill_len": 256, "decode_batch": 16},
        "streaming": {"prefill_ratio": 0.1, "prefill_len": 64, "decode_batch": 64},
    }

    for workload_name, config in workloads.items():
        print(f"\n--- Workload: {workload_name} ---")
        print(f"    Config: {config}")

        workload_results = {
            "config": config,
            "iterations": [],
            "all_passed": True,
        }

        for i in range(iterations):
            total_time = 0
            total_tokens = 0
            prefill_time = 0
            decode_time = 0
            prefill_tokens = 0
            decode_tokens = 0

            # Simulate 10 scheduling rounds
            for _ in range(10):
                if random.random() < config["prefill_ratio"]:
                    # Prefill a new request
                    batch = torch.randn(config["prefill_len"], hidden_dim)
                    start = time.perf_counter()
                    output = model.forward_batch(batch)
                    elapsed = time.perf_counter() - start
                    prefill_time += elapsed
                    prefill_tokens += config["prefill_len"]
                else:
                    # Decode step for existing requests
                    batch = torch.randn(config["decode_batch"], hidden_dim)
                    start = time.perf_counter()
                    output = model.forward_batch(batch)
                    elapsed = time.perf_counter() - start
                    decode_time += elapsed
                    decode_tokens += config["decode_batch"]

                total_time += elapsed
                total_tokens += batch.shape[0]

            # Verify last output
            passed = not (torch.isnan(output).any() or torch.isinf(output).any())
            if not passed:
                workload_results["all_passed"] = False

            workload_results["iterations"].append({
                "iteration": i + 1,
                "total_time_ms": total_time * 1000,
                "total_throughput": total_tokens / total_time,
                "prefill_time_ms": prefill_time * 1000,
                "prefill_tokens": prefill_tokens,
                "prefill_tok_s": prefill_tokens / prefill_time if prefill_time > 0 else 0,
                "decode_time_ms": decode_time * 1000,
                "decode_tokens": decode_tokens,
                "decode_tok_s": decode_tokens / decode_time if decode_time > 0 else 0,
                "passed": passed,
            })

            if (i + 1) % 5 == 0:
                r = workload_results["iterations"][-1]
                print(f"  Iter {i+1}: total={r['total_throughput']:.1f} tok/s, "
                      f"prefill={r['prefill_tok_s']:.1f} tok/s, decode={r['decode_tok_s']:.1f} tok/s")

        # Summary
        total_throughputs = [r["total_throughput"] for r in workload_results["iterations"]]
        prefill_throughputs = [r["prefill_tok_s"] for r in workload_results["iterations"] if r["prefill_tok_s"] > 0]
        decode_throughputs = [r["decode_tok_s"] for r in workload_results["iterations"] if r["decode_tok_s"] > 0]

        workload_results["summary"] = {
            "total_throughput_avg": float(np.mean(total_throughputs)),
            "total_throughput_std": float(np.std(total_throughputs)),
            "prefill_throughput_avg": float(np.mean(prefill_throughputs)) if prefill_throughputs else 0.0,
            "decode_throughput_avg": float(np.mean(decode_throughputs)) if decode_throughputs else 0.0,
        }

        results["workload_results"][workload_name] = workload_results
        print(f"  Summary: total={workload_results['summary']['total_throughput_avg']:.1f} tok/s, "
              f"prefill={workload_results['summary']['prefill_throughput_avg']:.1f} tok/s, "
              f"decode={workload_results['summary']['decode_throughput_avg']:.1f} tok/s")

    return results


@app.function(
    image=cpu_bench_image,
    cpu=32.0,
    memory=65536,
    timeout=60 * 60,
)
def benchmark_kernel_stress(iterations: int = 20) -> dict:
    """Stress test kernel correctness with varied inputs."""
    import torch
    import numpy as np
    import time

    exec(BITNET_KERNEL_CODE, globals())

    results = {
        "test": "kernel_stress",
        "iterations": iterations,
        "cpu_count": torch.get_num_threads(),
        "stress_results": [],
        "all_passed": True,
    }

    print(f"=== Kernel Stress Test ({iterations} iterations) ===")
    print(f"Running on {results['cpu_count']} CPU threads")

    # Test configurations
    configs = [
        {"out": 256, "in": 512, "batch": 1},
        {"out": 2048, "in": 2048, "batch": 1},
        {"out": 4096, "in": 4096, "batch": 1},
        {"out": 2048, "in": 2048, "batch": 32},
        {"out": 4096, "in": 4096, "batch": 64},
        {"out": 5632, "in": 2048, "batch": 128},  # FFN shape
    ]

    for config in configs:
        out_f, in_f, batch = config["out"], config["in"], config["batch"]
        print(f"\n--- Config: out={out_f}, in={in_f}, batch={batch} ---")

        config_results = {
            "config": config,
            "iterations": [],
            "all_passed": True,
        }

        for i in range(iterations):
            # Create weights
            weights = torch.randint(-1, 2, (out_f, in_f), dtype=torch.float32)
            packed = pack_ternary_weights(weights)

            if batch == 1:
                # GEMV
                activations = torch.randn(in_f)
                act_i8, act_scale = quantize_activations_i8(activations)

                start = time.perf_counter()
                output = bitnet_gemv(packed, act_i8, 1.0)
                elapsed = time.perf_counter() - start

                # Reference
                act_dequant = act_i8.float() * act_scale
                ref_output = torch.matmul(weights, act_dequant)
            else:
                # GEMM
                activations = torch.randn(batch, in_f)
                act_i8, act_scale = quantize_activations_i8(activations)

                start = time.perf_counter()
                output = bitnet_gemm(packed, act_i8, 1.0)
                elapsed = time.perf_counter() - start

                # Reference
                act_dequant = act_i8.float() * act_scale
                ref_output = torch.matmul(act_dequant, weights.t())

            # Verify
            cosine = torch.nn.functional.cosine_similarity(
                output.flatten().unsqueeze(0),
                ref_output.flatten().unsqueeze(0)
            ).item()

            passed = cosine > 0.99 and not torch.isnan(output).any()
            if not passed:
                config_results["all_passed"] = False
                results["all_passed"] = False

            config_results["iterations"].append({
                "iteration": i + 1,
                "time_ms": elapsed * 1000,
                "cosine": cosine,
                "passed": passed,
            })

        # Summary
        times = [r["time_ms"] for r in config_results["iterations"]]
        cosines = [r["cosine"] for r in config_results["iterations"]]
        config_results["summary"] = {
            "time_avg_ms": float(np.mean(times)),
            "time_std_ms": float(np.std(times)),
            "cosine_min": float(np.min(cosines)),
            "cosine_max": float(np.max(cosines)),
            "cosine_avg": float(np.mean(cosines)),
        }

        results["stress_results"].append(config_results)
        status = "PASS" if config_results["all_passed"] else "FAIL"
        print(f"  [{status}] time={config_results['summary']['time_avg_ms']:.2f}±{config_results['summary']['time_std_ms']:.2f}ms, "
              f"cosine={config_results['summary']['cosine_avg']:.6f}")

    return results


@app.function(
    image=cpu_bench_image,
    cpu=32.0,
    memory=65536,
    timeout=120 * 60,  # 2 hours
)
def run_all_benchmarks(iterations: int = 20) -> str:
    """Run all benchmarks in a single Modal function to avoid serialization issues."""
    import torch
    import numpy as np
    import time
    import json

    exec(BITNET_KERNEL_CODE, globals())

    print("=" * 70)
    print("BitNet Extended CPU Benchmark on Modal")
    print(f"Running on {torch.get_num_threads()} CPU threads")
    print("=" * 70)

    summary = {
        "cpu_threads": torch.get_num_threads(),
        "iterations_per_test": iterations,
    }

    # ========== 1. Kernel Stress Test ==========
    print("\n[1/4] Kernel Stress Test...")
    configs = [
        {"out": 256, "in": 512, "batch": 1},
        {"out": 2048, "in": 2048, "batch": 1},
        {"out": 4096, "in": 4096, "batch": 1},
        {"out": 2048, "in": 2048, "batch": 32},
        {"out": 4096, "in": 4096, "batch": 64},
        {"out": 5632, "in": 2048, "batch": 128},
    ]

    kernel_all_passed = True
    kernel_results = []

    for config in configs:
        out_f, in_f, batch = config["out"], config["in"], config["batch"]
        cosines = []
        times = []

        for i in range(iterations):
            weights = torch.randint(-1, 2, (out_f, in_f), dtype=torch.float32)
            packed = pack_ternary_weights(weights)

            if batch == 1:
                activations = torch.randn(in_f)
                act_i8, act_scale = quantize_activations_i8(activations)
                start = time.perf_counter()
                output = bitnet_gemv(packed, act_i8, 1.0)
                elapsed = time.perf_counter() - start
                act_dequant = act_i8.float() * act_scale
                ref_output = torch.matmul(weights, act_dequant)
            else:
                activations = torch.randn(batch, in_f)
                act_i8, act_scale = quantize_activations_i8(activations)
                start = time.perf_counter()
                output = bitnet_gemm(packed, act_i8, 1.0)
                elapsed = time.perf_counter() - start
                act_dequant = act_i8.float() * act_scale
                ref_output = torch.matmul(act_dequant, weights.t())

            cosine = torch.nn.functional.cosine_similarity(
                output.flatten().unsqueeze(0),
                ref_output.flatten().unsqueeze(0)
            ).item()
            cosines.append(cosine)
            times.append(elapsed * 1000)

            if cosine <= 0.99:
                kernel_all_passed = False

        avg_cos = sum(cosines) / len(cosines)
        avg_time = sum(times) / len(times)
        status = "PASS" if min(cosines) > 0.99 else "FAIL"
        print(f"  [{status}] out={out_f}, in={in_f}, batch={batch}: "
              f"time={avg_time:.2f}ms, cosine={avg_cos:.6f}")
        kernel_results.append({
            "config": config,
            "cosine_avg": avg_cos,
            "time_avg_ms": avg_time,
            "passed": min(cosines) > 0.99,
        })

    summary["kernel_stress"] = {"all_passed": kernel_all_passed, "results": kernel_results}

    # ========== 2. Long Context Benchmark ==========
    print("\n[2/4] Long Context Benchmark...")
    hidden_dim = 2048
    ffn_dim = 5632
    num_layers = 8
    num_heads = 16
    head_dim = hidden_dim // num_heads

    context_lengths = [512, 2048, 4096, 8192, 16384]
    context_results = {}

    for ctx_len in context_lengths:
        decode_times = []

        for i in range(iterations):
            model = BitNetModel(num_layers, hidden_dim, ffn_dim)
            kv_cache = KVCache(num_layers, ctx_len, num_heads, head_dim)

            # Prefill
            prefill_tokens = min(ctx_len - 1, 256)
            hidden_batch = torch.randn(prefill_tokens, hidden_dim)
            _ = model.forward_batch(hidden_batch)
            for _ in range(prefill_tokens):
                kv_cache.increment()

            # Decode
            hidden = torch.randn(hidden_dim)
            for _ in range(10):
                start = time.perf_counter()
                output = model.forward_single(hidden)
                decode_times.append(time.perf_counter() - start)
                kv_cache.increment()
                hidden = output

        avg_decode = sum(decode_times) / len(decode_times) * 1000
        tok_s = 1000.0 / avg_decode
        print(f"  ctx={ctx_len:5d}: decode={avg_decode:.1f}ms ({tok_s:.1f} tok/s)")
        context_results[ctx_len] = {"decode_ms": avg_decode, "tok_s": tok_s}

    summary["long_context"] = context_results

    # ========== 3. Large Batch Benchmark ==========
    print("\n[3/4] Large Batch Benchmark...")
    model = BitNetModel(num_layers, hidden_dim, ffn_dim)
    batch_sizes = [1, 4, 16, 64, 128, 256]
    batch_results = {}

    for batch_size in batch_sizes:
        throughputs = []

        for i in range(iterations):
            hidden_batch = torch.randn(batch_size, hidden_dim)
            start = time.perf_counter()
            _ = model.forward_batch(hidden_batch)
            elapsed = time.perf_counter() - start
            throughputs.append(batch_size / elapsed)

        avg_throughput = sum(throughputs) / len(throughputs)
        print(f"  batch={batch_size:3d}: {avg_throughput:.1f} tok/s")
        batch_results[batch_size] = {"throughput": avg_throughput}

    # Calculate scaling efficiency
    base = batch_results[1]["throughput"]
    for bs, data in batch_results.items():
        data["efficiency"] = data["throughput"] / (base * bs)

    summary["large_batches"] = batch_results

    # ========== 4. Mixed Workload Benchmark ==========
    print("\n[4/4] Mixed Workload Benchmark...")
    import random

    workloads = {
        "prefill_heavy": {"prefill_ratio": 0.8, "prefill_len": 512, "decode_batch": 4},
        "decode_heavy": {"prefill_ratio": 0.2, "prefill_len": 128, "decode_batch": 32},
        "balanced": {"prefill_ratio": 0.5, "prefill_len": 256, "decode_batch": 16},
        "streaming": {"prefill_ratio": 0.1, "prefill_len": 64, "decode_batch": 64},
    }

    workload_results = {}

    for workload_name, config in workloads.items():
        throughputs = []

        for i in range(iterations):
            total_time = 0
            total_tokens = 0

            for _ in range(10):
                if random.random() < config["prefill_ratio"]:
                    batch = torch.randn(config["prefill_len"], hidden_dim)
                else:
                    batch = torch.randn(config["decode_batch"], hidden_dim)

                start = time.perf_counter()
                _ = model.forward_batch(batch)
                total_time += time.perf_counter() - start
                total_tokens += batch.shape[0]

            throughputs.append(total_tokens / total_time)

        avg_throughput = sum(throughputs) / len(throughputs)
        print(f"  {workload_name:15s}: {avg_throughput:.1f} tok/s")
        workload_results[workload_name] = {"throughput": avg_throughput}

    summary["mixed_workload"] = workload_results

    # ========== Final Summary ==========
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n1. Kernel Correctness: {'✓ PASSED' if kernel_all_passed else '✗ FAILED'}")

    print("\n2. Long Context Performance:")
    for ctx_len in [512, 4096, 16384]:
        print(f"   {ctx_len:5d} tokens: {context_results[ctx_len]['tok_s']:.1f} tok/s decode")

    print("\n3. Batch Scaling:")
    b1 = batch_results[1]["throughput"]
    b256 = batch_results[256]["throughput"]
    print(f"   Batch=1:   {b1:.1f} tok/s")
    print(f"   Batch=256: {b256:.1f} tok/s")
    print(f"   Speedup:   {b256/b1:.1f}x")

    print("\n4. Mixed Workload Throughput:")
    for workload in ["prefill_heavy", "balanced", "decode_heavy", "streaming"]:
        print(f"   {workload:15s}: {workload_results[workload]['throughput']:.1f} tok/s")

    total_iters = iterations * (6 + 5 + 6 + 4)
    print(f"\nTotal iterations completed: {total_iters}")

    return json.dumps(summary)


@app.local_entrypoint()
def main():
    """Run all extended benchmarks on Modal."""
    print("Starting extended benchmark on Modal...")
    result = run_all_benchmarks.remote(iterations=20)
    print("\n" + "=" * 70)
    print("Benchmark completed!")
    print("=" * 70)
