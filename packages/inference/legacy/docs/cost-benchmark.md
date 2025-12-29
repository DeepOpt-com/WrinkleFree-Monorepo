# Cost Benchmarking for BitNet 1.58-bit Inference

This document describes the cost benchmarking methodology for 1.58-bit inference using BitNet.cpp on cloud hardware.

## Overview

The cost benchmarking suite measures inference performance and calculates cost efficiency for:
- **Native BitNet models**: Pre-trained with 1.58-bit weights (highest quality)
- **Naive ternary conversion**: Existing models converted to ternary weights (lower quality, for cost analysis)

## Key Concepts

### BitNet 1.58-bit Quantization

BitNet uses ternary weights (-1, 0, +1) with 1.58 bits per weight on average. This enables:
- ~6x memory reduction vs FP16
- Optimized matrix multiplication (additions only, no multiplications)
- CPU-efficient inference via AVX512 instructions

### Native vs Naive Conversion

| Approach | Quality | Use Case |
|----------|---------|----------|
| **Native BitNet** | High | Production inference, quality benchmarks |
| **Naive Ternary** | Low | Cost/speed analysis, throughput benchmarks |

**Native BitNet** models are trained from scratch with ternary weights. Examples:
- `microsoft/BitNet-b1.58-2B-4T`
- `HF1BitLLM/Llama3-8B-1.58-100B-tokens`

**Naive ternary conversion** rounds existing FP16 weights to -1/0/+1:
```
scale = mean(|weights|)
ternary = clamp(round(weights / scale), -1, 1)
```

This produces poor quality outputs but enables cost benchmarking of larger model architectures.

## Hardware Configurations

### RunPod Pricing (as of December 2024)

| Hardware | Spot $/hr | On-Demand $/hr | Memory |
|----------|-----------|----------------|--------|
| NVIDIA A40 | $0.39 | $0.69 | 48GB VRAM |
| NVIDIA L4 | $0.24 | $0.36 | 24GB VRAM |
| CPU 32-core | ~$0.12 | ~$0.15 | 128GB RAM |
| CPU 64-core | ~$0.24 | ~$0.30 | 256GB RAM |

BitNet.cpp is CPU-optimized with AVX512, making high-core-count CPUs competitive with GPUs for 1.58-bit inference.

## Running Benchmarks

### Prerequisites

```bash
# Install benchmark dependencies
uv sync --extra benchmark

# For naive conversion
uv sync --extra convert
```

### Local Benchmarking

1. Start the inference server:
```bash
uv run wrinklefree-inference serve -m models/bitnet-2b.gguf
```

2. Run the benchmark:
```bash
uv run wrinklefree-inference benchmark-cost \
    --url http://localhost:8080 \
    --hardware a40 \
    --model bitnet-2b-4t
```

### Cloud Benchmarking with SkyPilot

```bash
# A40 GPU benchmark
sky launch skypilot/benchmark/runpod_a40_benchmark.yaml -y

# 64-core CPU benchmark
sky launch skypilot/benchmark/runpod_cpu_64core.yaml -y

# 32-core CPU benchmark
sky launch skypilot/benchmark/runpod_cpu_32core.yaml -y
```

### Naive Ternary Conversion

Convert existing models for benchmarking (produces low quality outputs):

```bash
# Estimate memory requirements
uv run wrinklefree-inference naive-convert \
    --model-id meta-llama/Llama-3.1-70B \
    --estimate-only

# Full conversion (requires GPU with sufficient memory)
uv run wrinklefree-inference naive-convert \
    --model-id meta-llama/Llama-3.1-70B \
    --output-dir models/naive
```

For large models, use the SkyPilot conversion job:
```bash
sky launch skypilot/benchmark/naive_convert_a40.yaml -y
```

## Metrics

### Performance Metrics

| Metric | Description |
|--------|-------------|
| **Tokens/sec** | Generation throughput |
| **TTFT (ms)** | Time to first token |
| **Latency P50/P99 (ms)** | Request latency percentiles |
| **Memory (GB)** | Peak memory usage |

### Cost Metrics

| Metric | Description |
|--------|-------------|
| **$/hr** | Hardware cost per hour |
| **$/1M tokens** | Cost per million generated tokens |
| **$/1M @ 70%** | Cost at 70% utilization |
| **$/1M @ 50%** | Cost at 50% utilization |

**Cost calculation**:
```
cost_per_million = (hourly_cost / tokens_per_hour) * 1,000,000
```

**Utilization adjustment** accounts for real-world scenarios where hardware isn't 100% utilized:
```
cost_at_utilization = cost_per_million / utilization
```

## Results Structure

Results are stored in `results/raw/` as JSON:

```json
{
  "run_id": "2024-12-22-bitnet2b-native-a40",
  "model": "bitnet-2b-4t",
  "quantization": "native",
  "hardware": "a40",
  "hardware_cost_per_hour": 0.39,
  "tokens_per_second": 150.5,
  "ttft_p50_ms": 45,
  "latency_p99_ms": 320,
  "memory_usage_gb": 1.2,
  "cost_per_million_tokens": 0.72,
  "cost_per_million_at_70pct": 1.03,
  "cost_per_million_at_50pct": 1.44
}
```

Generated reports are in `results/reports/`.

## Models

### Native BitNet (Recommended for Quality)

| Model | Size | GGUF Size | Notes |
|-------|------|-----------|-------|
| BitNet-b1.58-2B-4T | 2B params | ~500MB | Quality baseline |
| Llama3-8B-1.58-100B | 8B params | ~1.1GB | Larger native model |

### Naive Conversion Candidates (For Cost Analysis Only)

| Model | Original Size | Ternary Size | Notes |
|-------|---------------|--------------|-------|
| Llama 3.1 70B | ~140GB | ~9GB | Large dense model |
| Mixtral 8x7B | ~94GB | ~6GB | MoE architecture |

## Interpreting Results

### Cost Efficiency

Lower $/1M tokens = more cost efficient. Compare:
- Same model on different hardware (GPU vs CPU)
- Different models on same hardware
- Spot vs on-demand pricing impact

### Utilization Scenarios

| Scenario | Utilization | Notes |
|----------|-------------|-------|
| Batch processing | ~100% | Constant workload |
| API with steady traffic | ~70% | Typical production |
| Interactive/bursty | ~50% | Variable demand |

### Quality vs Cost Trade-off

Native BitNet models provide both quality AND cost efficiency. Naive ternary conversion:
- Shows theoretical throughput limits
- Useful for architecture comparisons
- NOT suitable for quality evaluation

## Best Practices

1. **Use on-demand instances** for benchmarking (spot instances may be preempted mid-benchmark)
2. **Pre-cache Docker images** to reduce startup time
3. **Run multiple iterations** for statistical significance
4. **Test at different batch sizes** to find optimal throughput
5. **Include warmup requests** before measuring

**Note:** Do NOT use spot instances for benchmarks - they may be preempted during runs, leading to incomplete or inconsistent results. Use spot only for cost estimates, not actual benchmark measurements.

## Limitations

- Naive ternary conversion produces unusable outputs for production
- BitNet.cpp performance varies with CPU architecture (AVX512 required for best results)
- RunPod pricing changes frequently - verify current rates
- Memory estimates may vary based on context size and batch size

## References

- [BitNet.cpp](https://github.com/microsoft/BitNet) - Official 1.58-bit inference framework
- [BitNet Paper](https://arxiv.org/abs/2310.11453) - Original BitNet research
- [RunPod Pricing](https://www.runpod.io/pricing) - Current GPU/CPU rates
- [SkyPilot](https://skypilot.readthedocs.io/) - Cloud orchestration
