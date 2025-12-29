# BitNet Cost Benchmark Results

## Summary

Date: 2025-12-22

### BitNet 2B-4T on RunPod A40 (CPU Inference)

| Metric | Value |
|--------|-------|
| Model | BitNet-b1.58-2B-4T |
| Hardware | RunPod A40 node (CPU inference) |
| Quantization | Native 1.58-bit (i2_s) |
| Hardware Cost | $0.39/hr (on-demand) |

#### Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | 2.78 tokens/sec |
| TTFT P50 | 21.6 seconds |
| TTFT P99 | 35.5 seconds |
| Latency P50 | 72.0 seconds |
| Latency P99 | 118.5 seconds |
| Memory Usage | 0.04 GB |

#### Cost Analysis

| Utilization | Cost per 1M Tokens |
|-------------|-------------------|
| 100% | $38.90 |
| 70% | $55.57 |
| 50% | $77.80 |

### Observations

1. **CPU vs GPU**: BitNet.cpp runs on CPU with AVX512 optimization. The A40 GPU is not used for inference.

2. **Performance**: The 2.78 tok/s throughput is lower than expected. Server logs showed 3.5-4.5 tok/s for individual requests.

3. **High Latency**: The 21.6s TTFT and 72s request latency indicate the server is processing requests sequentially with only 1 slot.

4. **Cost Efficiency**: At $38.90/1M tokens, this is significantly more expensive than cloud APIs:
   - GPT-4o-mini: $0.60/1M output tokens
   - Claude 3 Haiku: $1.25/1M output tokens

### Recommendations

1. **Use dedicated CPU instances**: The A40 GPU is wasted on CPU inference. Use pure CPU instances (e.g., c2-standard-60) for better cost efficiency.

2. **Enable multi-slot**: Run with `-np N` to enable multiple concurrent slots.

3. **Optimize thread count**: Current build uses auto-detected threads. Explicit tuning may improve performance.

4. **Consider batch processing**: Batch requests together to maximize throughput.

### Raw Data

Results stored in `results/raw/`:
- `20251222_235425_bitnet-2b-4t_native_a40.json`

### Methodology

- Warmup: 5 requests
- Batch sizes tested: 1, 4, 8, 16
- Duration: 120 seconds
- Tokens per request: 100 (default)
