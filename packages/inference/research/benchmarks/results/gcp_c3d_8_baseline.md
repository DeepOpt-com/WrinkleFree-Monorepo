# Baseline Benchmark Results

**Date**: 2025-12-31
**Instance**: GCP c3d-standard-8 (us-central1-a)
**Model**: BitNet 2B TQ2_0 (770.94 MiB)
**CPU**: 8 vCPUs, 32GB RAM
**Threads**: 8

## Prefill Performance (Prompt Processing)

| Context Length | Tokens/sec | Slowdown vs 512 |
|----------------|------------|-----------------|
| 512 | 27.17 ± 0.11 | 1.00x |
| 1024 | 26.64 ± 0.29 | 1.02x |
| 2048 | 26.09 ± 0.08 | 1.04x |
| 4096 | 24.28 ± 0.23 | 1.12x |

## Decode Performance (Token Generation)

| Metric | Value |
|--------|-------|
| Decode speed | ~29.3 tok/s |

*Note: Decode measured from interactive session, llama-bench decode test in progress.*

## Analysis

### Good News
- Prefill scales well from 512 → 4096 (only 12% slowdown)
- Decode speed (~29 tok/s) is reasonable for CPU-only inference
- 2B model fits comfortably in 32GB RAM

### Areas for Optimization
- Prefill will likely degrade significantly at 8k-16k+ tokens
- Current implementation uses vanilla llama.cpp (no specialized BitNet kernels)
- KV cache is using default settings (likely FP16)

## Recommendations

Based on research docs in `packages/inference/research/techniques/`:

1. **For 8k-16k context**: Enable sliding window attention (SWAA)
   - Expected speedup: 2-4x on prefill
   - No training required

2. **For 32k+ context**: Implement LSH attention (MagicPIG)
   - Expected speedup: 3-5x
   - CPU-heterogeneous architecture ideal for this setup

3. **Memory optimization**: Enable INT4 KV cache
   - Would reduce KV memory by 75%
   - llama.cpp supports: `-ctk q4_0 -ctv q4_0`

## Raw llama-bench Output

```
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| bitnet ?B TQ2_0 - 2.06 bpw ternary | 770.94 MiB |     2.41 B | CPU        |       8 |           pp512 |         27.17 ± 0.11 |
| bitnet ?B TQ2_0 - 2.06 bpw ternary | 770.94 MiB |     2.41 B | CPU        |       8 |          pp1024 |         26.64 ± 0.29 |
| bitnet ?B TQ2_0 - 2.06 bpw ternary | 770.94 MiB |     2.41 B | CPU        |       8 |          pp2048 |         26.09 ± 0.08 |
| bitnet ?B TQ2_0 - 2.06 bpw ternary | 770.94 MiB |     2.41 B | CPU        |       8 |          pp4096 |         24.28 ± 0.23 |
```

## Cost

- Instance: ~$0.25/hour
- Benchmark time: ~20 minutes
- Total cost: ~$0.08
