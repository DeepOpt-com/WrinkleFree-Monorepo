# Long Context Inference Optimization Research

Research documentation for optimizing BitNet 2B inference at long context lengths (8k-96k+ tokens) on CPU.

## Current Baseline
- **Model**: DLM BitNet 2B (1.58-bit quantized)
- **Performance**: ~20 tok/s at 2k tokens
- **Target**: Maintain speed at 8k-96k+ tokens

## Techniques Overview

| Technique | Best For | Speedup | Requires Training |
|-----------|----------|---------|-------------------|
| [LSH Attention](techniques/lsh_attention.md) | 32k-96k+ | Up to 5x | No |
| [Sliding Window](techniques/sliding_window.md) | 8k-32k | 2-10x | Optional |
| [KV Cache Optimization](techniques/kv_cache_optimization.md) | All | 1.5-2x memory | No |
| [Sparse Attention](techniques/sparse_attention.md) | All | 1.5-3x | No |

## Pretraining Approaches

| Approach | Description | Training Cost |
|----------|-------------|---------------|
| [SWAA](pretraining/swaa_adaptation.md) | Sliding window adaptation | Low (fine-tune) |
| [Long Context Finetuning](pretraining/long_context_finetuning.md) | Progressive extension | Medium |

## Quick Start

1. **Benchmark current performance**:
   ```bash
   uv run python packages/inference/research/benchmarks/baseline_benchmark.py
   ```

2. **Choose technique based on target context**:
   - 8k-16k: Start with sliding window + existing sparse attention
   - 32k-64k: Add KV cache quantization + LSH sampling
   - 96k+: Full MagicPIG implementation

3. **Evaluate quality/speed tradeoff** using benchmark suite

## Priority Recommendations

### Immediate (No Training)
1. **KV Cache INT4 Quantization** - Already have INT8, move to INT4 for 2x memory savings
2. **Enable Existing Sparse Attention** - `sparse_attention.py` already implements dynamic sparsity
3. **LSH Sampling** - Implement MagicPIG for CPU-heterogeneous attention

### With Fine-tuning
1. **SWAA Adaptation** - 5 combined methods for up to 100x speedup
2. **RoPE Scaling** - Extend position embeddings for longer contexts

## References
See [papers.md](references/papers.md) for key paper summaries.
