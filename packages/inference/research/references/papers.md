# Key Papers Reference

Summaries of important papers for long-context inference optimization.

## LSH Attention

### MagicPIG: LSH Sampling for Efficient LLM Generation
**arXiv:2410.16179** | October 2024

**Problem**: KV cache bottleneck in long-context LLM inference.

**Key Insight**: "Attention is not always as sparse as expected" - TopK selection misses important tokens.

**Method**:
- Use LSH to build hash tables of keys
- Sample candidates from matching buckets (not exact TopK)
- Store hash tables on CPU, compute attention on GPU
- Theoretical guarantees on approximation quality

**Results**:
- 5x throughput improvement
- 54ms decode latency at 96K context on RTX 4090
- Preserves quality across diverse tasks

**Relevance for BitNet**: CPU-only inference can use same architecture - store LSH on CPU RAM, compute attention there too.

---

### Reformer: The Efficient Transformer
**arXiv:2001.04451** | ICLR 2020

**Problem**: O(L²) attention complexity limits context length.

**Method**:
- LSH attention: hash similar Q/K to same buckets, attend within buckets
- Reversible residual layers: don't store activations for backprop
- Complexity: O(L log L) attention

**Results**:
- Process 1M tokens on single GPU
- Competitive quality on long-sequence tasks

**Relevance**: Original LSH attention formulation. MagicPIG improves on this with sampling.

---

## Sliding Window Attention

### SWAA: Sliding Window Attention Adaptation
**arXiv:2512.10411** | December 2025

**Problem**: Naively applying sliding window to full-attention models degrades quality.

**Method**: 5 practical recipes:
1. Prefill-only SWA (decode uses full attention)
2. Sink tokens (first N tokens see everything)
3. Interleaved FA/SWA layers
4. Chain-of-thought prompting
5. Short fine-tuning (~1% pretraining)

**Results**: Up to 100x speedup with fine-tuning, 3-4x without.

**Relevance**: Immediately applicable - can enable existing sparse attention code.

---

### Longformer: The Long-Document Transformer
**arXiv:2004.05150** | 2020

**Method**:
- Sliding window attention (local)
- Global tokens (task-specific tokens attend everywhere)
- Dilated sliding window (larger receptive field in deeper layers)

**Results**: 4K → 16K context, linear complexity.

**Relevance**: Our `sparse_attention.py` already implements this pattern.

---

## KV Cache Optimization

### H2O: Heavy-Hitter Oracle for Efficient LLM Inference
**arXiv:2306.14048** | 2023

**Problem**: KV cache grows linearly with context, limiting max length.

**Key Insight**: Small fraction of tokens (heavy hitters) receive most attention.

**Method**:
- Track cumulative attention each token receives
- Evict low-importance tokens when cache full
- Keep recent tokens + heavy hitters

**Results**: 5-10x cache reduction with minimal quality loss.

**Relevance**: Can be added to existing KVCache class.

---

### Efficient Memory Management for LLM Serving (vLLM)
**arXiv:2309.06180** | 2023

**Method**:
- Paged attention: allocate KV cache in fixed-size pages
- Avoid fragmentation, enable dynamic batching
- Copy-on-write for shared prefixes

**Results**: 2-4x throughput improvement.

**Relevance**: Radix cache in `radix_cache.rs` implements similar prefix sharing.

---

## Position Embeddings

### YaRN: Efficient Context Window Extension
**arXiv:2309.00071** | 2023

**Problem**: Extending context beyond training length degrades quality.

**Method**:
- Frequency-dependent interpolation
- High frequencies: keep original (local patterns)
- Low frequencies: scale (global patterns)
- Attention temperature adjustment

**Results**: 4x extension with minimal PPL increase, 8x with fine-tuning.

**Relevance**: Best method for extending BitNet context window.

---

### LongRoPE: Extending LLM Context to 2M Tokens
**arXiv:2402.13753** | 2024

**Method**:
- Two-stage approach: 256K then 2M
- Non-uniform position interpolation
- Progressive fine-tuning

**Results**: 2M token context with acceptable quality.

**Relevance**: Extreme context extension if needed beyond 100K.

---

## Efficient Attention (General)

### FlashAttention: Fast and Memory-Efficient Exact Attention
**arXiv:2205.14135** | 2022

**Problem**: Attention is memory-bound, not compute-bound.

**Method**:
- Tiled attention computation
- Avoid materializing full attention matrix
- IO-aware algorithm (minimize HBM reads/writes)

**Results**: 2-4x speedup, O(N) memory instead of O(N²).

**Relevance**: Backend for efficient attention computation. CPU version: FlashInfer.

---

### FlashAttention-2: Faster with Better Parallelism
**arXiv:2307.08691** | 2023

**Improvements**:
- Better work partitioning
- Reduced non-matmul FLOPs
- Support for multi-query/grouped-query attention

**Results**: 2x speedup over FlashAttention-1.

---

## Sparse Attention

### Efficient Attention Mechanisms for LLMs: A Survey
**arXiv:2507.19595** | July 2025

**Categories**:
1. **Linear attention**: Kernel approximations, recurrent formulations
2. **Sparse attention**: Fixed patterns, learned sparsity, clustering
3. **Efficient implementations**: FlashAttention, paged attention

**Key Techniques**:
- Static sparsity: sliding window, strided, dilated
- Dynamic sparsity: top-k, threshold, learned masks
- Content-based: LSH, clustering, retrieval

**Relevance**: Comprehensive taxonomy for understanding trade-offs.

---

### Native Sparse Attention: Hardware-Aligned and Trainable
**arXiv:2502.11089** | February 2025

**Method**:
- Triton kernel for sparse attention patterns
- Trainable sparsity masks
- Hardware-aligned block sizes

**Results**: Up to 11x speedup over dense attention.

**Relevance**: Implementation guide for custom sparse patterns.

---

## BitNet-Specific

### BitNet: Scaling 1-bit Transformers
**arXiv:2310.11453** | 2023

**Method**:
- 1-bit weights: {-1, +1}
- Full-precision activations
- Straight-Through Estimator for training

---

### BitNet b1.58: Every Weight is Ternary
**arXiv:2402.17764** | 2024

**Method**:
- 1.58-bit weights: {-1, 0, +1}
- Improved quality over binary
- Absmean quantization function

**Key Quote**: "BitNet b1.58 matches full-precision LLaMA at 3B parameters while being 2.5x faster."

---

### The Era of 1-bit LLMs
**Microsoft Research, 2024**

**Contributions**:
- bitnet.cpp: CPU inference engine for BitNet
- Custom kernels for ternary matmul
- 5-7x faster than FP16 LLaMA.cpp on CPU

**Relevance**: Our inference engine builds on this work.

---

## Quick Reference Table

| Paper | Technique | Speedup | Complexity | Training |
|-------|-----------|---------|------------|----------|
| MagicPIG | LSH sampling | 5x | O(L log L) | No |
| SWAA | Sliding window adapt | 3-100x | O(L*w) | Optional |
| H2O | KV eviction | 5-10x mem | O(L) | No |
| YaRN | RoPE scaling | - | - | Optional |
| FlashAttention | IO-aware | 2-4x | O(L²) | No |
| Native Sparse | Trainable sparse | 11x | Variable | Yes |

## Implementation Priority

1. **Enable SWAA** (our sparse_attention.py + config)
2. **Add LSH attention** (MagicPIG-style)
3. **Implement H2O eviction** (extend KVCache)
4. **Apply YaRN** (if extending context)
5. **Compile kernels** (Triton/native for speedup)
