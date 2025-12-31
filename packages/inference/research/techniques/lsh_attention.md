# LSH Attention for Long Context Inference

Locality Sensitive Hashing (LSH) enables sub-quadratic attention by hashing similar queries and keys into the same buckets, then only computing attention within buckets.

## Key Papers

- **MagicPIG** (arXiv:2410.16179) - LSH sampling for LLM generation, CPU-heterogeneous
- **Reformer** (arXiv:2001.04451) - Original LSH attention, O(L log L) complexity

## How LSH Attention Works

### Traditional Attention Problem
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
```
- Complexity: O(L^2) where L = sequence length
- At 96k tokens: 9.2 billion attention score computations per layer

### LSH Solution

1. **Hash queries and keys** using locality-sensitive hash functions
2. **Group into buckets** - similar vectors land in same bucket with high probability
3. **Compute attention within buckets only** - reduces O(L^2) to O(L log L)

```python
# Conceptual LSH attention
def lsh_attention(Q, K, V, num_hashes=8, bucket_size=64):
    # Hash Q and K using random projections
    q_hashes = [lsh_hash(Q, seed=i) for i in range(num_hashes)]
    k_hashes = [lsh_hash(K, seed=i) for i in range(num_hashes)]

    # For each query, attend only to keys in same bucket
    for bucket in get_matching_buckets(q_hashes, k_hashes):
        q_bucket, k_bucket, v_bucket = bucket
        attn = softmax(q_bucket @ k_bucket.T) @ v_bucket

    return combine_bucket_outputs(attentions)
```

## MagicPIG: CPU-Heterogeneous LSH

MagicPIG (arXiv:2410.16179) is specifically designed for long-context inference with a heterogeneous CPU-GPU architecture.

### Key Insight
> "Attention is not always as sparse as expected" - naive TopK selection loses important tokens

MagicPIG uses **sampling with theoretical guarantees** instead of TopK:

### Architecture
```
┌─────────────────────────────────────────┐
│                  GPU                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Prefill │→ │ Decode  │→ │ Output  │  │
│  └────┬────┘  └────┬────┘  └─────────┘  │
│       │            │                     │
└───────┼────────────┼─────────────────────┘
        │            │
        ▼            ▼
┌─────────────────────────────────────────┐
│                  CPU                     │
│  ┌──────────────────────────────────┐   │
│  │     LSH Hash Tables (KV Cache)    │   │
│  │  - Query hashing                  │   │
│  │  - Bucket lookup                  │   │
│  │  - Candidate sampling             │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Why CPU for LSH?
1. **Memory**: LSH hash tables can be large, CPU RAM is cheaper
2. **Parallelism**: Hash table lookups parallelize well on CPU
3. **Frees GPU**: GPU focuses on matmuls, CPU handles indexing

### Performance
- **96k context on RTX 4090**: 54ms decode latency
- **Throughput**: Up to 5x improvement over full attention
- **Accuracy**: Preserves quality across diverse tasks (better than TopK)

## Implementation for BitNet 2B

### Considerations for BitNet

1. **Ternary weights** don't affect attention (Q, K, V are still FP16/BF16)
2. **CPU-only inference** - can use same CPU for both LSH and attention
3. **Lower memory** - BitNet's 2-bit weights leave more RAM for LSH tables

### Proposed Integration

```python
# packages/inference/src/wrinklefree_inference/lsh_attention.py

class LSHAttention:
    def __init__(
        self,
        num_hashes: int = 8,        # Number of hash tables (more = better recall)
        bucket_size: int = 64,       # Tokens per bucket
        sample_ratio: float = 0.2,   # Fraction of KV to sample (20% = 5x speedup)
    ):
        self.hash_tables = [RandomProjectionLSH() for _ in range(num_hashes)]

    def build_index(self, keys: torch.Tensor):
        """Build LSH index for keys (during prefill)."""
        for table in self.hash_tables:
            table.add(keys)

    def query(self, query: torch.Tensor) -> List[int]:
        """Get candidate key indices for a query."""
        candidates = set()
        for table in self.hash_tables:
            bucket = table.get_bucket(query)
            candidates.update(bucket)
        return list(candidates)

    def attention(self, Q, K, V, mask=None):
        """Compute attention using LSH sampling."""
        batch_size, num_heads, seq_len, head_dim = Q.shape

        # Get candidates for each query
        outputs = []
        for q_idx in range(seq_len):
            candidates = self.query(Q[:, :, q_idx])

            # Sample subset of candidates
            sampled = sample_with_replacement(candidates, k=self.sample_ratio * len(K))

            # Compute attention only on sampled keys
            k_sampled = K[:, :, sampled]
            v_sampled = V[:, :, sampled]

            attn = softmax(Q[:, :, q_idx] @ k_sampled.T) @ v_sampled
            outputs.append(attn)

        return torch.stack(outputs, dim=2)
```

### Integration with Existing KV Cache

```python
# Extend existing kv_cache.py

class LSHKVCache(KVCache):
    def __init__(self, config: KVCacheConfig, lsh_config: LSHConfig):
        super().__init__(config)
        self.lsh = LSHAttention(**lsh_config)

    def update(self, layer_idx, key, value, seq_pos):
        super().update(layer_idx, key, value, seq_pos)
        # Update LSH index
        self.lsh.build_index(key)

    def attention_with_lsh(self, query, layer_idx):
        """Compute attention using LSH sampling."""
        key, value = self.get(layer_idx)
        return self.lsh.attention(query, key, value)
```

## Hyperparameter Tuning

| Parameter | Range | Effect |
|-----------|-------|--------|
| `num_hashes` | 4-16 | More = better recall, slower indexing |
| `bucket_size` | 32-128 | Smaller = more sparse, risk of missing important tokens |
| `sample_ratio` | 0.1-0.3 | Lower = faster, higher quality loss risk |

### Recommended Settings by Context Length

| Context | num_hashes | bucket_size | sample_ratio |
|---------|------------|-------------|--------------|
| 8k-16k | 4 | 64 | 0.25 |
| 32k-64k | 8 | 64 | 0.20 |
| 96k+ | 12 | 128 | 0.15 |

## Comparison with Alternatives

| Method | Complexity | Quality | Training Required |
|--------|------------|---------|-------------------|
| Full Attention | O(L^2) | Best | No |
| TopK Attention | O(L*k) | Good | No |
| **LSH Attention** | O(L log L) | Good | No |
| Sliding Window | O(L*w) | Degraded for long-range | Sometimes |
| Linear Attention | O(L) | Lower | Yes |

## Next Steps

1. **Implement `LSHAttention` class** with CPU-optimized hash tables
2. **Benchmark on 8k, 32k, 96k contexts** vs current sparse attention
3. **Tune hyperparameters** for BitNet 2B specifically
4. **Consider hybrid approach**: LSH for long context + full attention for recent tokens

## References

- MagicPIG: https://arxiv.org/abs/2410.16179
- Reformer: https://arxiv.org/abs/2001.04451
- FAISS (efficient similarity search): https://github.com/facebookresearch/faiss
