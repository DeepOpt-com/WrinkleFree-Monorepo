# Sparse Attention Techniques

Content-based sparse attention reduces computation by attending only to important tokens, determined dynamically based on attention patterns.

## Current Implementation

We have comprehensive sparse attention in `packages/inference/src/wf_infer/sglang_backend/sparse_attention.py`:

### Supported Modes

```python
class AttentionSparsityMode(str, Enum):
    NONE = "none"
    TOP_K = "top_k"           # Keep top-k attention scores per query
    THRESHOLD = "threshold"    # Zero out attention below threshold
    WINDOW = "window"          # Sliding window + global tokens
    DYNAMIC = "dynamic"        # Adaptive based on sequence complexity
```

### Configuration

```python
@dataclass
class AttentionSparsityConfig:
    enabled: bool = False
    mode: AttentionSparsityMode = AttentionSparsityMode.NONE

    # Top-k parameters
    top_k: Optional[int] = None  # Absolute number
    top_k_ratio: float = 0.25    # Ratio of seq_len (if top_k not set)

    # Threshold parameters
    threshold: float = 0.01

    # Window parameters
    window_size: int = 256
    global_tokens: int = 1
    stride: int = 64

    # Dynamic parameters
    dynamic_min_ratio: float = 0.1
    dynamic_max_ratio: float = 0.5
```

## Top-K Attention

Keep only the k highest attention scores per query position.

### How It Works

```python
def apply_top_k_attention(attn_weights: torch.Tensor, k: int):
    """Keep top-k attention scores, zero the rest."""
    seq_len = attn_weights.shape[-1]
    k = min(k, seq_len)

    # Get top-k indices
    _, topk_idx = torch.topk(attn_weights, k, dim=-1)

    # Create mask
    mask = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)

    # Apply and renormalize
    sparse_attn = attn_weights * mask.float()
    sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-8)

    return sparse_attn
```

### Pros and Cons

| Pros | Cons |
|------|------|
| Simple to implement | Still requires computing full attention first |
| Preserves most important tokens | Fixed k may not be optimal for all queries |
| No training required | Quality degrades with very small k |

### Recommended Settings

| Context | Top-k | Sparsity |
|---------|-------|----------|
| 4k | 256 | 93.75% |
| 8k | 512 | 93.75% |
| 32k | 1024 | 96.88% |
| 96k | 2048 | 97.87% |

## Threshold Attention

Zero out attention weights below a threshold.

### How It Works

```python
def apply_threshold_attention(attn_weights: torch.Tensor, threshold: float = 0.01):
    """Zero attention weights below threshold."""
    mask = attn_weights > threshold
    sparse_attn = attn_weights * mask.float()
    sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-8)
    return sparse_attn
```

### Adaptive Threshold

For different layers/heads:

```python
def adaptive_threshold(attn_weights: torch.Tensor, target_sparsity: float = 0.9):
    """Find threshold that achieves target sparsity."""
    # Sort all attention weights
    sorted_weights = attn_weights.flatten().sort().values

    # Find cutoff for target sparsity
    cutoff_idx = int(len(sorted_weights) * target_sparsity)
    threshold = sorted_weights[cutoff_idx].item()

    return apply_threshold_attention(attn_weights, threshold)
```

## Dynamic Sparsity (Entropy-Based)

Adjust sparsity per query based on attention entropy.

### Key Insight
- **Low entropy** = focused attention → can be more sparse
- **High entropy** = diffuse attention → need more context

### Implementation (Already in sparse_attention.py)

```python
def apply_dynamic_attention(
    attn_weights: torch.Tensor,
    min_ratio: float = 0.1,
    max_ratio: float = 0.5,
):
    """Adaptive sparsity based on attention entropy."""
    seq_len = attn_weights.shape[-1]

    # Compute entropy per query
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
    max_entropy = math.log(seq_len)
    normalized_entropy = entropy / max_entropy

    # Map entropy to keep ratio
    # Low entropy → low ratio (more sparse)
    # High entropy → high ratio (less sparse)
    keep_ratio = min_ratio + (max_ratio - min_ratio) * normalized_entropy

    # Apply per-query top-k
    k_per_query = (keep_ratio * seq_len).int().clamp(min=1, max=seq_len)

    # ... apply varying k per query
```

### Benefits
- Automatically adapts to content
- Preserves quality for complex queries
- Aggressive sparsity for focused queries

## Comparison: Sparse vs LSH

| Aspect | Top-K/Threshold | LSH Sampling |
|--------|----------------|--------------|
| **When computed** | After full attention | Before attention |
| **Complexity** | Still O(L^2) then sparse | O(L log L) |
| **Memory** | Full attention matrix | Hash tables |
| **Quality** | Exact top-k | Approximate but guaranteed |
| **Best for** | Moderate contexts | Very long contexts |

### When to Use Each

```
Context 4k-16k: Top-K or Dynamic sparsity
  - Full attention is still fast
  - Sparsity saves memory, minor speedup

Context 32k-64k: Combine sparse + LSH
  - Use LSH to find candidate keys
  - Apply top-k on candidates

Context 96k+: Full LSH
  - Full attention infeasible
  - LSH required for tractability
```

## Integration with BitNet Inference

### Current Integration Point

The sparse attention module is ready but needs to be integrated into the inference path.

```python
# In attention computation
from wf_infer.sglang_backend.sparse_attention import (
    apply_attention_sparsity,
    get_dynamic_attention_config,
)

def scaled_dot_product_attention(Q, K, V, sparse_config=None):
    # Compute raw attention scores
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    attn_weights = torch.softmax(attn_weights, dim=-1)

    # Apply sparsity if configured
    if sparse_config and sparse_config.enabled:
        attn_weights, sparsity = apply_attention_sparsity(attn_weights, sparse_config)
        logger.debug(f"Applied {sparse_config.mode.value} sparsity: {sparsity:.2%}")

    # Compute output
    output = torch.matmul(attn_weights, V)
    return output
```

### Enabling in DLM Server

Add configuration option:

```python
# In dlm_config or similar

@dataclass
class InferenceConfig:
    # ... existing fields ...

    attention_sparsity: AttentionSparsityConfig = field(
        default_factory=lambda: AttentionSparsityConfig(
            enabled=True,
            mode=AttentionSparsityMode.DYNAMIC,
            dynamic_min_ratio=0.1,
            dynamic_max_ratio=0.4,
        )
    )
```

## Benchmarking Sparse Attention

```python
# packages/inference/research/benchmarks/sparse_attention_benchmark.py

import torch
import time
from wf_infer.sglang_backend.sparse_attention import *

def benchmark_sparsity_modes(seq_lengths=[1024, 4096, 8192, 16384]):
    configs = {
        "none": get_default_attention_config(),
        "top_k_256": get_top_k_attention_config(k=256),
        "top_k_512": get_top_k_attention_config(k=512),
        "window_256": get_window_attention_config(window_size=256),
        "dynamic": get_dynamic_attention_config(),
    }

    results = []

    for seq_len in seq_lengths:
        # Random attention weights (simulating softmax output)
        attn = torch.softmax(torch.randn(1, 16, seq_len, seq_len), dim=-1)

        for name, config in configs.items():
            start = time.perf_counter()
            for _ in range(100):
                sparse_attn, sparsity = apply_attention_sparsity(attn, config)
            elapsed = (time.perf_counter() - start) / 100 * 1000  # ms

            results.append({
                "seq_len": seq_len,
                "mode": name,
                "time_ms": elapsed,
                "sparsity": sparsity,
            })

    return results
```

## Next Steps

1. **Enable in inference path** - Integrate `apply_attention_sparsity` into attention computation
2. **Add CLI flag** - `--attention-sparsity dynamic` or similar
3. **Benchmark quality** - Measure perplexity impact on LongBench
4. **Hybrid approach** - Combine dynamic sparsity with LSH for very long contexts
5. **Compile with Triton** - JIT compile sparsity operations for GPU speedup

## References

- DejaVu: https://arxiv.org/abs/2310.17157 (contextual sparsity)
- Longformer: https://arxiv.org/abs/2004.05150 (sparse patterns)
- BigBird: https://arxiv.org/abs/2007.14062 (sparse transformers)
- StreamingLLM: https://arxiv.org/abs/2309.17453 (attention sinks)
