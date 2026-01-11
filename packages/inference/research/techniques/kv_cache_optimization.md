# KV Cache Optimization

The KV cache stores key and value tensors from previous tokens to avoid recomputation. For long contexts, it becomes the dominant memory consumer.

## Memory Analysis

### BitNet 2B KV Cache Size

```
Model: BitNet 2B
- Layers: 24
- Heads: 16
- Head dim: 128
- KV per token per layer: 2 * 16 * 128 = 4096 values

Memory per token:
- BF16: 4096 * 2 bytes * 24 layers = 196 KB
- INT8: 4096 * 1 byte * 24 layers = 98 KB
- INT4: 4096 * 0.5 bytes * 24 layers = 49 KB

Total for different contexts:
| Context | BF16 | INT8 | INT4 |
|---------|------|------|------|
| 4k | 768 MB | 384 MB | 192 MB |
| 8k | 1.5 GB | 768 MB | 384 MB |
| 32k | 6 GB | 3 GB | 1.5 GB |
| 96k | 18 GB | 9 GB | 4.5 GB |
```

## Current Implementation

We have `KVCache` in `packages/inference/src/wf_infer/kv_cache/kv_cache.py`:

```python
class KVCacheDtype(Enum):
    BF16 = "bfloat16"
    FP16 = "float16"
    FP32 = "float32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"  # Default - 50% memory savings
```

### Current Features
- INT8 quantization with per-layer scales
- FP8 simulation (stored as int8 with scaling)
- Memory usage tracking

## Optimization 1: INT4 KV Cache

Further reduce memory by 2x compared to INT8.

### Implementation

```python
class KVCacheDtype(Enum):
    # ... existing ...
    INT4 = "int4"  # 75% memory savings vs BF16

class KVCache:
    def _quantize_to_int4(
        self, tensor: torch.Tensor, layer_idx: int, is_key: bool
    ) -> torch.Tensor:
        """Quantize to INT4 (packed as uint8, 2 values per byte)."""
        abs_max = tensor.abs().max().item()
        if abs_max < 1e-6:
            abs_max = 1.0

        # Scale to [-7, 7] (4-bit signed range)
        scale = 7.0 / abs_max
        quantized = (tensor * scale).round().clamp(-7, 7).to(torch.int8)

        # Pack 2 int4 values per uint8
        even = quantized[..., 0::2] + 8  # Shift to [1, 15]
        odd = quantized[..., 1::2] + 8
        packed = (even << 4) | odd

        if is_key:
            self.k_scales[layer_idx] = 1.0 / scale
        else:
            self.v_scales[layer_idx] = 1.0 / scale

        return packed.to(torch.uint8)

    def _dequantize_from_int4(
        self, tensor: torch.Tensor, layer_idx: int, is_key: bool
    ) -> torch.Tensor:
        """Dequantize INT4 to BF16."""
        scale = self.k_scales[layer_idx] if is_key else self.v_scales[layer_idx]

        # Unpack 2 int4 values per uint8
        even = ((tensor >> 4) & 0xF).to(torch.int8) - 8
        odd = (tensor & 0xF).to(torch.int8) - 8

        # Interleave back
        unpacked = torch.stack([even, odd], dim=-1).flatten(-2)

        return (unpacked.to(torch.float32) * scale).to(torch.bfloat16)
```

### llama.cpp INT4 KV Cache

llama.cpp supports quantized KV cache:

```bash
# Enable INT4 KV cache
llama-cli -m model.gguf -ctk q4_0 -ctv q4_0 -fa on
```

For our BitNet.cpp integration, add to launch script:
```bash
# packages/inference/scripts/serve_native.sh
./llama-server \
    --model "$GGUF_PATH" \
    --ctx-size 32768 \
    --kv-cache-type q4_0 \  # INT4 KV cache
    --flash-attn \           # Required for quantized KV
    ...
```

## Optimization 2: KV Cache Eviction (H2O)

Heavy-Hitter Oracle (H2O) evicts less important KV entries based on attention scores.

### Key Insight
Not all tokens are equally important. Some tokens (heavy hitters) receive most attention across all queries.

### Implementation

```python
class H2OKVCache(KVCache):
    """KV Cache with Heavy-Hitter Oracle eviction."""

    def __init__(self, config: KVCacheConfig, budget_ratio: float = 0.2):
        super().__init__(config)
        self.budget_ratio = budget_ratio  # Keep top 20% of tokens
        self.token_importance = torch.zeros(config.max_seq_len)

    def update_importance(self, attention_weights: torch.Tensor):
        """Track cumulative attention each token receives."""
        # Sum attention across heads and queries
        importance = attention_weights.sum(dim=(0, 1, 2))  # [seq_len]
        self.token_importance[:len(importance)] += importance

    def evict_if_needed(self, current_len: int):
        """Evict low-importance tokens if over budget."""
        budget = int(self.config.max_seq_len * self.budget_ratio)

        if current_len <= budget:
            return

        # Keep top-k important tokens
        _, keep_indices = torch.topk(
            self.token_importance[:current_len],
            k=budget
        )
        keep_indices = keep_indices.sort().values

        # Compact cache
        for layer_idx in range(self.config.num_layers):
            self.cache[layer_idx, :, :budget] = self.cache[layer_idx, :, keep_indices]

        self.current_seq_len = budget
        self.token_importance[:budget] = self.token_importance[keep_indices]
        self.token_importance[budget:] = 0
```

### Eviction Strategies

| Strategy | Description | Quality Impact |
|----------|-------------|----------------|
| **H2O** | Keep heavy-hitter tokens | Minimal for most tasks |
| **LRU** | Keep recent tokens | Good for local tasks, bad for long-range |
| **Strided** | Keep every Nth token | Predictable but loses local context |
| **Hybrid** | Recent + heavy-hitters | Best balance |

## Optimization 3: Paged Attention

Avoid memory fragmentation with page-based allocation.

### Benefits
- No memory fragmentation
- Dynamic batch sizes
- Efficient memory reuse across requests

### Current Implementation

Already have `page_size` in config:
```python
@dataclass
class KVCacheConfig:
    page_size: int = 16  # Token page size for paged attention
```

### Full Paged Attention

```python
class PagedKVCache:
    """Paged attention KV cache like vLLM."""

    def __init__(self, config: KVCacheConfig):
        self.page_size = config.page_size
        self.num_pages = config.max_seq_len // config.page_size

        # Page table: maps logical page -> physical page
        self.page_table = torch.zeros(self.num_pages, dtype=torch.int32)

        # Physical pages: [num_physical, page_size, ...]
        self.physical_pages = torch.zeros(
            self.num_pages,
            config.num_layers,
            2,  # K and V
            self.page_size,
            config.num_heads,
            config.head_dim,
            dtype=self.storage_dtype,
        )

        self.free_pages = list(range(self.num_pages))
        self.allocated_pages = []

    def allocate_page(self) -> int:
        """Get a free physical page."""
        if not self.free_pages:
            raise RuntimeError("No free pages - need eviction")
        return self.free_pages.pop()

    def get_kv_for_token(self, token_idx: int, layer_idx: int):
        """Get KV for a specific token."""
        logical_page = token_idx // self.page_size
        offset = token_idx % self.page_size

        physical_page = self.page_table[logical_page]
        return self.physical_pages[physical_page, layer_idx, :, offset]
```

## Optimization 4: Radix Prefix Caching

Already implemented in `radix_cache.rs`:

```rust
// packages/inference/extern/sglang-bitnet/sgl-model-gateway/src/inference/radix_cache.rs

pub struct RadixCache {
    tree: RadixTree<CachedSequence>,
    max_cached_tokens: usize,
    eviction_policy: EvictionPolicy,
}
```

### Benefits
- Zero-copy KV sharing for common prefixes
- O(k) prefix matching where k = prefix length
- LRU eviction for memory management

### Usage
Prefix caching is automatic when multiple requests share prompts:
```
Request 1: "You are a helpful assistant. What is 2+2?"
Request 2: "You are a helpful assistant. What is the capital of France?"
                ^--- Shared prefix, KV cache reused
```

## Recommended Optimization Order

1. **INT4 KV Cache** (Easy, 2x memory savings)
   - Enable in llama.cpp with `-ctk q4_0 -ctv q4_0`
   - Implement in Python `kv_cache.py`

2. **Enable Radix Prefix Caching** (Already implemented)
   - Verify it's active in `dlm_server`

3. **H2O Eviction** (Medium complexity)
   - Add importance tracking to attention
   - Implement eviction when approaching memory limit

4. **Full Paged Attention** (Complex)
   - Major refactor but enables dynamic batching
   - Consider using vLLM's implementation

## Benchmarking

```python
# Test memory and quality impact

configs = [
    ("BF16", KVCacheDtype.BF16),
    ("INT8", KVCacheDtype.INT8),
    ("INT4", KVCacheDtype.INT4),
]

for name, dtype in configs:
    cache = KVCache(KVCacheConfig(
        max_seq_len=8192,
        dtype=dtype,
    ))

    print(f"{name}: {cache.memory_usage_mb():.1f} MB")

    # Run perplexity eval
    ppl = evaluate_perplexity(model, cache)
    print(f"{name} perplexity: {ppl:.2f}")
```

## Expected Results

| Optimization | Memory Savings | Quality Impact | Implementation Effort |
|--------------|----------------|----------------|----------------------|
| INT8 (current) | 50% | Minimal | Done |
| INT4 | 75% | ~1-2% PPL increase | Low |
| H2O Eviction | Variable | Task-dependent | Medium |
| Paged Attention | Fragmentation | None | High |
| Radix Caching | Sharing | None | Done |

## References

- H2O: https://arxiv.org/abs/2306.14048
- vLLM Paged Attention: https://arxiv.org/abs/2309.06180
- KV Cache Quantization: https://arxiv.org/abs/2401.18079
- SGLang Radix Caching: https://arxiv.org/abs/2312.07104
