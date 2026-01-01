# Sliding Window Attention

Sliding Window Attention (SWA) limits each token to attend only to a local window of neighboring tokens, reducing complexity from O(L^2) to O(L*w) where w is the window size.

## Key Papers

- **Longformer** (arXiv:2004.05150) - Sliding window + global attention
- **BigBird** (arXiv:2007.14062) - Sparse patterns including random + window
- **Mistral 7B** (2023) - Production sliding window implementation
- **SWAA** (arXiv:2512.10411) - Sliding window adaptation for pretrained models

## How Sliding Window Works

### Standard Attention
Every token attends to all tokens:
```
Token 0: attends to [0, 1, 2, ..., L-1]
Token 1: attends to [0, 1, 2, ..., L-1]
...
```

### Sliding Window Attention
Each token only attends to nearby tokens:
```
Window size = 256

Token 0:   attends to [0, 1, ..., 255]
Token 128: attends to [0, 1, ..., 383]
Token 256: attends to [0, 1, ..., 511]
Token 512: attends to [256, 257, ..., 767]
...
```

### Memory Complexity
- Standard: O(L^2) - 96k tokens = 9.2B scores
- Window (w=256): O(L*w) - 96k tokens = 24.6M scores (374x reduction)

## Existing Implementation

We already have sliding window in `sparse_attention.py`:

```python
# packages/inference/src/wrinklefree_inference/sglang_backend/sparse_attention.py

def create_window_mask(
    seq_len: int,
    window_size: int,
    global_tokens: int = 1,  # CLS token
    stride: int = 64,        # Global attention every N tokens
    device: torch.device = None,
) -> torch.Tensor:
    """Create sliding window + global attention mask."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    for i in range(seq_len):
        # Local window
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = True

        # Global tokens (first N can attend/be attended by all)
        mask[i, :global_tokens] = True
        mask[:global_tokens, i] = True

        # Strided global attention
        if i % stride == 0:
            mask[i, ::stride] = True
            mask[::stride, i] = True

    return mask
```

### Usage

```python
from wrinklefree_inference.sglang_backend.sparse_attention import (
    get_window_attention_config,
    apply_attention_sparsity,
)

# Configure window attention
config = get_window_attention_config(window_size=256)

# Apply during inference
sparse_attn, sparsity = apply_attention_sparsity(attn_weights, config)
# sparsity ≈ 0.95 (95% of attention scores zeroed) for 4k context
```

## Longformer-Style Global + Local

Longformer combines:
1. **Sliding window** - O(L*w) local attention
2. **Global tokens** - Selected tokens attend to all (e.g., [CLS], task tokens)
3. **Dilated sliding window** - Gaps in window for larger receptive field

```
Layer 1: Window size 256, no dilation
Layer 2: Window size 256, dilation 2 (effective window 512)
Layer 3: Window size 256, dilation 4 (effective window 1024)
```

### Implementation Enhancement

```python
def create_dilated_window_mask(
    seq_len: int,
    window_size: int,
    dilation: int = 1,
    global_tokens: int = 1,
) -> torch.Tensor:
    """Create dilated sliding window mask."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):
        # Dilated local window
        for offset in range(-window_size // 2, window_size // 2 + 1):
            j = i + offset * dilation
            if 0 <= j < seq_len:
                mask[i, j] = True

        # Global tokens
        mask[i, :global_tokens] = True
        mask[:global_tokens, i] = True

    return mask
```

## The Problem: Quality Degradation

**Naive sliding window on models trained with full attention causes quality loss.**

From SWAA paper (arXiv:2512.10411):
> "Naively enabling complete SWA at inference-time for models pretrained with full attention causes severe long-context performance degradation."

### Why Quality Degrades

1. **Long-range dependencies lost** - Information beyond window cannot propagate
2. **Position embedding mismatch** - Model expects to see all positions
3. **Attention pattern shift** - Model learned to use full context

## SWAA: Making SWA Work Without Full Retraining

SWAA (Sliding Window Attention Adaptation) proposes 5 methods:

### 1. Prefill-Only SWA
Apply SWA only during prefill (prompt processing), use full attention for decode.

```python
def attention(Q, K, V, phase="prefill"):
    if phase == "prefill":
        mask = create_window_mask(seq_len, window_size=512)
        attn = masked_attention(Q, K, V, mask)
    else:  # decode
        attn = full_attention(Q, K, V)
    return attn
```

**Speedup**: 2-5x on prefill (main bottleneck for long prompts)

### 2. Sink Tokens
Preserve first N tokens (usually 4-8) with full attention.

```python
def attention_with_sinks(Q, K, V, num_sinks=4, window_size=512):
    seq_len = K.shape[2]
    mask = create_window_mask(seq_len, window_size)

    # Sink tokens attend to and are attended by all
    mask[:, :num_sinks] = True
    mask[:num_sinks, :] = True

    return masked_attention(Q, K, V, mask)
```

**Why it works**: First tokens accumulate important global information (attention sinks)

### 3. Interleaved FA/SWA Layers
Alternate between full attention and sliding window layers.

```
Layer 0: Full Attention
Layer 1: Sliding Window
Layer 2: Full Attention
Layer 3: Sliding Window
...
```

**Trade-off**: 2x speedup while preserving long-range in every other layer

### 4. Chain-of-Thought (CoT) Prompting
For reasoning tasks, use CoT to reduce effective context dependency.

### 5. Fine-tuning
Short fine-tuning (1-5% of pretraining) on long-context data with SWA.

## Recommended Configuration for BitNet 2B

### Without Fine-tuning (Inference-Only)

```python
# Best combination of SWAA methods
class SWAAConfig:
    window_size: int = 512          # Covers most dependencies
    num_sinks: int = 4              # Preserve attention sinks
    global_stride: int = 128        # Strided global attention
    prefill_only: bool = True       # Full attention for decode

    # Layer-wise configuration (interleaved)
    def get_layer_config(self, layer_idx: int, num_layers: int = 24):
        # First and last 2 layers use full attention
        if layer_idx < 2 or layer_idx >= num_layers - 2:
            return "full"
        # Alternate for middle layers
        return "window" if layer_idx % 2 == 1 else "full"
```

### With Fine-tuning

If continued pretraining is acceptable:

```python
# Fine-tuning config for SWA adaptation
finetune_config = {
    "num_steps": 1000,              # ~1% of pretraining
    "context_length": 8192,         # Target context
    "window_size": 512,
    "learning_rate": 1e-5,          # Low LR for stability
    "data": "long_context_subset",  # Need long documents
}
```

## Performance Expectations

| Config | Context | Speedup | Quality Impact |
|--------|---------|---------|----------------|
| Window only | 8k | 4x | Moderate degradation |
| Window + sinks | 8k | 3.5x | Minor degradation |
| Prefill-only SWA | 8k | 2x prefill | Minimal |
| Interleaved | 8k | 2x | Minimal |
| Full SWAA combo | 8k | 3x | Minimal |
| SWAA + finetune | 8k | 5x+ | None |

## Integration with Existing Code

Current `sparse_attention.py` already supports window attention. To add SWAA:

```python
# Extend AttentionSparsityConfig

@dataclass
class SWAAConfig(AttentionSparsityConfig):
    """SWAA-specific configuration."""
    num_sinks: int = 4
    prefill_only: bool = True
    interleaved_layers: bool = True
    full_attention_layers: List[int] = field(default_factory=lambda: [0, 1, -2, -1])

def apply_swaa_attention(
    attn_weights: torch.Tensor,
    config: SWAAConfig,
    layer_idx: int,
    phase: str = "prefill",
) -> torch.Tensor:
    """Apply SWAA-style attention."""
    # Check if this layer uses full attention
    if layer_idx in config.full_attention_layers:
        return attn_weights, 0.0

    # Decode phase: use full attention
    if phase == "decode" and config.prefill_only:
        return attn_weights, 0.0

    # Apply window with sinks
    return apply_window_with_sinks(
        attn_weights,
        window_size=config.window_size,
        num_sinks=config.num_sinks,
        stride=config.stride,
    )
```

## Next Steps

1. **Enable existing window attention** in inference path
2. **Add sink token support** to `create_window_mask()`
3. **Implement layer-wise config** for interleaved FA/SWA
4. **Benchmark quality** on long-context tasks (RULER, LongBench)
5. **Consider fine-tuning** if quality is insufficient

## References

- Longformer: https://arxiv.org/abs/2004.05150
- BigBird: https://arxiv.org/abs/2007.14062
- SWAA: https://arxiv.org/abs/2512.10411
- Mistral 7B: https://mistral.ai/news/announcing-mistral-7b/
