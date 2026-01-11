# Long Context Fine-tuning

Strategies for extending BitNet 2B's context window through continued pretraining, including RoPE scaling and progressive context extension.

## Current State

BitNet 2B was trained with a specific context length (likely 2K-4K based on DLM training). Extending beyond this requires addressing:

1. **Position embeddings** - RoPE needs scaling
2. **Attention patterns** - Model needs to learn long-range dependencies
3. **Memory/compute** - Training on long sequences is expensive

## RoPE Position Embedding Scaling

Rotary Position Embeddings (RoPE) encode position through rotation. Scaling the rotation frequencies extends context.

### How RoPE Works

```python
def rope_embedding(x, position, dim, base=10000):
    """Standard RoPE embedding."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
    angles = position * freqs
    cos_emb = torch.cos(angles)
    sin_emb = torch.sin(angles)
    return apply_rotary(x, cos_emb, sin_emb)
```

### Scaling Methods

#### 1. Linear Scaling (Position Interpolation)

Scale positions linearly to fit in the original range.

```python
def linear_scaled_rope(x, position, dim, base=10000, scale=4.0):
    """Linear interpolation for context extension."""
    # Scale positions to fit original range
    scaled_position = position / scale

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
    angles = scaled_position * freqs
    return apply_rotary(x, torch.cos(angles), torch.sin(angles))

# Usage: scale=4.0 extends 2K -> 8K context
```

**Pros**: Simple, works reasonably well
**Cons**: Degrades perplexity, needs fine-tuning

#### 2. NTK-Aware Scaling (Dynamic NTK)

Adjust the RoPE base frequency instead of position scaling.

```python
def ntk_scaled_rope(x, position, dim, base=10000, context_len=8192, original_len=2048):
    """NTK-aware interpolation."""
    scale = context_len / original_len
    # Increase base frequency
    scaled_base = base * (scale ** (dim / (dim - 2)))

    freqs = 1.0 / (scaled_base ** (torch.arange(0, dim, 2) / dim))
    angles = position * freqs
    return apply_rotary(x, torch.cos(angles), torch.sin(angles))
```

**Pros**: Better perplexity than linear, minimal fine-tuning needed
**Cons**: Still some quality loss at extreme lengths

#### 3. YaRN (Yet another RoPE extensioN)

Combines NTK scaling with attention scaling and temperature adjustment.

```python
def yarn_rope(
    x, position, dim, base=10000,
    original_len=2048, target_len=8192,
    beta_fast=32, beta_slow=1,
):
    """YaRN RoPE scaling."""
    scale = target_len / original_len

    # Frequency-dependent interpolation
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))

    # High frequencies: linear interpolation
    # Low frequencies: NTK scaling
    wavelength = 2 * math.pi / freqs
    low_freq_mask = wavelength > (original_len / beta_fast)
    high_freq_mask = wavelength < (original_len / beta_slow)
    mid_mask = ~(low_freq_mask | high_freq_mask)

    # Apply different scaling to different frequency bands
    scaled_freqs = freqs.clone()
    scaled_freqs[high_freq_mask] = freqs[high_freq_mask]  # Keep high freq
    scaled_freqs[low_freq_mask] = freqs[low_freq_mask] / scale  # Scale low freq

    # Smooth interpolation for mid frequencies
    smooth = (wavelength[mid_mask] / (original_len / beta_fast) - 1) / (
        (original_len / beta_slow) / (original_len / beta_fast) - 1
    )
    scaled_freqs[mid_mask] = freqs[mid_mask] / ((1 - smooth) + smooth * scale)

    angles = position * scaled_freqs
    return apply_rotary(x, torch.cos(angles), torch.sin(angles))
```

**Pros**: Best quality at long contexts
**Cons**: More complex, requires attention temperature tuning

### Comparison

| Method | 4x Extension | 8x Extension | Training Needed |
|--------|--------------|--------------|-----------------|
| Linear | Good | Degrades | ~1% pretraining |
| NTK | Good | Moderate | ~0.5% pretraining |
| YaRN | Excellent | Good | ~0.1% pretraining |

## Progressive Context Extension

Train on progressively longer contexts instead of jumping directly.

### Schedule

```python
context_schedule = [
    (0, 1000, 2048),      # Steps 0-1000: 2K context
    (1000, 2000, 4096),   # Steps 1000-2000: 4K context
    (2000, 3000, 8192),   # Steps 2000-3000: 8K context
    (3000, 4000, 16384),  # Steps 3000-4000: 16K context
]

def get_context_length(step):
    for start, end, ctx_len in context_schedule:
        if start <= step < end:
            return ctx_len
    return context_schedule[-1][2]
```

### Benefits
- Model gradually adapts to longer contexts
- More stable training
- Better final quality

## BitNet-Specific Considerations

### Ternary Weight Stability

BitNet's 1.58-bit weights are discrete (−1, 0, +1). During continued pretraining:

1. **Gradient accumulation** - Small gradients may not flip ternary weights
2. **Learning rate** - May need higher LR to overcome quantization threshold
3. **STE (Straight-Through Estimator)** - Required for backprop through quantization

```python
def bitnet_long_context_training_config():
    return {
        # Higher LR for ternary weight updates
        "learning_rate": 3e-5,

        # Warmup important for stability
        "warmup_steps": 200,

        # Gradient clipping
        "max_grad_norm": 1.0,

        # Don't use weight decay (discrete weights)
        "weight_decay": 0.0,

        # Use STE for quantization
        "use_ste": True,
    }
```

### Memory Efficiency

BitNet's small weights allow training longer contexts:

```
Standard 2B model (FP16 weights):
- Weights: 4 GB
- Activations (8K context): 16 GB
- Total: 20 GB

BitNet 2B (1.58-bit weights):
- Weights: 0.5 GB
- Activations (8K context): 16 GB
- Total: 16.5 GB

→ Can fit 32K context in same memory!
```

## Training Protocol

### Data Preparation

```python
# Long-context data sources
data_config = {
    "sources": [
        {
            "name": "pg19",
            "weight": 0.3,
            "min_length": 8192,
        },
        {
            "name": "arxiv",
            "weight": 0.2,
            "min_length": 8192,
        },
        {
            "name": "github_code",
            "weight": 0.3,
            "min_length": 8192,
        },
        {
            "name": "slimpajama_long",
            "weight": 0.2,
            "min_length": 8192,
        },
    ],

    # Packing for efficiency
    "pack_sequences": True,
    "pack_separator": "<|endoftext|>",
}
```

### Training Script

```python
# packages/distillation/scripts/long_context_finetune.py

import torch
from dataclasses import dataclass

@dataclass
class LongContextConfig:
    # Model
    model_path: str = "models/bitnet-2b"
    output_path: str = "models/bitnet-2b-8k"

    # Context extension
    original_context: int = 2048
    target_context: int = 8192
    rope_scaling: str = "yarn"  # linear, ntk, yarn

    # Training
    num_steps: int = 2000
    batch_size: int = 4
    gradient_accumulation: int = 8
    learning_rate: float = 3e-5
    warmup_steps: int = 200

    # Progressive training
    progressive: bool = True
    context_schedule: list = None

def train_long_context(config: LongContextConfig):
    # Load model
    model = load_bitnet_model(config.model_path)

    # Apply RoPE scaling
    if config.rope_scaling == "yarn":
        model = apply_yarn_rope(
            model,
            original_len=config.original_context,
            target_len=config.target_context,
        )
    elif config.rope_scaling == "ntk":
        model = apply_ntk_rope(
            model,
            original_len=config.original_context,
            target_len=config.target_context,
        )
    elif config.rope_scaling == "linear":
        model = apply_linear_rope(
            model,
            scale=config.target_context / config.original_context,
        )

    # Training loop
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,  # No weight decay for BitNet
    )

    for step in range(config.num_steps):
        # Get context length for this step
        if config.progressive:
            ctx_len = get_progressive_context(step, config)
        else:
            ctx_len = config.target_context

        # Get batch
        batch = get_long_context_batch(ctx_len)

        # Forward pass
        loss = model(batch).loss

        # Backward pass
        loss.backward()

        if (step + 1) % config.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    # Save model
    model.save_pretrained(config.output_path)
```

### Cloud Deployment

```bash
# Using SkyPilot
sky launch -c long-context-train \
    --gpus A100:8 \
    --cloud nebius \
    python packages/distillation/scripts/long_context_finetune.py \
        --model-path gs://wrinklefree-checkpoints/bitnet-2b \
        --target-context 16384 \
        --rope-scaling yarn \
        --num-steps 2000
```

## Evaluation

### Perplexity at Different Lengths

```python
def eval_perplexity_by_length(model, eval_data, lengths=[2048, 4096, 8192, 16384]):
    results = {}

    for length in lengths:
        ppl = compute_perplexity(model, eval_data, max_length=length)
        results[length] = ppl
        print(f"Context {length}: PPL = {ppl:.2f}")

    return results
```

### Long-Context Benchmarks

```bash
# RULER (synthetic retrieval)
python eval/ruler.py --model bitnet-2b-8k --lengths 2048,4096,8192

# Passkey Retrieval
python eval/passkey.py --model bitnet-2b-8k --lengths 2048,4096,8192,16384

# LongBench (real-world tasks)
python eval/longbench.py --model bitnet-2b-8k
```

## Expected Results

| Method | 4K → 8K | 4K → 16K | 4K → 32K |
|--------|---------|----------|----------|
| No scaling | Broken | Broken | Broken |
| Linear (no train) | +5% PPL | +15% PPL | +40% PPL |
| Linear + train | +1% PPL | +3% PPL | +8% PPL |
| YaRN (no train) | +2% PPL | +5% PPL | +12% PPL |
| YaRN + train | ~0% PPL | +1% PPL | +3% PPL |

## Recommended Approach

1. **Start with YaRN** - Best quality without training
2. **Evaluate at target length** - Check if quality is acceptable
3. **If not, progressive fine-tune** - 1K-2K steps usually sufficient
4. **Combine with SWAA** - Use sliding window for even longer contexts

```python
# Combined approach: YaRN + SWAA
model_config = {
    "rope_scaling": "yarn",
    "yarn_original_len": 2048,
    "yarn_target_len": 32768,

    "swaa_enabled": True,
    "swaa_window_size": 4096,
    "swaa_num_sinks": 8,
}
# Result: 32K context with ~4K effective window per layer
```

## References

- Position Interpolation: https://arxiv.org/abs/2306.15595
- NTK-Aware Scaling: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
- YaRN: https://arxiv.org/abs/2309.00071
- LongLoRA: https://arxiv.org/abs/2309.12307
- LongRoPE: https://arxiv.org/abs/2402.13753
