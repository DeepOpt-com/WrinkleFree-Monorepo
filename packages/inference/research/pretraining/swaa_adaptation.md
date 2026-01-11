# SWAA: Sliding Window Attention Adaptation

SWAA (arXiv:2512.10411, December 2025) provides practical recipes for adapting models pretrained with full attention to use sliding window attention at inference time, achieving up to **100x inference speedup**.

## The Problem

Models trained with full attention suffer quality degradation when naively switching to sliding window at inference:

```
Full Attention (training):
Token i can see: [0, 1, 2, ..., L-1]

Sliding Window (inference):
Token i can see: [max(0, i-w), ..., i+w]

Mismatch causes:
- Lost long-range dependencies
- Position embedding confusion
- Attention pattern shift
```

## SWAA's Five Methods

SWAA proposes combining multiple techniques:

### 1. Prefill-Only SWA

Apply sliding window only during prompt processing (prefill), use full attention during generation (decode).

**Rationale**: Prefill is the bottleneck for long prompts; decode processes one token at a time anyway.

```python
def attention(Q, K, V, phase: str, window_size: int = 512):
    if phase == "prefill":
        # SWA during prefill
        mask = create_sliding_window_mask(len(K), window_size)
        return masked_attention(Q, K, V, mask)
    else:
        # Full attention during decode (single query)
        return full_attention(Q, K, V)
```

**Speedup**: 2-5x on prefill
**Quality Impact**: Minimal - decode still has full context access

### 2. Sink Tokens

Preserve first N tokens (typically 4-8) with full attention access.

**Rationale**: First tokens accumulate global information and act as "attention sinks" (StreamingLLM finding).

```python
def attention_with_sinks(Q, K, V, num_sinks: int = 4, window_size: int = 512):
    seq_len = K.shape[-2]
    mask = create_sliding_window_mask(seq_len, window_size)

    # Sink tokens attend to and are attended by ALL tokens
    mask[:num_sinks, :] = True  # Sinks can see everything
    mask[:, :num_sinks] = True  # Everything can see sinks

    return masked_attention(Q, K, V, mask)
```

**Speedup**: 3-4x
**Quality Impact**: Low - sinks preserve global information flow

### 3. Interleaved FA/SWA Layers

Alternate between full attention and sliding window across layers.

```python
def get_attention_type(layer_idx: int, num_layers: int) -> str:
    # First and last layers: full attention (embeddings/output)
    if layer_idx < 2 or layer_idx >= num_layers - 2:
        return "full"

    # Alternate middle layers
    return "window" if layer_idx % 2 == 1 else "full"
```

**Layer Pattern Example (24 layers)**:
```
Layer 0: Full (embedding proximity)
Layer 1: Full
Layer 2: Full
Layer 3: Window
Layer 4: Full
Layer 5: Window
...
Layer 22: Full
Layer 23: Full (output proximity)
```

**Speedup**: ~2x (half the layers are windowed)
**Quality Impact**: Minimal - full attention layers propagate long-range info

### 4. Chain-of-Thought (CoT) Prompting

For reasoning tasks, use CoT to break down problems, reducing effective context dependency.

**Not directly an architecture change**, but synergizes with SWA:
- CoT generates intermediate reasoning steps
- Each step depends on recent context
- Reduces need for attending to distant tokens

### 5. Fine-tuning

Short fine-tuning (1-5% of pretraining compute) on long-context data with SWA enabled.

```python
# Fine-tuning configuration
finetune_config = {
    "model": "bitnet-2b",
    "attention": "sliding_window",
    "window_size": 512,
    "num_sinks": 4,

    # Training
    "learning_rate": 1e-5,       # Low LR for stability
    "num_steps": 1000,           # ~1% of pretraining
    "batch_size": 8,
    "gradient_accumulation": 4,

    # Data
    "dataset": "long_context_mix",  # RedPajama-long, BookCorpus, etc.
    "max_length": 8192,
}
```

**Speedup**: Up to 100x (fully adapted)
**Quality Impact**: None after adaptation

## Combining Methods

SWAA recommends combining methods for best results:

### Configuration 1: No Training (Immediate)
```python
swaa_config = SWAAConfig(
    prefill_only=True,          # Method 1
    num_sinks=4,                # Method 2
    interleaved_layers=True,    # Method 3
    window_size=512,
)
# Expected: ~3x speedup, minimal quality loss
```

### Configuration 2: With Fine-tuning
```python
swaa_config = SWAAConfig(
    prefill_only=False,         # SWA for both phases
    num_sinks=4,
    interleaved_layers=False,   # All layers windowed
    window_size=512,
    finetuned=True,
)
# Expected: ~10x+ speedup, no quality loss
```

## Implementation for BitNet 2B

### Step 1: Add SWAA Config

```python
# packages/inference/src/wf_infer/swaa/config.py

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class AttentionType(Enum):
    FULL = "full"
    WINDOW = "window"

@dataclass
class SWAAConfig:
    """SWAA configuration for long-context inference."""

    enabled: bool = True
    window_size: int = 512
    num_sinks: int = 4

    # Method toggles
    prefill_only: bool = True           # Method 1
    use_sinks: bool = True              # Method 2
    interleaved_layers: bool = True     # Method 3

    # Layer-specific overrides
    full_attention_layers: List[int] = field(
        default_factory=lambda: [0, 1, -2, -1]  # First 2, last 2
    )

    def get_layer_attention_type(
        self,
        layer_idx: int,
        num_layers: int,
        phase: str = "prefill"
    ) -> AttentionType:
        """Determine attention type for a specific layer."""

        # Decode phase: always full attention (for prefill_only mode)
        if phase == "decode" and self.prefill_only:
            return AttentionType.FULL

        # Specified full attention layers
        resolved_layers = [
            l if l >= 0 else num_layers + l
            for l in self.full_attention_layers
        ]
        if layer_idx in resolved_layers:
            return AttentionType.FULL

        # Interleaved pattern
        if self.interleaved_layers:
            return AttentionType.WINDOW if layer_idx % 2 == 1 else AttentionType.FULL

        # Default: window
        return AttentionType.WINDOW
```

### Step 2: Create Attention Mask Generator

```python
# packages/inference/src/wf_infer/swaa/masks.py

import torch

def create_swaa_mask(
    seq_len: int,
    config: SWAAConfig,
    device: torch.device = None,
) -> torch.Tensor:
    """Create SWAA attention mask with sinks and sliding window."""

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # Sliding window
    for i in range(seq_len):
        start = max(0, i - config.window_size // 2)
        end = min(seq_len, i + config.window_size // 2 + 1)
        mask[i, start:end] = True

    # Sink tokens (if enabled)
    if config.use_sinks and config.num_sinks > 0:
        mask[:config.num_sinks, :] = True
        mask[:, :config.num_sinks] = True

    return mask

def create_causal_swaa_mask(
    seq_len: int,
    config: SWAAConfig,
    device: torch.device = None,
) -> torch.Tensor:
    """Create causal (autoregressive) SWAA mask."""
    mask = create_swaa_mask(seq_len, config, device)

    # Apply causal masking
    causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    mask = mask & ~causal.T  # Can only attend to previous positions

    return mask
```

### Step 3: Integrate into Attention

```python
# packages/inference/src/wf_infer/swaa/attention.py

def swaa_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    config: SWAAConfig,
    layer_idx: int,
    num_layers: int,
    phase: str = "prefill",
) -> torch.Tensor:
    """Compute attention with SWAA optimization."""

    attn_type = config.get_layer_attention_type(layer_idx, num_layers, phase)

    if attn_type == AttentionType.FULL:
        # Standard scaled dot-product attention
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    # Create SWAA mask
    seq_len = K.shape[-2]
    mask = create_causal_swaa_mask(seq_len, config, K.device)

    # Apply masked attention
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Apply mask (set masked positions to -inf)
    attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
    attn_weights = torch.softmax(attn_weights, dim=-1)

    return torch.matmul(attn_weights, V)
```

## Fine-tuning Protocol

If inference-only SWAA is insufficient:

### Data Requirements

```python
# Long-context training data sources
data_sources = [
    "cerebras/SlimPajama-627B",     # General text
    "emozilla/pg19",                 # Books (long docs)
    "EleutherAI/proof-pile-2",       # Math/code (long reasoning)
    "tau/scrolls",                   # Long-context QA
]

# Filter for long documents
def filter_long_docs(dataset, min_length=4096):
    return dataset.filter(
        lambda x: len(x["text"].split()) > min_length
    )
```

### Training Script

```python
# packages/distillation/scripts/swaa_finetune.py

import torch
from transformers import Trainer, TrainingArguments
from wf_infer.swaa import SWAAConfig, patch_model_with_swaa

def finetune_swaa(
    model_path: str,
    output_path: str,
    data_path: str,
    window_size: int = 512,
    num_sinks: int = 4,
    num_steps: int = 1000,
):
    # Load model
    model = load_bitnet_model(model_path)

    # Patch attention with SWAA
    swaa_config = SWAAConfig(
        window_size=window_size,
        num_sinks=num_sinks,
        prefill_only=False,  # Train with full SWAA
        interleaved_layers=False,  # All layers use window
    )
    model = patch_model_with_swaa(model, swaa_config)

    # Training config
    training_args = TrainingArguments(
        output_dir=output_path,
        max_steps=num_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_steps=100,
        save_steps=200,
        logging_steps=10,
        bf16=True,
    )

    # Load data
    dataset = load_long_context_data(data_path)

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # Save adapted model
    model.save_pretrained(output_path)
```

### Training on Cloud

```bash
# Using SkyPilot for distributed training
wf train \
    --script packages/distillation/scripts/swaa_finetune.py \
    --model bitnet-2b \
    --cloud nebius \
    --gpus A100:4 \
    --num-steps 1000
```

## Expected Results

| Configuration | Speedup | Quality (vs Full Attn) | Training Required |
|--------------|---------|------------------------|-------------------|
| Prefill-only SWA | 2-5x | ~99% | No |
| + Sink tokens | 3-4x | ~98% | No |
| + Interleaved layers | 2x | ~99% | No |
| Combined (no training) | 3-4x | ~97% | No |
| With fine-tuning | 10-100x | 100% | 1K steps |

## Evaluation

Test on long-context benchmarks:

```bash
# RULER benchmark (synthetic needle-in-haystack)
python eval/ruler.py --model bitnet-2b-swaa --context 8192

# LongBench (real-world long-context tasks)
python eval/longbench.py --model bitnet-2b-swaa

# Perplexity on long documents
python eval/ppl.py --model bitnet-2b-swaa --dataset pg19
```

## References

- SWAA Paper: https://arxiv.org/abs/2512.10411
- StreamingLLM (attention sinks): https://arxiv.org/abs/2309.17453
- LongLoRA (efficient fine-tuning): https://arxiv.org/abs/2309.12307
- Mistral 7B (production SWA): https://mistral.ai/news/announcing-mistral-7b/
