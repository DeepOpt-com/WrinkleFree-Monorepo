# Architecture

This document describes the architecture of the WrinkleFree DLM Converter.

## Overview

The converter transforms BitNet 1.58-bit quantized models into Diffusion Language Models (DLMs) that support parallel token generation within blocks.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   BitNet Model  │ ──► │  Block Adapter  │ ──► │    DLM Model    │
│  (1.58-bit AR)  │     │  (Attention +   │     │  (Parallel Gen) │
│                 │     │   Noise Embed)  │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   Fine-tuning   │
                        │   (~1B tokens)  │
                        └─────────────────┘
```

## Components

### 1. Model Loader (`models/loader.py`)

Loads BitNet checkpoints from multiple sources:
- **Local paths**: `/path/to/checkpoint`, `./checkpoint`
- **HuggingFace Hub**: `hf://organization/model-name`
- **Modal volumes**: `/checkpoints/model_name`

Key functions:
- `load_bitnet_checkpoint()` - Main entry point
- `validate_bitnet_model()` - Verify ternary weights
- `extract_model_config()` - Get architecture details

### 2. Block Diffusion Adapter (`models/adapter.py`)

Modifies the model architecture for block diffusion:

#### Block-wise Causal Attention

Standard autoregressive attention only allows tokens to see previous tokens:

```
Position:  0  1  2  3
Token 0:  [1, 0, 0, 0]
Token 1:  [1, 1, 0, 0]
Token 2:  [1, 1, 1, 0]
Token 3:  [1, 1, 1, 1]
```

Block causal attention (block_size=2) allows tokens to see their entire block:

```
Position:  0  1  2  3
Token 0:  [1, 1, 0, 0]  ← Sees all of block 0
Token 1:  [1, 1, 0, 0]  ← Sees all of block 0
Token 2:  [1, 1, 1, 1]  ← Sees blocks 0 and 1
Token 3:  [1, 1, 1, 1]  ← Sees blocks 0 and 1
```

This enables parallel generation within each block.

#### Noise Embedding

The adapter adds a noise embedding layer that encodes the current diffusion timestep:

```python
class NoiseEmbedding(nn.Module):
    def __init__(self, num_steps, hidden_size):
        self.step_embedding = nn.Embedding(num_steps, hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
```

#### Token Shift

The token shift mechanism allows predicting masked tokens from preceding token logits, maintaining some autoregressive properties:

```python
def token_shift(logits, target_positions):
    # Predict position i from logits at position i-1
    shifted = F.pad(logits[:, :-1], (0, 0, 1, 0))
    return shifted
```

### 3. Diffusion Fine-Tuner (`conversion/training.py`)

Fine-tunes the adapted model with diffusion objectives:

#### Training Recipe

Based on Fast-dLLM v2:
- **Total tokens**: ~1B (500x less than training from scratch)
- **Learning rate**: 5e-5 (moderate to preserve BitLinear)
- **Batch size**: 8 with 8x gradient accumulation
- **Effective batch**: 32,768 tokens per step

#### Masked Token Prediction

Each training step:
1. Randomly mask ~15% of tokens (or use complementary mask)
2. Forward pass with block causal attention
3. Compute cross-entropy loss on masked positions
4. Apply token shift for within-block predictions

#### Complementary Training

Alternates which tokens are masked across steps to ensure all positions receive supervision:

```python
def create_complementary_mask(seq_len, step, num_steps):
    positions = torch.arange(seq_len)
    pattern = step % num_steps
    return (positions % num_steps) == pattern
```

### 4. Checkpoint Manager (`conversion/checkpoint.py`)

Saves models in Fast-dLLM compatible format:

```
output/
├── config.json           # HuggingFace model config
├── model.safetensors     # Model weights
├── tokenizer/            # Tokenizer files
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── dlm_config.json       # Block diffusion parameters
```

The `dlm_config.json` contains:
```json
{
    "block_size": 32,
    "num_diffusion_steps": 8,
    "noise_schedule": "cosine",
    "source_model": "qwen3_4b",
    "is_bitnet": true
}
```

### 5. Modal Deployer (`modal/deployer.py`)

Runs conversion jobs on Modal with GPU acceleration:

- **GPU**: H100 (default)
- **Timeout**: 24 hours
- **Volumes**:
  - `/checkpoints` - BitNet source checkpoints
  - `/hf_cache` - HuggingFace cache
  - `/outputs` - Converted DLM models

## Data Flow

```
1. User calls: wf-dlm convert -m qwen3_4b -c hf://org/model

2. CLI parses args and calls core.convert()

3. core.convert() decides backend:
   - Modal: Launches remote job via modal/deployer.py
   - Local: Runs conversion in-process

4. Conversion pipeline:
   a. Load BitNet checkpoint (models/loader.py)
   b. Apply block diffusion adapter (models/adapter.py)
   c. Fine-tune with diffusion objectives (conversion/training.py)
   d. Save DLM checkpoint (conversion/checkpoint.py)

5. Result returned to user with output path
```

## Configuration

### Hydra Structure

```
configs/
├── config.yaml           # Main config with defaults
├── model/
│   ├── smollm2_135m.yaml # Small model config
│   └── qwen3_4b.yaml     # Production model config
└── conversion/
    └── finetune.yaml     # Training hyperparameters
```

### Key Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `block_size` | `config.yaml` | 32 | Tokens per block |
| `num_diffusion_steps` | `config.yaml` | 8 | Steps per block |
| `total_tokens` | `conversion/finetune.yaml` | 1B | Fine-tuning budget |
| `learning_rate` | `conversion/finetune.yaml` | 5e-5 | Optimizer LR |

## Preserving BitLinear Quantization

The fine-tuning is designed to preserve the ternary weight distribution from BitNet:

1. **Moderate learning rate** (5e-5): Prevents large weight updates
2. **Short training** (~1B tokens): Minimal deviation from source
3. **Weight monitoring**: Track unique values to verify ternary pattern

During conversion, we avoid:
- Freezing BitLinear layers entirely (they need slight adaptation)
- Too high learning rates (would destroy ternary structure)
- Too long training (would deviate from source behavior)
