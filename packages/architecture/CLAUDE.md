# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

BitNet Architecture (`bitnet-arch`) is a shared library providing 1.58-bit quantized components:
- **BitLinear**: Ternary weight quantization (-1, 0, 1) with 8-bit activation quantization
- **BitLinearLRC**: BitLinear with Low-Rank Correction for post-quantization recovery
- **SubLN**: Sub-Layer Normalization for stable BitNet training
- **LambdaWarmup**: Gradual quantization schedule manager
- **Conversion**: Utilities to convert standard models to BitNet on-the-fly

## Monorepo Integration

This is a **shared library** imported by other packages:

```
architecture (this package)
    │
    └──► packages/training (wrinklefree)
            Uses: bitnet_arch.layers, bitnet_arch.conversion
```

**Workspace dependency** (in consumer's pyproject.toml):
```toml
[project]
dependencies = ["bitnet-arch"]

[tool.uv.sources]
bitnet-arch = { workspace = true }
```

**Related packages**:
| Package | Relationship |
|---------|--------------|
| `training` | Uses BitLinear, SubLN for quantized training |
| `data_handler` | Sibling library (data loading) |

## Quick Start

```bash
# From monorepo root
uv sync --all-packages

# Run tests
uv run --package bitnet-arch pytest packages/architecture/tests/

# Import in Python
from bitnet_arch import BitLinear, SubLN, LambdaWarmup
from bitnet_arch import convert_model_to_bitnet, auto_convert_if_needed
```

## Key Components

### Layers (`bitnet_arch.layers`)

| Component | Purpose |
|-----------|---------|
| `BitLinear` | Linear layer with ternary weights, 8-bit activation quant |
| `BitLinearNoActivationQuant` | BitLinear without activation quantization |
| `BitLinearLRC` | BitLinear with trainable low-rank correction (U, V matrices) |
| `SubLN` | Sub-layer RMSNorm for training stability |
| `RMSNorm` | Root Mean Square Layer Normalization |
| `convert_linear_to_bitlinear` | Replace nn.Linear with BitLinear |
| `convert_bitlinear_to_lrc` | Replace BitLinear with BitLinearLRC (adds U, V, freezes weights) |
| `freeze_model_except_lrc` | Freeze all params except LRC matrices |
| `get_lrc_stats` | Get statistics about LRC layers in a model |

### Quantization (`bitnet_arch.quantization`)

| Component | Purpose |
|-----------|---------|
| `LambdaWarmup` | Gradual quantization warmup schedule |
| `get_global_lambda_warmup` | Access global warmup instance |
| `set_global_lambda_warmup` | Set global warmup instance |
| `get_current_lambda` | Get current quantization lambda (0→1) |

### Conversion (`bitnet_arch.conversion`)

| Component | Purpose |
|-----------|---------|
| `convert_model_to_bitnet` | Full model conversion to BitNet |
| `is_bitnet_model` | Check if model is already BitNet |
| `auto_convert_if_needed` | Convert only if not already BitNet |
| `run_stage1` | Stage 1 SubLN insertion |

## Architecture

```
src/bitnet_arch/
├── __init__.py              # Public API exports
├── layers/
│   ├── __init__.py
│   ├── bitlinear.py         # BitLinear implementation
│   ├── bitlinear_lrc.py     # BitLinearLRC with low-rank correction
│   └── subln.py             # SubLN/RMSNorm
├── quantization/
│   ├── __init__.py
│   └── lambda_warmup.py     # Quantization schedule
└── conversion/
    ├── __init__.py
    └── convert.py           # Model conversion utilities
```

## Usage Examples

### BitLinear Layer

```python
from bitnet_arch import BitLinear

# Create a quantized linear layer
layer = BitLinear(in_features=768, out_features=768)

# Forward pass (weights quantized to {-1, 0, 1})
output = layer(input_tensor)
```

### Model Conversion

```python
from bitnet_arch import convert_model_to_bitnet, auto_convert_if_needed
from transformers import AutoModelForCausalLM

# Load a standard model
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Convert to BitNet (replaces nn.Linear with BitLinear, adds SubLN)
bitnet_model = convert_model_to_bitnet(model)

# Or auto-convert only if needed
bitnet_model = auto_convert_if_needed(model)
```

### Lambda Warmup (Optional, Disabled by Default)

Lambda warmup gradually increases quantization strength from 0 to 1. When disabled (default), `get_current_lambda()` returns 1.0 and the model trains with full quantization from step 0.

```python
from bitnet_arch import LambdaWarmup, set_global_lambda_warmup, get_current_lambda

# Without warmup (default): get_current_lambda() returns 1.0
# Full ternary quantization from the start

# To enable gradual warmup:
warmup = LambdaWarmup(warmup_steps=1000)
set_global_lambda_warmup(warmup)

# In training loop
for step in range(total_steps):
    warmup.step()  # Update lambda
    lambda_val = get_current_lambda()  # 0.0 → 1.0
    # ... training code uses lambda_val for quantization strength
```

## Testing

```bash
# Run all tests
uv run --package bitnet-arch pytest packages/architecture/tests/

# Run specific test file
uv run --package bitnet-arch pytest packages/architecture/tests/test_bitlinear.py -v

# With coverage
uv run --package bitnet-arch pytest packages/architecture/tests/ --cov=bitnet_arch
```

### Low-Rank Correction (LRC)

Based on [Low-Rank Correction for Quantized LLMs](https://arxiv.org/abs/2412.07902).

```python
from bitnet_arch import BitLinearLRC, convert_bitlinear_to_lrc, freeze_model_except_lrc

# Convert existing BitNet model to LRC (freezes weights, adds U/V matrices)
model = convert_bitlinear_to_lrc(
    model,
    rank_percentage=0.1,  # 10% of min(in, out) per layer
    init_method="zeros",  # or "svd_residual" for better init
)

# Ensure only LRC params are trainable
freeze_model_except_lrc(model)

# Forward pass: output = W_quant @ Q_a(X) + U @ V^T @ X
# - W_quant @ Q_a(X): frozen quantized path
# - U @ V^T @ X: trainable low-rank correction on unquantized activations
output = layer(x)
```

**Key principle**: Only `lrc_U` and `lrc_V` are trainable. All other params (quantized weights, bias, embeddings, norms) are **frozen**.

## Notes

- This is a **pure library** - no CLI or scripts
- Changes affect the training package - test both after modifications
- BitLinear uses Straight-Through Estimator (STE) for gradients
- SubLN is critical for training stability (prevents gradient explosion)
- BitLinearLRC freezes quantized weights; only U, V matrices are trainable

## References

- [BitDistill Paper](https://arxiv.org/abs/2510.13998) - 1.58-bit training approach
- [BitNet Paper](https://arxiv.org/abs/2310.11453) - Ternary weight quantization
- [LRC Paper](https://arxiv.org/abs/2412.07902) - Low-Rank Correction for quantized LLMs
