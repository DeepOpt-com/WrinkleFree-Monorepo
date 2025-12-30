# WrinkleFree Architecture

> Part of [WrinkleFree Monorepo](https://github.com/DeepOpt-com/WrinkleFree-Monorepo) - BitNet layers and model conversion utilities.

Core building blocks for 1.58-bit (ternary) quantized LLM models.

## Features

- **BitLinear**: Ternary weight quantization {-1, 0, 1} with 8-bit activation quantization
- **SubLN**: Sub-Layer Normalization for stable BitNet training
- **LambdaWarmup**: Gradual quantization schedule management
- **Model Conversion**: Convert any HuggingFace model to BitNet on-the-fly

## Installation

```bash
# From monorepo root
uv sync --package bitnet-arch

# Or install all packages
uv sync --all-packages
```

## Quick Start

### BitLinear Layer

```python
from bitnet_arch import BitLinear

# Create a quantized linear layer
layer = BitLinear(in_features=768, out_features=768)

# Forward pass automatically quantizes weights to {-1, 0, 1}
output = layer(input_tensor)
```

### Model Conversion

```python
from bitnet_arch import convert_model_to_bitnet, auto_convert_if_needed
from transformers import AutoModelForCausalLM

# Load any HuggingFace model
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Convert to BitNet (replaces nn.Linear with BitLinear, adds SubLN)
bitnet_model = convert_model_to_bitnet(model)
```

### Lambda Warmup (Gradual Quantization)

```python
from bitnet_arch import LambdaWarmup, set_global_lambda_warmup, get_current_lambda

# Gradually introduce quantization over 1000 steps
warmup = LambdaWarmup(warmup_steps=1000)
set_global_lambda_warmup(warmup)

for step in range(total_steps):
    warmup.step()
    # lambda goes from 0.0 (no quantization) to 1.0 (full quantization)
```

## API Reference

### Layers

| Component | Description |
|-----------|-------------|
| `BitLinear` | Linear layer with ternary weights and 8-bit activation quantization |
| `BitLinearNoActivationQuant` | BitLinear without activation quantization |
| `SubLN` | Sub-layer RMSNorm for training stability |
| `RMSNorm` | Root Mean Square Layer Normalization |

### Conversion

| Function | Description |
|----------|-------------|
| `convert_model_to_bitnet(model)` | Full model conversion to BitNet |
| `auto_convert_if_needed(model)` | Convert only if not already BitNet |
| `is_bitnet_model(model)` | Check if model uses BitLinear layers |
| `run_stage1(model)` | Stage 1 SubLN insertion |

### Quantization Schedule

| Function | Description |
|----------|-------------|
| `LambdaWarmup(warmup_steps)` | Create warmup schedule |
| `set_global_lambda_warmup(warmup)` | Set global warmup instance |
| `get_current_lambda()` | Get current lambda value (0→1) |

## How It Works

### BitLinear Quantization

BitLinear quantizes weights to ternary values {-1, 0, 1} using:

```
W_q = sign(W) * round(|W| / α)
```

Where α is a scaling factor. Gradients flow through via Straight-Through Estimator (STE).

### SubLN for Stability

SubLN adds RMSNorm before output projections in attention and FFN blocks. This prevents gradient explosion during quantized training.

## Integration with Training

The training package uses this library for BitNet model creation:

```python
from bitnet_arch import convert_model_to_bitnet, LambdaWarmup

# Stage 1: Convert model
model = convert_model_to_bitnet(base_model)

# Stage 2+: Training with gradual quantization
warmup = LambdaWarmup(warmup_steps=1000)
```

## Testing

```bash
# Run all tests
uv run --package bitnet-arch pytest packages/architecture/tests/

# With coverage
uv run --package bitnet-arch pytest --cov=bitnet_arch
```

## References

- [BitDistill Paper](https://arxiv.org/abs/2510.13998) - 1.58-bit training approach
- [BitNet Paper](https://arxiv.org/abs/2310.11453) - Ternary weight quantization
- [Microsoft BitNet](https://github.com/microsoft/BitNet) - Inference engine
