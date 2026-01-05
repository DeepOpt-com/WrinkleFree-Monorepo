# Contributing to BitNet Architecture

> Contributor guide for navigating and understanding the wf-arch codebase.

## Quick Orientation

### What This Package Does
Core library providing 1.58-bit quantized components: BitLinear layers, SubLN normalization, and model conversion utilities.

### Dependencies

| Depends On | What For |
|------------|----------|
| torch | Tensor operations, nn.Module base |
| transformers | Model structure inspection for conversion |

| Used By | What For |
|---------|----------|
| `training` | BitLinear/SubLN for quantized training |
| `inference` | Model architecture definitions |

---

## Codebase Architecture

### Directory Structure

```
src/wf_arch/
├── __init__.py              # Public API - exports all public symbols
├── layers/
│   ├── __init__.py          # Layer exports
│   ├── bitlinear.py         # BitLinear: ternary weight quantization
│   ├── bitlinear_lrc.py     # BitLinearLRC: adds low-rank correction
│   └── subln.py             # SubLN/RMSNorm normalization
├── quantization/
│   ├── __init__.py          # Quantization exports
│   └── lambda_warmup.py     # Global quantization schedule state
└── conversion/
    ├── __init__.py          # Conversion exports
    └── convert.py           # Model transformation utilities
```

### Key Abstractions

| Class/Function | File | Purpose |
|----------------|------|---------|
| `BitLinear` | `layers/bitlinear.py` | Ternary weight quant (-1, 0, 1) with 8-bit activation quant |
| `BitLinearLRC` | `layers/bitlinear_lrc.py` | BitLinear + trainable low-rank correction (U, V matrices) |
| `SubLN` | `layers/subln.py` | Sub-layer RMSNorm placed before projections |
| `LambdaWarmup` | `quantization/lambda_warmup.py` | Global singleton managing quantization strength (0→1) |
| `convert_model_to_bitnet()` | `conversion/convert.py` | Replaces nn.Linear with BitLinear, inserts SubLN |

---

## Code Flow

### Forward Pass (BitLinear)

```
BitLinear.forward(x)
│
├─► weight_quant(self.weight)
│   └─► scale = 1/mean(|W|)
│   └─► W_q = round(clip(W*scale, -1, 1)) / scale  [STE gradient]
│
├─► activation_quant(x)
│   └─► gamma = max(|x|) per token
│   └─► x_q = round(clip(127*x/gamma, -128, 127)) * gamma/127
│
└─► F.linear(x_q, W_q, bias)
```

### Model Conversion

```
convert_model_to_bitnet(model)
│
├─► Iterate all nn.Linear modules
│   └─► Replace with BitLinear (copy weights)
│
├─► Insert SubLN before q_proj, k_proj, v_proj, o_proj
│   └─► Wraps projection in SubLN(proj)
│
└─► Return converted model (same structure, BitNet layers)
```

### Lambda Warmup (Global State)

```
Training loop:
│
├─► set_global_lambda_warmup(LambdaWarmup(1000))
│
└─► Each step:
    ├─► warmup.step()  # Increment internal counter
    └─► get_current_lambda()  # Returns 0.0 → 1.0
        └─► Used by BitLinear to interpolate quant strength
```

---

## Entry Points

| Task | Start Here |
|------|------------|
| Modify weight quantization | `layers/bitlinear.py:weight_quant()` |
| Modify activation quantization | `layers/bitlinear.py:activation_quant()` |
| Add new layer type | Create in `layers/`, export from `layers/__init__.py` |
| Change conversion logic | `conversion/convert.py:convert_model_to_bitnet()` |
| Modify warmup schedule | `quantization/lambda_warmup.py:LambdaWarmup` |

---

## Patterns & Conventions

### Straight-Through Estimator (STE)

All quantization uses STE for gradient flow. The pattern:
```python
# Forward: use quantized value
# Backward: gradient flows through as if no quantization
w_quant = (w * scale).round().clamp(-1, 1) / scale
# PyTorch autograd treats round() and clamp() as identity for gradients
```

### Global Singleton Pattern (Lambda)

Lambda warmup is global state accessed via functions:
```python
from wf_arch import set_global_lambda_warmup, get_current_lambda

# Set once at training start
set_global_lambda_warmup(LambdaWarmup(warmup_steps=1000))

# Access anywhere (thread-safe read)
lambda_val = get_current_lambda()  # 0.0 to 1.0
```

### Module Export Pattern

All public API goes through `__init__.py`:
```python
# In layers/__init__.py
from wf_arch.layers.bitlinear import BitLinear
__all__ = ["BitLinear", ...]

# In root __init__.py
from wf_arch.layers import BitLinear
__all__ = ["BitLinear", ...]
```

---

## Testing

### Running Tests

```bash
# All tests
uv run --package wf-arch pytest packages/architecture/tests/ -v

# Specific test
uv run --package wf-arch pytest packages/architecture/tests/test_bitlinear.py -v

# With coverage
uv run --package wf-arch pytest packages/architecture/tests/ --cov=wf_arch
```

### Test Organization

| File | What's Tested |
|------|---------------|
| `test_bitlinear.py` | BitLinear forward/backward, quantization correctness |
| `test_conversion.py` | Model conversion, SubLN insertion |
| `test_lambda_warmup.py` | Warmup schedule, global state |

---

## Common Tasks

### Adding a New Quantization Method

1. Create `layers/my_layer.py` with your layer class
2. Export from `layers/__init__.py`
3. Export from root `__init__.py`
4. Add tests in `tests/test_my_layer.py`
5. Update CLAUDE.md with usage examples

### Modifying Weight Quantization

1. Edit `BitLinear.weight_quant()` in `layers/bitlinear.py`
2. Ensure gradients still flow (STE pattern)
3. Test with `pytest tests/test_bitlinear.py`
4. Test integration: run training smoke test

### Adding Conversion Support for New Model

1. Check model structure in `conversion/convert.py:convert_model_to_bitnet()`
2. Add model-specific projection names if needed
3. Test with the model: `convert_model_to_bitnet(AutoModel.from_pretrained(...))`

---

## Gotchas & Tips

- **Global Lambda State**: `get_current_lambda()` returns the same value across all BitLinear instances. Changing warmup affects all layers simultaneously.

- **LRC Freezing**: When using `BitLinearLRC`, the base `weight` tensor is frozen. Only `lrc_U` and `lrc_V` are trainable. Call `freeze_model_except_lrc()` to ensure this.

- **SubLN Placement**: SubLN goes *before* projections (q_proj, k_proj, etc.), not after. This is critical for training stability.

- **Test Both Packages**: Changes here affect `training` package. Always run training smoke test after modifications:
  ```bash
  uv run --package wf-train python packages/training/scripts/train_lightning.py \
    model=smollm2_135m training=base training.max_steps=10
  ```

- **Bias Default**: BitLinear defaults to `bias=False` (standard for BitNet). Don't change without reason.

- **Numerical Stability**: The `eps` parameter prevents division by zero in quantization. Don't remove it.
