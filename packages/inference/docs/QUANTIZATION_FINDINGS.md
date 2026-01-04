# BitNet Quantization Findings

## Summary

This document describes the investigation into why our checkpoints produce garbage output while Microsoft's official BitNet works correctly.

## TL;DR

**The issue is NOT the conversion pipeline - it's undertrained checkpoints.**

| Model | Python Output | C++ Output | Status |
|-------|---------------|------------|--------|
| Microsoft BitNet 2B | N/A (custom loader needed) | "Paris is a city known for..." | ✅ Works |
| Our 2B step3600 | ",,,,,,,,,,," | "GGGGGGG..." | ❌ Undertrained |
| distill-final | Never evaluated | N/A | ❌ global_step=0 |
| bitdistill-stage3-final | Never evaluated | N/A | ❌ global_step=100 |

## Latest Findings (2026-01-04)

### 1. Microsoft BitNet Works Correctly

```bash
# C++ inference produces sensible output
./llama-cli -m microsoft-bitnet-2b/ggml-model-i2_s.gguf -p "The capital of France is" -n 30 --temp 0
# Output: "The capital of France is Paris. Paris is a city that is known for its rich history, culture, and architecture."
```

### 2. Our Checkpoints are Undertrained

**2B step3600 checkpoint**: Produces garbage even in Python!
```python
# Python inference (no GGUF conversion)
prompt = "The capital of France is"
response = "The capital of France is,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"
```

**Other local checkpoints**:
- `distill-final`: `global_step=0`, `best_eval_loss=inf` - never trained
- `bitdistill-stage3-final`: `global_step=100` - only 100 steps

### 3. Checkpoint Training Status

With unified config (batch_size=32, grad_accum=16, seq_len=2048):
- ~1M tokens/step
- Step 3600 = 3.6B tokens = 36% through 10B token target
- At 36%, curriculum is in "main" phase with `continue_pretrain=1.0`

**But**: The checkpoint still produces garbage, suggesting either:
1. Training didn't actually use CE loss (continue_pretrain)
2. Some other training issue
3. Checkpoint was corrupted

### 4. Config Comparison

| Config | Our Checkpoint | Microsoft BitNet |
|--------|---------------|------------------|
| `architectures` | `BitNetForCausalLM` | `BitNetForCausalLM` |
| `quantization_mode` | `online` (bf16 weights) | `offline` (uint8 packed) |
| `hidden_size` | 2560 | 2560 |
| `num_hidden_layers` | 30 | 30 |
| `vocab_size` | 128257 | 128256 |

---

## Previous Findings

## Key Findings

### 1. BitLinear Does Both Weight AND Activation Quantization

The BitLinear layer quantizes both:
- **Weights**: `w_quant = (w * scale).round().clamp(-1, 1) / scale` where `scale = 1/absmean`
- **Activations**: `x_quant = (x * 127/absmax).round().clamp(-128, 127) * absmax/127`

Pre-quantizing weights alone is not sufficient for correct inference.

### 2. Weight Quantization Halves the Frobenius Norm

| Metric | Original | Quantized |
|--------|----------|-----------|
| Frobenius Norm | 162 | 82 |
| Weight Range | [-4.75, 5.22] | [-0.188, 0.188] |
| AbsMean | 0.188 | 0.108 |

This 2x reduction in weight magnitude compounds through 30 layers.

### 3. Pre-quantized Weights Cause EOS Token Dominance

| Model Configuration | EOS Probability |
|---------------------|-----------------|
| Original (no quant) | 0.00006 |
| BitLinear (weight + act quant) | 0.049 |
| Pre-quantized (weight only) | **0.670** |

The 10,000x increase in EOS probability causes the model to generate garbage.

### 4. Activation Quantization is Critical

BitLinear's activation quantization helps normalize the outputs, preventing the EOS token from dominating. Without it, the reduced weight magnitudes cause numerical instability.

## Root Cause

The model in `models/smollm2-135m-dlm-subln/` was NOT trained with BitLinear quantization. It's a standard model with:
- Full-precision weights
- SubLN layers (60 additional norm layers)
- DLM (Diffusion Language Model) architecture

When we apply quantization:
1. Weight magnitudes drop by ~2x
2. Hidden state magnitudes drop proportionally
3. After 30 layers, the output distribution is completely different
4. EOS token becomes dominant → garbage output

## Recommendations

### Option 1: Train with BitLinear from Start (Recommended)

Train the model with `lambda=1.0` from the beginning so weights learn to work in their quantized form.

```python
warmup = LambdaWarmup(warmup_steps=0, min_lambda=1.0, max_lambda=1.0)
set_global_lambda_warmup(warmup)
# Train model...
```

### Option 2: Implement Activation Quantization in C++

Add activation quantization to llama.cpp:
```cpp
// Per-token activation quantization
float absmax = 0.0f;
for (int i = 0; i < n; i++) absmax = std::max(absmax, std::abs(x[i]));
float scale = 127.0f / std::max(absmax, 1e-5f);
for (int i = 0; i < n; i++) {
    x[i] = std::round(x[i] * scale) / scale;
    x[i] = std::clamp(x[i], -128.0f/scale, 127.0f/scale);
}
```

### Option 3: Use DLM Block Diffusion

The `dlm_server` uses block diffusion decoding which may be more robust to weight quantization. Test with:
```bash
./dlm_server --model-path model.gguf --decode-mode adaptive
```

## Files Created

| File | Purpose |
|------|---------|
| `scripts/convert_with_quantization.py` | Applies BitLinear weight quantization |
| `scripts/debug_conversion.py` | Compares quantization methods |
| `scripts/test_python_inference.py` | Tests different model configurations |
| `scripts/test_correct_inference.py` | Demonstrates correct vs incorrect pipelines |
| `scripts/test_kernel_correctness.py` | Validates I2_S kernel math |
| `scripts/benchmark_and_validate.py` | Compares Python vs C++ output |

## Verification Commands

```bash
# Test BitLinear quantization
uv run --package wrinklefree python packages/inference/scripts/test_correct_inference.py

# Debug weight conversion
uv run python packages/inference/scripts/debug_conversion.py models/smollm2-135m-dlm-subln

# Test kernel correctness
uv run python packages/inference/scripts/test_kernel_correctness.py
```

## Next Steps

### Immediate Priority: Get a Properly Trained Checkpoint

1. **Check WandB logs** for the 2B training run to verify CE loss was enabled
2. **Train longer** - step 3600 may be too early for quality output
3. **Use Microsoft BitNet** as reference for cosine similarity testing

### Python vs C++ Cosine Similarity Testing

Once we have a working checkpoint:

```bash
# 1. Load model in Python, get logits for test prompt
# 2. Convert to GGUF with I2_S
# 3. Run same prompt in llama-cli, extract logits
# 4. Compare cosine similarity

# Blocker: llama-cpp-python (pip) doesn't support I2_S quantization
# Solution: Either build from our custom llama.cpp or use logit extraction from llama-cli
```

### Technical Notes

- **llama-cpp-python from pip**: Does NOT support I2_S quantization (BitNet-specific)
- **Our llama.cpp**: Has I2_S support, but Python bindings need custom build
- **Microsoft BitNet loader**: Requires `1bitllm` library or custom `configuration_bitnet.py`/`modeling_bitnet.py`
