# Training Stability Improvements for BitNet + DLM

**Date**: 2025-12-29
**Status**: Implemented

## Problem

Training BitNet models (both Stage 2 pretraining and DLM conversion) exhibited:
- Loss spikes during training
- Poor convergence
- Instability in both DLM and Stage 2 training

## Root Cause Analysis

### 1. Mask Token Initialization (DLM-specific, Critical)

**Issue**: The `|<MASK>|` token used in DLM training was initialized with random embeddings.

**Why it matters for BitNet**:
- BitNet uses per-token activation quantization: `scale = 127 / max(|x|)`
- Random embeddings have different scale/distribution than trained vocabulary
- When `|<MASK>|` appears (frequent in DLM), it creates activation outliers
- Outliers force quantization scale to be very small → all other signal collapses to zero
- Result: Gradient spikes and poor convergence

**Solution**: Initialize `|<MASK>|` embedding to the **mean of existing vocabulary**:
```python
mean_embedding = input_embeddings.weight[:limit_idx].mean(dim=0)
input_embeddings.weight[mask_id] = mean_embedding
```

### 2. Fixed Gradient Clipping

**Issue**: Using a fixed gradient clipping threshold (e.g., 1.0) doesn't adapt to:
- Natural gradient scale of different models
- Evolving gradient dynamics during training
- True anomalies vs normal large gradients

**Solution**: ZClip - Adaptive gradient clipping using z-score anomaly detection.

## Implemented Solutions

### ZClip (Adaptive Gradient Clipping)

ZClip dynamically adjusts the clipping threshold based on gradient norm statistics:

1. Maintains an EMA (exponential moving average) of gradient norms
2. Computes a z-score for each gradient norm
3. Only clips when z-score exceeds threshold (default: 3.0 = ~0.3% of updates if normally distributed)

**Benefits**:
- Adapts to natural gradient scale for each model
- Only clips true anomalies, not normal large gradients
- Prevents both under-clipping (spikes cause instability) and over-clipping (slows convergence)

**Usage**:
```python
from wf_data.training import ZClip

zclip = ZClip(z_threshold=3.0, ema_decay=0.99)

for batch in dataloader:
    loss.backward()
    stats = zclip.clip(model)
    # stats.raw_norm, stats.clipped_norm, stats.was_clipped, stats.z_score
    optimizer.step()
```

**Reference**: [ZClip: Adaptive Spike Mitigation for LLM Pre-Training](https://arxiv.org/abs/2504.02507)

### Lambda Warmup (Gradual Quantization)

For BitNet training, gradually ramps up quantization strength:
- Starts at λ=0 (full precision)
- Linearly increases to λ=1 (full quantization)
- Prevents catastrophic forgetting of pre-trained knowledge

**Already integrated in**:
- `wf_train.quantization.lambda_warmup.LambdaWarmup`
- Stage 2 training (`stage2.py`)

**Now also integrated in**:
- DLM training (`train_dlm.py`) via `quantization_warmup_steps` config

**Usage**:
```yaml
# In config
conversion:
  quantization_warmup_steps: 500  # 5-10% of total steps recommended
```

### Data Mixing Improvements

**Larger shuffle buffer**: Increased from 10,000 to 50,000 for better mixing of 5+ sources.

**Reduced code weight**: Code datasets (e.g., `github_code`) are often noisy. Reduced from 15% to 10%, redistributed to higher-quality educational content.

## Files Modified

| File | Changes |
|------|---------|
| `packages/data_handler/src/wf_data/training/gradient_clipping.py` | ZClip implementation |
| `packages/data_handler/src/wf_data/training/__init__.py` | Export ZClip |
| `packages/training/src/wf_train/training/` | ZClip integration |
| `packages/data_handler/configs/data/mixed_pretrain.yaml` | shuffle_buffer_size, adjusted weights |

## W&B Metrics

New metrics logged:
- `train/grad_norm_raw`: Gradient norm before clipping
- `train/grad_norm_clipped` / `train/grad_norm`: Gradient norm after clipping
- `train/grad_clipped`: 1.0 if clipping was applied, 0.0 otherwise
- `train/lambda`: Current quantization lambda value
- `train/zclip_ema_mean`: ZClip's EMA mean (for monitoring)
- `train/zclip_ema_std`: ZClip's EMA standard deviation

## Configuration

### DLM Training (`train_dlm.py`)

```yaml
conversion:
  quantization_warmup_steps: 500  # Lambda warmup steps (0 = disabled)
```

ZClip is enabled by default with z_threshold=3.0.

### Stage 2 Training (`stage2.py`)

```yaml
training:
  lambda_warmup:
    enabled: true
    warmup_steps: 1000
    schedule: linear  # or cosine

  zclip:
    enabled: true
    z_threshold: 3.0
    ema_decay: 0.99
```

### Data Mixing (`mixed_pretrain.yaml`)

```yaml
shuffle_buffer_size: 50000  # Increased from default 10000

sources:
  - name: github_code
    weight: 0.10  # Reduced from 0.15
  - name: fineweb_edu
    weight: 0.35  # Increased from 0.30
```

## Expected Outcomes

1. **Smart mask init** → Removes activation outliers from quantization
2. **ZClip** → Proactive spike mitigation, adapts to training dynamics
3. **Lambda warmup** → Gradual quantization constraint introduction
4. **Larger shuffle buffer** → Better mixing, smoother gradients
5. **Adjusted weights** → Fewer noisy samples causing spikes

## References

- [ZClip: Adaptive Spike Mitigation for LLM Pre-Training](https://arxiv.org/abs/2504.02507) - 2025
- [AdaGC: Adaptive Gradient Clipping](https://arxiv.org/abs/2502.11034) - 2025
- [Spike No More: Stabilizing the Pre-training of Large Language Models](https://arxiv.org/abs/2312.16903) - 2024
- [HuggingFace BitNet 1.58-bit Fine-tuning Blog](https://huggingface.co/blog/1_58_llm_extreme_quantization)
