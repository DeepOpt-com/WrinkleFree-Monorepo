# LRC Training Fixes - 2026-01-07

## Summary
Fixed multiple issues blocking LRC (Low-Rank Correction) training on H100.

## Issues Fixed

### 1. Gradient Flow Bug (CRITICAL)
**Symptom**: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**Root Cause**: When embeddings are frozen (via `freeze_model_except_lrc()`), the embedding output has `requires_grad=False`. This breaks gradient flow even though LRC U/V matrices have `requires_grad=True`.

**Fix**: Added `model.enable_input_require_grads()` call after freezing LRC model.
```python
# In packages/training/scripts/train_lightning.py
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
```

**Reference**: Gemini analysis identified this as "Gradient Checkpointing Trap" pattern.

### 2. LDC-MTL Instability with LRC
**Symptom**: Loss going up, val loss flat-lining

**Root Cause**: LDC-MTL meta-optimization was destabilizing objective weights with LRC training.

**Fix**: Disabled LDC-MTL in `lrc_run.yaml`:
```yaml
meta_optimization:
  enabled: false
  ldc_mtl:
    enabled: false  # Disabled - investigate instability with LRC
```

**Tracking Issue**: GitHub #38

### 3. torch.compile Breaks LRC
**Symptom**: `RuntimeError: element 0 of tensors does not require grad`

**Root Cause**: torch.compile doesn't handle custom autograd functions/STE operations in BitLinearLRC correctly.

**Fix**: Disabled torch.compile in `lrc_run.yaml`:
```yaml
torch_compile:
  enabled: false  # DISABLED: Breaks gradient flow with LRC custom layers
```

**Tracking Issue**: GitHub #39 (also includes gradient checkpointing)

### 4. Gradient Checkpointing Breaks LRC
**Symptom**: Same gradient flow error

**Root Cause**: Gradient checkpointing + frozen embeddings = no gradient graph.

**Fix**: Disabled gradient checkpointing in `lrc_run.yaml`:
```yaml
memory:
  gradient_checkpointing: false  # DISABLED: May break gradient flow with LRC
```

### 5. OOM with Large Batch Size
**Symptom**: CUDA out of memory on H100 with batch_size=32

**Root Cause**:
- 25% LRC rank = 161M trainable params (too many)
- Logits tensor: batch_size * seq_len * vocab_size = 16 * 1024 * 152064 = ~10GB

**Fix**: Reduced batch_size and LRC rank:
```yaml
lrc:
  rank_percentage: 0.15  # Down from 0.25 (96M vs 161M params)

batch_size: 8  # Down from 32
gradient_accumulation_steps: 64  # Maintain effective batch = 512
```

### 6. Python Output Buffering
**Symptom**: Logs not appearing in SkyPilot output during training

**Fix**: Added unbuffered stdout at top of `train_lightning.py`:
```python
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
```

Also added `PYTHONUNBUFFERED=1` to `skypilot/lrc_run.yaml`.

## Files Modified

| File | Changes |
|------|---------|
| `packages/training/scripts/train_lightning.py` | Added unbuffered stdout, `enable_input_require_grads()` |
| `packages/training/configs/training/lrc_run.yaml` | Disabled torch.compile, LDC-MTL, gradient checkpointing; reduced batch_size to 8, rank to 15% |
| `packages/deployer/skypilot/lrc_run.yaml` | Added PYTHONUNBUFFERED=1 |

## GitHub Issues Created
- **#38**: LDC-MTL + LRC instability
- **#39**: torch.compile and gradient checkpointing break gradient flow with LRC layers

## Current Status
Training running on Nebius H100:
- GPU utilization: 93%
- Memory usage: 72GB / 81GB
- Trainable params: 96.5M
- Total params: 692M
- WandB: https://wandb.ai/umd-leans-well/wrinklefree_v2/runs/lrc_run_qwen3_0.6b-7f0d4236

## Lessons Learned
1. When freezing model params, always call `enable_input_require_grads()` to ensure gradient flow
2. torch.compile and gradient checkpointing don't work with custom LRC layers - need to disable for LRC training
3. LRC rank percentage directly affects trainable params - 25% was too high for memory
4. Always set PYTHONUNBUFFERED for cloud training to see logs in real-time
