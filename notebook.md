# LRC Training Fixes - 2026-01-07

## Summary
Fixed multiple issues blocking LRC (Low-Rank Correction) training on H100.

**UPDATE**: torch.compile and gradient checkpointing now WORK with LRC after the `enable_input_require_grads()` fix!

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

### 3. torch.compile NOW WORKS with LRC
**Previous Status**: Disabled - thought to break gradient flow

**Root Cause Analysis**: The issue was NOT torch.compile itself, but frozen embeddings breaking gradient flow. After `enable_input_require_grads()` fix, torch.compile works.

**Current Config** (ENABLED):
```yaml
torch_compile:
  enabled: true  # NOW WORKS after enable_input_require_grads fix
  mode: default
  fullgraph: false
```

**Speedup**: ~38% faster training on H100

### 4. Gradient Checkpointing NOW WORKS with LRC
**Previous Status**: Disabled - broke gradient flow

**Root Cause Analysis**: The issue was `use_reentrant=True` (old default) not preserving requires_grad state during recomputation. Fixed by using `use_reentrant=False`.

**Fix**: Updated `train_lightning.py` to use `use_reentrant=False`:
```python
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

**Current Config** (ENABLED):
```yaml
memory:
  gradient_checkpointing: true  # NOW WORKS with use_reentrant=False
```

**Benefit**: Saves significant VRAM, allows larger batch sizes

### 5. OOM with Large Batch Size
**Symptom**: CUDA out of memory on H100 with batch_size=32

**Root Cause**:
- 25% LRC rank = 161M trainable params (too many)
- Logits tensor: batch_size * seq_len * vocab_size = 16 * 1024 * 152064 = ~10GB

**Fix**: Reduced LRC rank and increased batch size with gradient checkpointing:
```yaml
lrc:
  rank_percentage: 0.15  # Down from 0.25 (96M vs 161M params)

# With gradient checkpointing enabled, can use larger batches
batch_size: 16  # Up from 8 (gradient checkpointing saves VRAM)
gradient_accumulation_steps: 32  # Effective batch = 512
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
| `packages/training/scripts/train_lightning.py` | Added unbuffered stdout, `enable_input_require_grads()`, gradient checkpointing with `use_reentrant=False`, torch.compile support |
| `packages/training/configs/training/lrc_run.yaml` | ENABLED torch.compile and gradient checkpointing; batch_size=16, rank=15% |
| `packages/deployer/skypilot/lrc_run.yaml` | Added PYTHONUNBUFFERED=1 |
| `packages/architecture/tests/test_bitlinear_lrc.py` | Added tests for torch.compile and gradient checkpointing compatibility |

## GitHub Issues
- **#38**: LDC-MTL + LRC instability (still open)
- **#39**: torch.compile and gradient checkpointing break gradient flow with LRC layers - **FIXED**

## Current Status
Training running on Nebius H100 with FULL optimizations:
- torch.compile: ENABLED (38% speedup)
- Gradient checkpointing: ENABLED (use_reentrant=False)
- Batch size: 16 (doubled from 8)
- Trainable params: 96.5M (15% LRC rank)
- Total params: 692M
- WandB: https://wandb.ai/umd-leans-well/wrinklefree_v2/runs/lrc_run_qwen3_0.6b-438521f5

## Lessons Learned
1. When freezing model params, always call `enable_input_require_grads()` to ensure gradient flow
2. torch.compile WORKS with LRC after the enable_input_require_grads fix
3. Gradient checkpointing WORKS with LRC when using `use_reentrant=False`
4. LRC rank percentage directly affects trainable params - 25% was too high for memory
5. Always set PYTHONUNBUFFERED for cloud training to see logs in real-time

## Key Code Changes

### train_lightning.py - Gradient Checkpointing Fix
```python
# Use use_reentrant=False for compatibility with LRC (frozen embeddings)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

### train_lightning.py - torch.compile Support
```python
torch_compile_cfg = cfg.training.get("torch_compile", {})
if torch_compile_cfg.get("enabled", False):
    compile_mode = torch_compile_cfg.get("mode", "default")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torch_compile_cache")
    model = torch.compile(model, mode=compile_mode, fullgraph=False)
```
