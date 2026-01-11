# Stage 1.9 Training on 8x H100 - Findings

**Date:** 2025-12-26
**Model:** Qwen3-4B
**Stage:** 1.9 (Layer-wise Distillation)
**Infrastructure:** Nebius 8x H100-80GB ($23.60/hr)

## Summary

Successfully launched Stage 1.9 training on 8x H100 after fixing several issues:
- W&B API key not being passed to remote jobs
- Wrong data config (downstream instead of fineweb)
- OOM errors with default batch sizes
- Missing distributed config for multi-GPU

## Issues Found & Fixed

### 1. WANDB_API_KEY Not Passed to Remote Jobs

**Problem:** Training failed with "WANDB_API_KEY environment variable is not set"

**Fix:** Updated `core.py` to pass secrets from local environment:
```python
# Pass through secrets from local environment
# FAIL LOUDLY if W&B key is not set - training will fail anyway
wandb_key = os.environ.get("WANDB_API_KEY")
if not wandb_key:
    raise RuntimeError(
        "WANDB_API_KEY not set! Training requires W&B logging.\n"
        "Fix: export WANDB_API_KEY=your_key  (or add to ~/.config/.env.global)"
    )
envs["WANDB_API_KEY"] = wandb_key
```

### 2. Wrong Data Config

**Problem:** Default config used `data: downstream` (GLUE SST-2) instead of pretraining data

**Fix:** Updated `train.yaml` to set data config based on stage:
```bash
case $STAGE in
  1)   DATA_CONFIG="" ;;
  1.9) DATA_CONFIG="data=fineweb" ;;
  2)   DATA_CONFIG="data=fineweb" ;;
  3)   DATA_CONFIG="data=downstream" ;;
esac
```

### 3. OOM with Default Batch Sizes

**Problem:** Qwen3-4B OOM'd on H100-80GB even with batch_size=8

Stage 1.9 loads BOTH teacher and student models:
- Teacher (Qwen3-4B): ~8GB in bfloat16
- Student (Qwen3-4B): ~8GB in bfloat16
- Plus activations, gradients, optimizer states

**Working config:**
- `batch_size=4` (per GPU)
- `gradient_accumulation_steps=16`
- Effective batch size: 4 × 16 × 8 GPUs = 512

### 4. Missing Distributed Config for Multi-GPU

**Problem:** `train.yaml` hardcoded `distributed=single_gpu`

**Fix:** Updated `core.py` to set distributed config based on GPU count:
```python
distributed_config = "fsdp_multi" if gpu_count > 1 else "single_gpu"
base_overrides = [f"distributed={distributed_config}"]
```

## GPU Configuration Notes

**Nebius H100 availability:**
- H100:1 ✓
- H100:2 ✗ (not available)
- H100:4 ✗ (not available)
- H100:8 ✓

For 2/4 GPU configs, use RunPod or GCP instead.

## Working Command

```bash
# Set environment variables first
export WANDB_API_KEY=your_key
source .venv/bin/activate

# Launch Stage 1.9 on 8x H100
wf train -m qwen3_4b -s 1.9 --scale xlarge \
  training.batch_size=4 \
  training.gradient_accumulation_steps=16 \
  training.max_steps=100
```

Or use the dedicated YAML:
```bash
sky jobs launch skypilot/qwen3_4b_stage1_9_8xh100.yaml
```

## W&B Runs

- **Run:** https://wandb.ai/umd-leans-well/wrinklefree/runs/qo582ssa
- **Progress:** 50/100 steps at time of documentation
- **Loss:** 6.0137
- **LR:** 3.0e-3 (Muon optimizer)

## Files Changed

| File | Change |
|------|--------|
| `src/wf_deploy/core.py` | Added WANDB_API_KEY passthrough, fail-loud check, distributed config |
| `skypilot/train.yaml` | Added data config selection by stage, removed hardcoded distributed |
| `skypilot/qwen3_4b_stage1_9_8xh100.yaml` | NEW: Dedicated YAML for 8x H100 Stage 1.9 |
| `docs/quick-start.md` | Updated setup instructions for SkyPilot-only |

## TODO

- [ ] Add `TORCHINDUCTOR_CACHE_DIR` to train.yaml for faster torch.compile on subsequent runs
- [ ] Investigate why some runs get stuck (heartbeat active but no step progress)
- [ ] Add GPU profile configs that auto-set batch sizes based on model + stage + GPU
