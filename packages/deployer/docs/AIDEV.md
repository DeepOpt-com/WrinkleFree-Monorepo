# AI Developer Guide (AIDEV.md)

This document provides a comprehensive guide to the WrinkleFree-Deployer codebase for AI assistants and developers.

## 1. Project Overview

WrinkleFree-Deployer is a **launcher** for training 1.58-bit quantized LLMs. It does NOT contain training code itself - it launches training jobs on Modal or SkyPilot infrastructure.

### Relationship to Other Repos

```
WrinkleFree-Deployer (This Repo - Launcher)
    │
    ├─► Modal Backend
    │       ├─ Clones WrinkleFree-CheaperTraining (optimization library)
    │       ├─ Clones WrinkleFree-1.58Quant (training code)
    │       └─ Runs: python scripts/train.py
    │
    └─► SkyPilot Backend
            ├─ Mounts: ../training (workdir)
            └─ Runs: uv run python scripts/train.py
```

## 2. Key Files

### Constants (`src/wf_deployer/constants.py`)

**This is the single source of truth for all configuration values.** Before adding any hardcoded string, check if it exists here.

Contains:
- `MODAL_APP_NAME`, `MODAL_VOLUME_*` - Modal identifiers
- `REPO_*` - GitHub repository URLs
- `DEFAULT_*` - Default values (data, wandb project, context size)
- `TRAINING_TIMEOUT`, `SMOKE_TEST_TIMEOUT` - Timeouts in seconds
- `RunIdPrefix` - Enum for run ID prefixes (`modal-`, `sky-`)
- `STAGE_CONFIG_MAP` - Maps stage numbers to Hydra config names
- `EnvVars` - Centralized environment variable names

### Core API (`src/wf_deployer/core.py`)

Main functions for training:
- `train(model, stage, backend, scale, overrides, detach)` - Launch training
- `logs(run_id, follow)` - View logs
- `cancel(run_id)` - Cancel job
- `list_runs(backend, limit)` - List recent runs
- `smoke_test(model, backend)` - Quick pipeline test

### Modal Backend (`src/wf_deployer/modal_deployer.py`)

Modal-specific implementation:
- `verify_gpu_allocation(requested_type)` - **Verifies GPU type matches request** (hard failure on mismatch)
- `run_training(...)` - Core training function (runs on Modal)
- `smoke_test(model)` - Quick validation
- `ModalTrainer` - High-level class for AI tools

### CLI (`src/wf_deployer/cli.py`)

Click-based CLI commands:
- `wf train`, `wf logs`, `wf cancel`, `wf runs`, `wf smoke`
- `wf serve`, `wf serve-down`, `wf serve-status`
- `wf wandb-status`

## 3. Run ID Format

Run IDs are prefixed to self-document the backend:

| Format | Backend | Example |
|--------|---------|---------|
| `modal-{model}-s{stage}` | Modal | `modal-qwen3_4b-s2` |
| `sky-{model}-s{stage}:{job_id}` | SkyPilot | `sky-qwen3_4b-s2:123` |
| `fc-{id}` | Modal (legacy) | `fc-abc123def` |

Detection logic in `core.py:logs()` and `core.py:cancel()`:
```python
if run_id.startswith(RunIdPrefix.SKYPILOT.value):  # "sky-"
    _logs_skypilot(run_id, follow)
elif run_id.startswith(RunIdPrefix.MODAL.value):   # "modal-"
    _logs_modal(run_id, follow)
elif run_id.startswith(RunIdPrefix.MODAL_LEGACY.value):  # "fc-"
    _logs_modal(run_id, follow)
elif ":" in run_id:  # Legacy SkyPilot format
    _logs_skypilot(run_id, follow)
else:
    _logs_modal(run_id, follow)
```

## 4. GPU Verification

At training start, we verify the allocated GPU matches the request:

```python
# In modal_deployer.py
def verify_gpu_allocation(requested_type: str) -> dict:
    import torch
    actual_name = torch.cuda.get_device_name(0)

    if requested_type.lower() not in actual_name.lower():
        raise RuntimeError(f"GPU MISMATCH: Requested {requested_type} but got {actual_name}")

    return {"gpu_name": actual_name, "gpu_memory_gb": ..., "verified": True}
```

This is a **hard failure** - training aborts if wrong GPU allocated. The return value includes:
- `gpu_actual`: Actual GPU name (e.g., "NVIDIA H100 80GB HBM3")
- `gpu_memory_gb`: Memory in GB
- `gpu_verified`: True if verification passed

## 5. Stage-to-Config Mapping

Training stages map to Hydra config files in `packages/training/configs/training/`:

| Stage | Config | Purpose |
|-------|--------|---------|
| 1 | `stage1_subln` | Convert FP16 to initial 1.58-bit (SubLN initialization) |
| 1.9 | `stage1_9_layerwise` | Layer-wise distillation to refine conversion |
| 2 | `stage2_pretrain` | Pre-training on large corpus (main training) |
| 3 | `stage3_distill` | Distillation from teacher (final refinement) |

Defined in `constants.STAGE_CONFIG_MAP`.

## 6. Scale Profiles

GPU scale profiles (defined in `modal_deployer.py:SCALES`):

| Scale | GPUs | Profile | Use Case |
|-------|------|---------|----------|
| `dev` | 1x A10G | `a10g_24gb` | Testing, cheap |
| `small` | 1x H100 | `h100_80gb` | Single GPU training |
| `medium` | 2x H100 | `h100_80gb` | Default for qwen3_4b |
| `large` | 4x H100 | `h100_80gb` | Fast training |
| `xlarge` | 8x H100 | `h100_80gb` | Maximum speed |

Model-specific defaults in `MODEL_SCALES`:
- `smollm2_135m` → `dev`
- `qwen3_4b` → `medium`

## 7. Environment Variables

Key env vars (see `.env.example` and `constants.EnvVars`):

| Variable | Purpose |
|----------|---------|
| `WANDB_ENTITY` | W&B user/team (no longer hardcoded) |
| `WANDB_API_KEY` | W&B authentication |
| `GH_TOKEN` | GitHub token for private repos |
| `MODAL_SCALE` | Default scale for Modal deployment |

## 8. Common Tasks

### Launching Training
```python
from wf_deployer import train
run_id = train("qwen3_4b", stage=2)
run_id = train("qwen3_4b", stage=2, scale="large")  # 4x H100
run_id = train("qwen3_4b", stage=2, overrides=["training.lr=1e-4"])
```

### Checking GPU Allocation
After launch, check logs for:
```
[GPU Verify] Requested: H100
[GPU Verify] Allocated: NVIDIA H100 80GB HBM3 (80.0 GB)
[GPU Verify] SUCCESS - GPU type verified
```

If mismatch:
```
RuntimeError: GPU MISMATCH: Requested H100 but got Tesla T4. Aborting.
```

### Adding New Model Support
1. Add entry to `MODEL_SCALES` in `modal_deployer.py`:
   ```python
   MODEL_SCALES = {
       "new_model": "medium",  # or appropriate scale
   }
   ```
2. Ensure model config exists in `packages/training/configs/model/`

### Adding New Constants
1. Add to `src/wf_deployer/constants.py`
2. Import where needed
3. Never hardcode the value directly in other files

## 9. Testing

```bash
# Unit tests
uv run pytest tests/unit/ -v

# Smoke test on Modal
wf smoke -m smollm2_135m

# Verify constants load
python -c "from wf_deployer.constants import MODAL_APP_NAME; print('OK')"
```

## 10. Architecture Notes

- Modal is the default backend (simpler, auto-scales)
- SkyPilot provides multi-cloud support with spot instances
- Checkpoints persist via Modal volumes or cloud storage (S3/GCS)
- Training code uses fingerprint-based run IDs for auto-resume
