# Contributing to Deployer (wrinklefree-deployer)

> Contributor guide for navigating and understanding the deployer package codebase.

## Quick Orientation

### What This Package Does
Cloud training job orchestration using SkyPilot for managed GPU jobs across Nebius, RunPod, and GCP with spot recovery and WandB integration.

### Dependencies

| Depends On | What For |
|------------|----------|
| skypilot | Cloud job management, spot recovery |
| google-cloud-storage | GCS checkpoint upload |
| modal (optional) | Alternative cloud deployment |

| Orchestrates | What |
|--------------|------|
| `training` | Training jobs (all stages) |
| `inference` | Model serving |
| `eval` | Model evaluation |

---

## Codebase Architecture

### Directory Structure

```
packages/deployer/
├── src/wf_deploy/
│   ├── cli.py              # CLI entry point (`wf` command)
│   ├── core.py             # Main API: train(), logs(), serve()
│   ├── constants.py        # SCALES, GAR config, defaults
│   ├── config.py           # Configuration utilities
│   ├── credentials.py      # Credential management
│   ├── deployer.py         # Deployment utilities
│   ├── infra.py            # Infrastructure setup
│   ├── trainer.py          # Training orchestration
│   └── modal_deployer.py   # Modal alternative deployment
│
├── skypilot/
│   ├── train.yaml                      # Main training template
│   ├── service.yaml                    # Inference service template
│   ├── smoke_test_lightning.yaml       # Lightning + auto batch
│   ├── smoke_test_influence.yaml       # Influence remixing
│   ├── smoke_test_unified_1gpu.yaml    # 1x GPU unified
│   ├── smoke_test_unified_2gpu.yaml    # 2x GPU FSDP
│   ├── smoke_test_bitdistill.yaml      # BitDistill
│   └── smoke_test_lrc.yaml             # LRC calibration
│
└── credentials/
    ├── .env                            # Local credentials (gitignored)
    └── gcp-service-account.json        # GCP service account
```

### Key Components

| File | Purpose |
|------|---------|
| `cli.py` | `wf train`, `wf logs`, `wf serve` commands |
| `core.py` | Main `train()` function that builds SkyPilot tasks |
| `constants.py` | GPU scales, cloud defaults, GAR paths |
| `skypilot/train.yaml` | Template for training jobs |

---

## Code Flow

### wf train Command Flow

```
wf train -m qwen3_4b -s 2 --cloud nebius
│
├─► cli.py: train_command()
│   └─► Parse args, resolve model/stage
│
├─► core.py: train()
│   │
│   ├─► Build envs dict:
│   │   ├─► MODEL=qwen3_4b
│   │   ├─► STAGE=2
│   │   ├─► WANDB_API_KEY=...
│   │   └─► GCS credentials
│   │
│   ├─► Load train.yaml template
│   │
│   └─► sky.jobs.launch(task)
│       └─► SkyPilot handles:
│           - Cloud instance provisioning
│           - Spot recovery
│           - Log streaming
│
└─► Job runs on cloud:
    └─► train.yaml:run section
        └─► cd packages/training && uv run python scripts/train_lightning.py ...
```

### SkyPilot YAML Structure

```yaml
# skypilot/train.yaml
name: wrinklefree-train

resources:
  cloud: ${CLOUD}           # nebius, runpod, gcp
  accelerators: ${GPU}:${COUNT}  # L40:1, H100:4, etc.
  disk_size: 100

envs:
  MODEL: ${MODEL}           # smollm2_135m, qwen3_4b
  STAGE: ${STAGE}           # 1, 1.9, 2, 3
  WANDB_API_KEY: ${WANDB_API_KEY}

setup: |
  pip install uv
  cd /workspace && uv sync --all-packages

run: |
  cd packages/training
  uv run python scripts/train_lightning.py \
    model=${MODEL} \
    training=${TRAINING_CONFIG} \
    ...
```

---

## Entry Points

| Task | Start Here |
|------|------------|
| Add new CLI command | `cli.py` |
| Modify train job | `core.py:train()` and `skypilot/train.yaml` |
| Add new GPU scale | `constants.py:SCALES` |
| Add new cloud | `constants.py` cloud defaults, `core.py` cloud handling |
| Modify credentials | `credentials.py` |
| Add smoke test | Create new `skypilot/smoke_test_*.yaml` |

---

## Patterns & Conventions

### Environment Variable Pattern

SkyPilot jobs receive config via envs:
```python
# core.py
envs = {
    "MODEL": model,
    "STAGE": str(stage),
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
    "TRAINING_CONFIG": training_config,
}
task = sky.Task.from_yaml("skypilot/train.yaml")
task.update_envs(envs)
```

### Scale Definitions

```python
# constants.py
SCALES = {
    "dev": {"gpu": "L40", "count": 1, "cloud": "nebius"},
    "small": {"gpu": "L40", "count": 2, "cloud": "nebius"},
    "medium": {"gpu": "H100", "count": 2, "cloud": "nebius"},
    "large": {"gpu": "H100", "count": 4, "cloud": "nebius"},
}
```

### Credential Sourcing

```bash
# Always source before any cloud command
source credentials/.env

# .env contains:
export WANDB_API_KEY=...
export GOOGLE_APPLICATION_CREDENTIALS=credentials/gcp-service-account.json
```

---

## Testing

### Smoke Tests

```bash
cd packages/deployer
source credentials/.env

# Lightning (recommended)
sky launch skypilot/smoke_test_lightning.yaml -y --cluster lightning-smoke

# Check logs
sky logs lightning-smoke

# Teardown
sky down lightning-smoke -y
```

### Local Testing

```bash
# Dry run (shows what would be launched)
uv run --package wf-train-deployer wf train -m smollm2_135m -s 2 --dry-run

# Check CLI works
uv run --package wf-train-deployer wf --help
```

---

## Common Tasks

### Adding a New Smoke Test

1. Create `skypilot/smoke_test_mytest.yaml`
2. Define resources (GPU, cloud)
3. Set envs for training config
4. Add run section calling train_lightning.py
5. Document in CLAUDE.md

### Adding a New Cloud Provider

1. Add cloud config to `constants.py`
2. Update `core.py` to handle cloud-specific settings
3. Test with smoke test
4. Update documentation

### Modifying Training Job Template

1. Edit `skypilot/train.yaml`
2. Update envs if needed
3. Test with smoke test
4. Update `core.py` if new envs needed

---

## Gotchas & Tips

- **Run from deployer directory**: All `wf` and `sky` commands must run from `packages/deployer`. The YAML files use relative paths.

- **Source credentials first**: Always `source credentials/.env` before any cloud command. Missing credentials cause silent failures.

- **Never cancel others' jobs**: Only cancel SkyPilot jobs you started. Use `sky jobs queue` to see who owns what.

- **Clean before retry**: Before retrying failed jobs, run `sky exec <cluster> "rm -rf /tmp/checkpoints/*"` to clear stale state.

- **Prefer Nebius**: Nebius has better H100 availability than RunPod/GCP. Use `--cloud nebius`.

- **Disk size 100**: Always set `disk_size: 100` in YAML. Storage is cheap, running out causes failures.

- **Spot recovery**: SkyPilot handles spot interruption. Jobs resume from last GCS checkpoint.

- **WandB logging**: Training logs to WandB automatically when WANDB_API_KEY is set.
