# WrinkleFree Monorepo

1.58-bit quantized LLM research using uv workspaces.

## CRITICAL Rules (MUST FOLLOW)

1. **CHECK SYNC BEFORE REMOTE:** Before ANY ssh/remote command, check if live sync is running:
   ```bash
   ./sync.sh --status --preset <preset>  # Returns JSON with running status
   ```
   - If `"running": true` → Files are auto-syncing, proceed with remote commands
   - If `"running": false` → Start sync first: `./sync.sh --preset <preset>`

2. **NO GCP GPU:** Use Nebius or RunPod only (GCP is expensive)

3. **NEVER CANCEL OTHERS' JOBS:** Only cancel SkyPilot jobs you started in this session

4. **READ PACKAGE CLAUDE.md FIRST:** Before modifying any package, read its `packages/<pkg>/CLAUDE.md`

5. **USE I2_S FOR GGUF:** NEVER use TQ2_0 for bf16 DLM checkpoints - produces garbage

## Quick Commands

| Task | Command |
|------|---------|
| **Training (Lightning)** | `uv run --package wf-train python scripts/train_lightning.py model=smollm2_135m training=base` |
| Training with auto batch | `... training.auto_batch_size=true` (single GPU only!) |
| Meta-optimization | `... training.meta_optimization.enabled=true` |
| BitDistill distillation | `... training=bitdistill_full` |
| LRC calibration | `... training=lrc_calibration` |
| Deploy to cloud | `cd packages/deployer && wf train -m smollm2_135m -s 2 --cloud nebius` |
| **Check sync status** | `./sync.sh --status --preset <preset>` |
| **Start live sync** | `./sync.sh --preset <preset>` (runs in foreground with inotify) |
| One-time sync | `./sync.sh --preset <preset> --no-watch` |
| Setup creds only | `./sync.sh --setup-creds --preset <preset>` |
| Run tests | `uv run pytest packages/<pkg>/tests/` |
| Type check | `uv run mypy packages/<pkg>/src/` |

## Dos and Don'ts

### DO
- Use PyTorch Lightning trainer (`train_lightning.py`) for new training runs
- Use `training.auto_batch_size=true` to auto-find optimal batch size (single GPU only - not supported with DDP/FSDP!)
- Check WandB for training metrics: https://wandb.ai/umd-leans-well/wrinklefree
- Use `sky exec <cluster> <yaml>` to re-run jobs on existing clusters
- Clean `/tmp/checkpoints/` on remote before re-running smoke tests

### DON'T
- Don't use TQ2_0 for bf16 checkpoints (destroys ternary weight distribution)
- Don't push to main without PR review
- Don't run `sky down` on clusters you didn't create

## Package Navigation

| Package | Namespace | Purpose | Key Files |
|---------|-----------|---------|-----------|
| `training` | `wf_train` | Training pipeline + Lightning | `scripts/train_lightning.py`, `src/wf_train/lightning/` |
| `architecture` | `wf_arch` | BitLinear/BitLinearLRC/SubLN layers | `src/wf_arch/layers/` |
| `data_handler` | `wf_data` | Data loading + mixing | `src/wf_data/data/` |
| `deployer` | `wf_deploy` | Cloud deployment (SkyPilot) | `skypilot/*.yaml` |
| `inference` | `wf_infer` | Model serving | `src/wf_infer/` |
| `mobile` | N/A | Android inference (C++/JNI) | `android/` |
| `eval` | `wf_eval` | Model evaluation | `src/wf_eval/` |
| `math-utils` | `wf_math` | Pure math utilities | `src/wf_math/` |

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PyTorch Lightning                        │
│  train_lightning.py → WrinkleFreeLightningModule           │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           ObjectiveManager (multi-task)              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │ CE Loss  │  │   DLM    │  │   LRC    │  ...     │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  Callbacks: BatchSizeFinder, GCS, ZClip, TokenCount        │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Meta-Optimization (optional)              │   │
│  │  ┌──────────────┐    ┌──────────────┐              │   │
│  │  │  LDC-MTL     │    │  ODM/EXP3    │              │   │
│  │  │  (obj wts)   │    │  (data wts)  │              │   │
│  │  └──────────────┘    └──────────────┘              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Key Config Overrides

```bash
# Model selection
model=smollm2_135m          # 135M params, good for testing
model=qwen3_4b              # 4B params, production

# Training configs
training=base            # Combined CE + DLM (recommended)
training=bitdistill_full    # Knowledge distillation
training=lrc_calibration    # Low-rank correction

# Common overrides
training.max_steps=100      # Limit steps for testing
training.auto_batch_size=true  # Auto-find max batch size (single GPU only!)
output_dir=/tmp/checkpoints
gcs.enabled=true gcs.bucket=wrinklefree-checkpoints
```

## GGUF Conversion (IMPORTANT)

**NEVER use TQ2_0 for bf16 DLM checkpoints - it produces garbage output!**

| Format | Size | Speed | Quality | Notes |
|--------|------|-------|---------|-------|
| **I2_S** | 1.1GB | 55 tok/s | Good | **RECOMMENDED** - best compatibility |
| TQ1_0 | 678MB | 63 tok/s | Good | Smaller, may conflict with some llama.cpp builds |
| TQ2_0 | 779MB | 76 tok/s | GARBAGE | Do NOT use for bf16! |

```bash
# Correct conversion workflow (I2_S recommended)
python packages/inference/scripts/convert_checkpoint_to_gguf.py \
    path/to/checkpoint --outfile model.gguf --outtype i2_s
```

## Reference

- Detailed pipeline docs: `docs/ai-code/reference.md`
- Architecture overview: `docs/architecture.md`
- Development guide: `docs/development.md`
- Quick start: `docs/quick-start.md`
