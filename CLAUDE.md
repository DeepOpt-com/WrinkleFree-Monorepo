# WrinkleFree Monorepo

1.58-bit quantized LLM research using uv workspaces.

## CRITICAL Rules (MUST FOLLOW)

1. **SYNC BEFORE REMOTE:** Before ANY ssh/remote command:
   `uv run gcd sync-ssh <host> --smart`

2. **NO GCP GPU:** Use Nebius or RunPod only (GCP is expensive)

3. **NEVER CANCEL OTHERS' JOBS:** Only cancel SkyPilot jobs you started in this session

4. **READ PACKAGE CLAUDE.md FIRST:** Before modifying any package, read its `packages/<pkg>/CLAUDE.md`

5. **USE I2_S FOR GGUF:** NEVER use TQ2_0 for bf16 DLM checkpoints - produces garbage

## Quick Commands

| Task | Command |
|------|---------|
| **Training (Lightning)** | `uv run --package wrinklefree python scripts/train_lightning.py model=smollm2_135m training=unified` |
| Training with auto batch | `... training.auto_batch_size=true` |
| BitDistill distillation | `... training=bitdistill_full` |
| LRC calibration | `... training=lrc_calibration` |
| Deploy to cloud | `cd packages/deployer && wf train -m smollm2_135m -s 2 --cloud nebius` |
| Sync to Desktop | `uv run gcd sync-ssh desktop --smart` |
| Watch mode sync | `uv run gcd sync-ssh desktop --watch` |
| Run tests | `uv run pytest packages/<pkg>/tests/` |
| Type check | `uv run mypy packages/<pkg>/src/` |

## Dos and Don'ts

### DO
- Use PyTorch Lightning trainer (`train_lightning.py`) for new training runs
- Use `training.auto_batch_size=true` to auto-find optimal batch size
- Check WandB for training metrics: https://wandb.ai/umd-leans-well/wrinklefree
- Use `sky exec <cluster> <yaml>` to re-run jobs on existing clusters
- Clean `/tmp/checkpoints/` on remote before re-running smoke tests

### DON'T
- Don't use TQ2_0 for bf16 checkpoints (destroys ternary weight distribution)
- Don't push to main without PR review
- Don't run `sky down` on clusters you didn't create

## Package Navigation

| Package | Purpose | Key Files |
|---------|---------|-----------|
| `training` | Training pipeline + Lightning | `scripts/train_lightning.py`, `src/wrinklefree/lightning/` |
| `architecture` | BitLinear/BitLinearLRC/SubLN layers | `src/bitnet_arch/layers/` |
| `data_handler` | Data loading + influence | `src/data_handler/data/` |
| `deployer` | Cloud deployment (SkyPilot) | `skypilot/*.yaml` |
| `inference` | Model serving | `src/wrinklefree_inference/` |
| `mobile` | Android inference | `android/` |
| `eval` | Model evaluation | `src/wrinklefree_eval/` |

> **Legacy:** `packages/_legacy/` contains archived packages (distillation, converter, cheapertraining)

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
└─────────────────────────────────────────────────────────────┘
```

## Key Config Overrides

```bash
# Model selection
model=smollm2_135m          # 135M params, good for testing
model=qwen3_4b              # 4B params, production

# Training configs
training=unified            # Combined CE + DLM (recommended)
training=bitdistill_full    # Knowledge distillation
training=lrc_calibration    # Low-rank correction

# Common overrides
training.max_steps=100      # Limit steps for testing
training.auto_batch_size=true  # Auto-find max batch size
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
