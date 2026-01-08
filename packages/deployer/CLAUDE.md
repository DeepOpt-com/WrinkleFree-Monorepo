# WrinkleFree-Deployer

Training job launcher for 1.58-bit quantized LLMs. Uses SkyPilot for managed GPU jobs with spot recovery.

## CRITICAL Rules

1. **RUN FROM DEPLOYER DIR**: All `wf` and `sky` commands must run from `packages/deployer`
2. **SOURCE CREDENTIALS FIRST**: Always `source credentials/.env` before any cloud command
3. **NEVER CANCEL OTHERS' JOBS**: Only cancel SkyPilot jobs you started in this session
4. **USE NEBIUS**: Prefer Nebius over RunPod/GCP (better availability, lower cost)
5. **CLEAN BEFORE RETRY**: Run `sky exec <cluster> "rm -rf /tmp/checkpoints/*"` before retrying failed jobs
6. **FIX, DON'T FALL BACK**: When something breaks, FIX IT. Don't silently fall back to alternatives (e.g., don't switch from MuonClip to AdamW - fix MuonClip)
7. **ALWAYS SET GCS PROJECT**: When creating SkyPilot YAMLs, ALWAYS include `GOOGLE_CLOUD_PROJECT: wrinklefree-481904` in envs section. The gcp-service-account.json is OAuth user creds (not service account), so the project ID must be set explicitly for GCS uploads to work.
8. **USE TRAINING CONFIGS**: Use `--training/-t` flag (not `--stage/-s`). Training configs in `packages/training/configs/training/` are the source of truth.

## Quick Commands

```bash
cd packages/deployer
source credentials/.env

# Training with specific config
wf train -m qwen3_4b -t base                       # Combined CE + DLM
wf train -m qwen3_4b -t bitdistill_full --scale large  # BitDistill
wf train -m smollm2_135m -t lrc_run                # Low-Rank Correction
wf train -m qwen3_4b -t base --dry-run             # Preview without launching

# Smoke tests
wf smoke                          # Default: dlm on L40S
wf smoke -o bitdistill            # BitDistill smoke test
wf smoke -o lrc --gpu-type H100   # LRC on H100
wf smoke --dry-run                # Preview without launching

# Check logs
wf logs <job_name>
wf runs  # List recent jobs
```

## Monorepo Integration

This package is the **orchestrator** for the WrinkleFree monorepo - it launches jobs for all other packages.

**Orchestrates**:
| Command | Package | Description |
|---------|---------|-------------|
| `wf train` | `training` | 1.58-bit quantization training (all stages) |
| `wf serve` | `inference` | Model serving |
| `wf eval` | `eval` | Model evaluation |

> **Note**: Distillation is now done via training objectives (`training=bitdistill_full` or `training=lrc_calibration`).

**Running commands**:
```bash
# IMPORTANT: wf commands must be run from packages/deployer directory
# (train.yaml uses relative paths that require this)
cd packages/deployer
source credentials/.env
uv run --package wf-train-deployer wf train -m qwen3_4b -s 2
```

## Training Configs

| Config | Purpose | Use Case |
|--------|---------|----------|
| `base` | Combined CE + DLM | Default training (recommended) |
| `bitdistill_full` | Knowledge distillation | Stage 3 distillation |
| `lrc_run` | Low-Rank Correction | Post-quant error recovery |
| `salient_run` | AWQ-style salient | Selective precision |
| `sft_run` | Supervised fine-tuning | Instruction-following |
| `smoke_test` | Fast 30-step validation | CI/CD testing |

## Smoke Test Objectives

| Objective | Description |
|-----------|-------------|
| `ce` | Cross-entropy only |
| `dlm` | CE + DLM (default) |
| `bitdistill` | BitDistill distillation |
| `lrc` | Low-Rank Correction |
| `salient` | AWQ-style salient columns |
| `meta_opt` | Meta-optimization (LDC-MTL + ODM) |

## Key Files

| File | Purpose |
|------|---------|
| `src/wf_deploy/constants.py` | All magic strings, defaults, scales, training configs |
| `src/wf_deploy/core.py` | Main API: train(), smoke_test_unified(), logs() |
| `src/wf_deploy/cli.py` | CLI commands: train, smoke, logs, runs |
| `skypilot/train.yaml` | SkyPilot training job template |
| `skypilot/smoke_test.yaml` | **Unified smoke test (use `wf smoke -o <objective>`)** |
| `skypilot/service.yaml` | SkyServe inference template |
| `skypilot/eval.yaml` | Model evaluation template |
| `scripts/dispatch_smoke.py` | Maps objective to training config + overrides |
| `credentials/.env` | Local credentials (gitignored) |
| `credentials/gcp-service-account.json` | GCP service account for GCS + Docker auth |
| `skypilot/_archived/` | Archived old YAMLs (for reference) |

## Smoke Tests

Quick validation of the training pipeline (~5 minutes):

```bash
cd packages/deployer
source credentials/.env

# Run smoke test with specific objective
wf smoke                          # Default: dlm on L40S
wf smoke -o bitdistill            # BitDistill distillation
wf smoke -o lrc --gpu-type H100   # LRC on H100
wf smoke -o meta_opt --gpu-count 2  # Meta-opt with 2 GPUs

# Preview without launching
wf smoke --dry-run

# Monitor
sky logs wf-smoke-dlm

# Teardown
sky down wf-smoke-dlm -y
```

**Test Configuration**:
- **Steps**: 20 total (4 warmup + 16 main training)
- **First 20%**: Warmup on fineweb-edu
- **Remaining 80%**: Mixed data training
- **Checkpoints**: GCS upload every 10 steps
- **Verifies**: Loss decreases, MuonClip works, GCS/WandB logging

**Expected Results**:
- First loss: ~10-12
- Last loss: ~6-8 (should decrease!)
- Checkpoints in GCS: step_10/, step_20/, final/

## Credential Management

The deployer uses SkyPilot's `secrets` feature for secure credential passing:

```yaml
# In train.yaml
secrets:
  WANDB_API_KEY: null           # Required - passed via --secret
  SKYPILOT_DOCKER_PASSWORD: null  # For Docker auth on non-GCP clouds
```

The CLI (`wf train`) auto-prepares Docker credentials for Nebius/RunPod.

## GCS Setup (Required for SkyPilot)

SkyPilot uses GCS for file mounts. **gsutil must be configured separately from gcloud** because standalone gsutil doesn't share gcloud credentials.

**Auto-setup** (recommended): Just run `source credentials/.env` - it auto-configures gsutil if needed.

**Manual setup**:
```bash
# Run from packages/deployer
./scripts/setup-gcs.sh
```

This creates `~/.boto` pointing to the service account. Verify with:
```bash
gsutil ls gs://wrinklefree-checkpoints/
```

**If gsutil fails with 401 errors**: The `.boto` file is missing or misconfigured. Re-run the setup script.

## Scales (GPU Configurations)

| Scale | GPUs | Use Case |
|-------|------|----------|
| `dev` | 1x A10G | Development, smoke tests |
| `small` | 1x H100 | Small models (SmolLM2-135M) |
| `medium` | 2x H100 | Medium models (Qwen3-4B) |
| `large` | 4x H100 | Large models, fast training |

## Docker Image

Pre-built images are stored in Google Artifact Registry (GAR):
```
us-docker.pkg.dev/wrinklefree-481904/wf-train/wf-train:latest
```

Build and push with:
```bash
./scripts/build-image.sh
```

## Cloud Providers

| Provider | Config | Notes |
|----------|--------|-------|
| Nebius | `--cloud nebius` | $1.99/hr H100, recommended |
| RunPod | `--cloud runpod` | Flexible, spot available |
| GCP | `--cloud gcp` | A100/H100, expensive |
| Vast.ai | `--cloud vast` | Cheap H100s, marketplace pricing |

## Troubleshooting

### gsutil 401 "Anonymous caller" errors
```bash
# gsutil doesn't share gcloud credentials - needs separate config
# Run the setup script to create ~/.boto
./scripts/setup-gcs.sh

# Or manually verify .boto exists and points to service account
cat ~/.boto
```

### SkyPilot not finding credentials
```bash
# Check cloud setup
sky check

# Re-authenticate
sky check nebius  # or runpod, gcp, vast
```

### Vast.ai setup
```bash
# 1. Get API key from https://vast.ai/console/cli/
# 2. Configure SkyPilot
sky check vast

# 3. Add to credentials/.env
echo "VASTAI_API_KEY=your_key_here" >> credentials/.env
```

### Docker auth failures on Nebius/RunPod
```bash
# Ensure GCP service account is set
export GOOGLE_APPLICATION_CREDENTIALS=packages/deployer/credentials/gcp-service-account.json

# CLI auto-prepares Docker credentials, but verify:
cat ~/.docker/config.json | jq '.auths["us-docker.pkg.dev"]'
```

### Job stuck in PENDING
```bash
# Check job queue
sky jobs queue

# Check specific job
sky logs <job_id>

# Cancel and retry
sky jobs cancel <job_id>
```

### W&B not logging
```bash
# Ensure WANDB_API_KEY is set
echo $WANDB_API_KEY

# Pass explicitly
wf train -m qwen3_4b -s 2 --secret WANDB_API_KEY=$WANDB_API_KEY
```

## Development

```bash
# Run tests
uv run --package wf-train-deployer pytest packages/deployer/tests/

# Dry run (don't launch)
uv run --package wf-train-deployer wf train -m smollm2_135m -s 2 --dry-run

# Lint
uv run ruff check packages/deployer/
```
