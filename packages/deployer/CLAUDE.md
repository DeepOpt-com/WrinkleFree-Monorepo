# WrinkleFree-Deployer

Training job launcher for 1.58-bit quantized LLMs. Uses SkyPilot for managed GPU jobs with spot recovery.

## CRITICAL Rules

1. **RUN FROM DEPLOYER DIR**: All `wf` and `sky` commands must run from `packages/deployer`
2. **SOURCE CREDENTIALS FIRST**: Always `source credentials/.env` before any cloud command
3. **NEVER CANCEL OTHERS' JOBS**: Only cancel SkyPilot jobs you started in this session
4. **USE NEBIUS**: Prefer Nebius over RunPod/GCP (better availability, lower cost)
5. **CLEAN BEFORE RETRY**: Run `sky exec <cluster> "rm -rf /tmp/checkpoints/*"` before retrying failed jobs
6. **FIX, DON'T FALL BACK**: When something breaks, FIX IT. Don't silently fall back to alternatives (e.g., don't switch from MuonClip to AdamW - fix MuonClip)

## Quick Smoke Test (Lightning)

```bash
cd packages/deployer
source credentials/.env

# Launch Lightning smoke test with auto batch size
sky launch skypilot/smoke_test_lightning.yaml -y --cluster lightning-smoke --env OBJECTIVE_COMBO=dlm

# Monitor
sky logs lightning-smoke

# Re-run on existing cluster (faster)
sky exec lightning-smoke skypilot/smoke_test_lightning.yaml --env OBJECTIVE_COMBO=dlm

# Teardown
sky down lightning-smoke -y
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

## Quick Reference

**Important:** Run all `wf` commands from `packages/deployer` directory.

```bash
# Set up credentials and run from deployer directory
cd packages/deployer
source credentials/.env

# Launch training
uv run --package wf-train-deployer wf train -m qwen3_4b -s 2 --cloud nebius

# With specific scale (4x H100)
uv run --package wf-train-deployer wf train -m qwen3_4b -s 2 --scale large

# BitDistill distillation (via training objectives)
uv run --package wf-train-deployer wf train -m qwen3_4b \
  --training bitdistill_full --cloud nebius

# LRC calibration (post-quantization recovery)
uv run --package wf-train-deployer wf train -m qwen3_4b \
  --training lrc_calibration --cloud nebius

# Check logs
uv run --package wf-train-deployer wf logs <run_id>

# List recent runs
uv run --package wf-train-deployer wf runs

# Direct SkyPilot commands
uv run --package wf-train-deployer sky check
uv run --package wf-train-deployer sky jobs queue
```

## Key Files

| File | Purpose |
|------|---------|
| `src/wf_deploy/constants.py` | All magic strings, defaults, scales, GAR config |
| `src/wf_deploy/core.py` | Main API: train(), logs() |
| `src/wf_deploy/cli.py` | CLI commands |
| `skypilot/train.yaml` | SkyPilot training job template |
| `skypilot/service.yaml` | SkyServe inference template |
| `skypilot/smoke_test_lightning.yaml` | **Smoke test: Lightning + auto batch (RECOMMENDED)** |
| `skypilot/smoke_test_meta_opt_1gpu.yaml` | Smoke test: Meta-optimization (1x L40) |
| `skypilot/smoke_test_meta_opt_2gpu.yaml` | Smoke test: Meta-optimization (2x L40 FSDP) |
| `skypilot/smoke_test_unified_1gpu.yaml` | Smoke test: 1x L40 unified training (legacy) |
| `skypilot/smoke_test_unified_2gpu.yaml` | Smoke test: 2x L40 with FSDP |
| `skypilot/smoke_test_bitdistill.yaml` | Smoke test: BitDistill distillation |
| `skypilot/smoke_test_lrc.yaml` | Smoke test: LRC calibration (1x L40) |
| `credentials/.env` | Local credentials (gitignored) |
| `credentials/gcp-service-account.json` | GCP service account for GCS + Docker auth |

## Smoke Tests

Quick validation of the training pipeline (~5 minutes):

```bash
cd packages/deployer
source credentials/.env

# Lightning + auto batch (RECOMMENDED)
sky launch skypilot/smoke_test_lightning.yaml -y --cluster lightning-smoke \
  --env OBJECTIVE_COMBO=dlm

# Meta-optimization (LDC-MTL + ODM, O(1) complexity)
# References: arxiv:2502.08585 (LDC-MTL), arxiv:2312.02406 (ODM/EXP3)
sky launch skypilot/smoke_test_meta_opt_1gpu.yaml -y --cluster meta-1gpu  # 1x L40
sky launch skypilot/smoke_test_meta_opt_2gpu.yaml -y --cluster meta-2gpu  # 2x L40 FSDP

# 1x L40 unified training (legacy)
sky launch skypilot/smoke_test_unified_1gpu.yaml -y --cluster unified-1gpu

# 2x L40 smoke test (FSDP data parallelism)
sky launch skypilot/smoke_test_unified_2gpu.yaml -y --cluster unified-2gpu

# LRC calibration smoke test (Low-Rank Correction)
export SKYPILOT_DOCKER_PASSWORD=$(cat credentials/gcp-service-account.json)
sky launch skypilot/smoke_test_lrc.yaml -y --secret WANDB_API_KEY --secret SKYPILOT_DOCKER_PASSWORD

# Monitor
sky logs lightning-smoke
sky logs unified-1gpu

# Teardown
sky down lightning-smoke meta-1gpu meta-2gpu unified-1gpu unified-2gpu wf-smoke-lrc -y
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
