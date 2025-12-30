# WrinkleFree-Deployer

Training job launcher for 1.58-bit quantized LLMs. Uses SkyPilot for managed GPU jobs with spot recovery.

**For detailed AI discovery docs, see `docs/AIDEV.md`.**

## Monorepo Integration

This package is the **orchestrator** for the WrinkleFree monorepo - it launches jobs for all other packages.

**Orchestrates**:
| Command | Package | Description |
|---------|---------|-------------|
| `wf train` | `training` | 1.58-bit quantization training |
| `wf distill` | `distillation` | Knowledge distillation |
| `wf dlm` | `converter` | DLM format conversion |
| `wf serve` | `inference` | Model serving |
| `wf eval` | `eval` | Model evaluation |

**Running commands**:
```bash
# IMPORTANT: wf commands must be run from packages/deployer directory
# (train.yaml uses relative paths that require this)
cd packages/deployer
source credentials/.env
uv run --package wrinklefree-deployer wf train -m qwen3_4b -s 2
```

## Quick Reference

**Important:** Run all `wf` commands from `packages/deployer` directory.

```bash
# Set up credentials and run from deployer directory
cd packages/deployer
source credentials/.env

# Launch training
uv run --package wrinklefree-deployer wf train -m qwen3_4b -s 2 --cloud nebius

# With specific scale (4x H100)
uv run --package wrinklefree-deployer wf train -m qwen3_4b -s 2 --scale large

# Check logs
uv run --package wrinklefree-deployer wf logs <run_id>

# List recent runs
uv run --package wrinklefree-deployer wf runs

# Direct SkyPilot commands
uv run --package wrinklefree-deployer sky check
uv run --package wrinklefree-deployer sky jobs queue
```

## Key Files

| File | Purpose |
|------|---------|
| `src/wf_deployer/constants.py` | All magic strings, defaults, scales, GAR config |
| `src/wf_deployer/core.py` | Main API: train(), train_distill(), logs(), cancel() |
| `src/wf_deployer/cli.py` | CLI commands |
| `skypilot/train.yaml` | SkyPilot training job template |
| `skypilot/distill_train.yaml` | SkyPilot distillation job template |
| `skypilot/service.yaml` | SkyServe inference template |
| `credentials/.env` | Local credentials (gitignored) |
| `credentials/gcp-service-account.json` | GCP service account for GCS + Docker auth |

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
uv run --package wrinklefree-deployer pytest packages/deployer/tests/

# Dry run (don't launch)
uv run --package wrinklefree-deployer wf train -m smollm2_135m -s 2 --dry-run

# Lint
uv run ruff check packages/deployer/
```
