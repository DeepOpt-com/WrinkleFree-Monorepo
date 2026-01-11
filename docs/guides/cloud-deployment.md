# Cloud Deployment Guide

Complete guide for setting up SkyPilot cloud training infrastructure.

## SkyPilot Installation

```bash
# Install SkyPilot with cloud providers
pip install "skypilot-nightly[nebius,runpod,gcp,aws]"

# Verify installation
sky --version
```

## Cloud Provider Setup

### Nebius (Recommended)

Best price/performance for H100s ($1.99/hr).

```bash
# 1. Create account at https://nebius.ai
# 2. Configure SkyPilot
sky check nebius

# 3. Verify
sky launch --cloud nebius --gpus H100:1 --dryrun
```

### RunPod

Flexible pricing, spot instances available.

```bash
# 1. Get API key from https://www.runpod.io/console/user/settings
# 2. Configure
sky check runpod

# 3. Verify
sky launch --cloud runpod --gpus H100:1 --dryrun
```

### GCP (Expensive)

Use only if required. More expensive than alternatives.

```bash
# 1. Install gcloud CLI
# 2. Authenticate
gcloud auth login
gcloud auth application-default login

# 3. Configure SkyPilot
sky check gcp
```

### Vast.ai

Marketplace pricing, variable availability.

```bash
# 1. Get API key from https://vast.ai/console/cli/
# 2. Configure
sky check vast

# 3. Add to credentials
echo "VASTAI_API_KEY=your_key_here" >> packages/deployer/credentials/.env
```

## Credentials Setup

### Directory Structure

```
packages/deployer/credentials/
├── .env                           # Environment variables (gitignored)
└── gcp-service-account.json       # GCP service account (gitignored)
```

### Required Credentials

Create `packages/deployer/credentials/.env`:

```bash
# WandB (required for logging)
WANDB_API_KEY=your_wandb_key

# GCP project for GCS (required for checkpoints)
GOOGLE_CLOUD_PROJECT=wrinklefree-481904

# Optional: cloud-specific
VASTAI_API_KEY=your_vast_key
```

### GCP Service Account

For GCS checkpoint uploads:

1. Create service account in GCP Console
2. Grant "Storage Object Admin" role
3. Download JSON key to `credentials/gcp-service-account.json`
4. Configure environment:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=packages/deployer/credentials/gcp-service-account.json
```

### GCS Setup (gsutil)

SkyPilot uses gsutil for file mounts. gsutil needs separate configuration from gcloud:

```bash
# Auto-setup (recommended)
cd packages/deployer
source credentials/.env  # Auto-configures gsutil

# Manual setup
./scripts/setup-gcs.sh

# Verify
gsutil ls gs://wrinklefree-checkpoints/
```

## Docker Image

Training uses a pre-built Docker image from Google Artifact Registry:

```
us-docker.pkg.dev/wrinklefree-481904/wf-train/wf-train:latest
```

### Docker Auth for Non-GCP Clouds

Nebius and RunPod need explicit Docker credentials:

```bash
# Set environment
export GOOGLE_APPLICATION_CREDENTIALS=packages/deployer/credentials/gcp-service-account.json

# CLI auto-prepares credentials, but verify:
cat ~/.docker/config.json | jq '.auths["us-docker.pkg.dev"]'
```

### Building Custom Image

```bash
cd packages/deployer
./scripts/build-image.sh
```

## SkyPilot Configuration

### GPU Scales

| Scale | GPUs | Memory | Use Case |
|-------|------|--------|----------|
| `dev` | 1x A10G | 24GB | Smoke tests |
| `small` | 1x H100 | 80GB | Small models |
| `medium` | 2x H100 | 160GB | Medium models |
| `large` | 4x H100 | 320GB | Large models |
| `xlarge` | 8x H100 | 640GB | Very large models |

### Cloud Selection

```bash
# Specify cloud provider
wf train -m qwen3_4b -t base --cloud nebius
wf train -m qwen3_4b -t base --cloud runpod

# SkyPilot auto-selects cheapest
wf train -m qwen3_4b -t base  # Uses optimizer
```

### Spot Instances

```bash
# Enable spot (SkyPilot handles preemption recovery)
wf train -m qwen3_4b -t base --use-spot
```

## Verification Checklist

Before launching training:

```bash
# 1. Check cloud connectivity
sky check

# 2. Verify GCS access
gsutil ls gs://wrinklefree-checkpoints/

# 3. Check credentials loaded
cd packages/deployer
source credentials/.env
echo $WANDB_API_KEY  # Should show key

# 4. Run smoke test
wf smoke --dry-run  # Preview
wf smoke            # Run (~5 min)
```

## Troubleshooting

### SkyPilot can't find credentials

```bash
# Re-run cloud setup
sky check nebius
sky check runpod

# Check status
sky status
```

### gsutil "Anonymous caller" errors

```bash
# gsutil doesn't share gcloud credentials
./scripts/setup-gcs.sh

# Verify .boto exists
cat ~/.boto
```

### Docker auth failures on Nebius/RunPod

```bash
# Ensure GCP service account is set
export GOOGLE_APPLICATION_CREDENTIALS=packages/deployer/credentials/gcp-service-account.json

# Re-authenticate
gcloud auth configure-docker us-docker.pkg.dev
```

### Job stuck in PENDING

```bash
# Check queue
sky jobs queue

# Check specific job
sky logs <job_id>

# Cancel and retry
sky jobs cancel <job_id>
```

### Insufficient quota

```bash
# Check available resources
sky show-gpus --cloud nebius

# Try different cloud
wf train -m qwen3_4b -t base --cloud runpod
```

## Next Steps

- **Quick training commands**: See [Training Getting Started](training-getting-started.md)
- **Full deployer reference**: See [packages/deployer/CLAUDE.md](../../packages/deployer/CLAUDE.md)
