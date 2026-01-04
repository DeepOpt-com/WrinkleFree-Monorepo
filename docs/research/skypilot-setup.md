# SkyPilot Setup Guide: Nebius + RunPod

This guide covers setting up SkyPilot with Nebius (primary, cheapest) and RunPod (fallback) for WrinkleFree training.

---

## Quick Start

```bash
# 1. Install SkyPilot with Nebius + RunPod support
pip install "skypilot-nightly[nebius,runpod]"

# 2. Set up credentials (see sections below)

# 3. Verify
sky check nebius
sky check runpod

# 4. Launch training (auto-teardown when done)
cd WrinkleFree-Deployer
sky jobs launch skypilot/train.yaml -e MODEL=qwen3_4b -e STAGE=2
```

---

## Nebius Configuration

Nebius offers H100s at $1.99/hr (Explorer Tier) - the cheapest option.

### Option 1: Automated Setup (Recommended)

```bash
pip install "skypilot-nightly[nebius]"

# Download and run setup script
wget https://raw.githubusercontent.com/nebius/nebius-solution-library/main/skypilot/nebius-setup.sh
chmod +x nebius-setup.sh
./nebius-setup.sh

# Follow prompts to select tenant and project
sky check nebius
```

### Option 2: Manual Setup

```bash
# Install Nebius CLI
pip install nebius-cli

# Configure credentials
mkdir -p ~/.nebius
nebius iam get-access-token > ~/.nebius/NEBIUS_IAM_TOKEN.txt
nebius --format json iam whoami | jq -r '.user_profile.tenants[0].tenant_id' > ~/.nebius/NEBIUS_TENANT_ID.txt

# Verify
sky check nebius
```

### Service Account (for CI/CD)

```bash
# Get service account ID
export SA_ID=$(nebius iam service-account get-by-name \
  --name wrinklefree-training \
  --format json | jq -r ".metadata.id")

# Generate credentials
nebius iam auth-public-key generate \
  --service-account-id $SA_ID \
  --output ~/.nebius/credentials.json

sky check nebius
```

---

## RunPod Configuration

RunPod offers H100s at $2.39/hr - good fallback when Nebius is busy.

```bash
pip install "skypilot[runpod]"

# Get API key from https://www.runpod.io/console/user/settings
export RUNPOD_API_KEY="your-api-key"

# Add to ~/.bashrc for persistence
echo 'export RUNPOD_API_KEY="your-api-key"' >> ~/.bashrc

sky check runpod
```

---

## GCS Checkpoint Storage

WrinkleFree requires GCS for checkpoint storage. Two options:

### Option 1: File Mount (Simple)

Place your GCP service account JSON at a known location:

```bash
# Standard location
mkdir -p ~/.config/gcloud
cp /path/to/service-account.json ~/.config/gcloud/wrinklefree-sa.json
```

In SkyPilot YAML:
```yaml
file_mounts:
  /tmp/gcp-creds.json: ~/.config/gcloud/wrinklefree-sa.json

envs:
  GOOGLE_APPLICATION_CREDENTIALS: /tmp/gcp-creds.json
  GCS_BUCKET: wrinklefree-checkpoints
```

### Option 2: Environment Variable (CI/CD Friendly)

```bash
# Base64 encode the JSON
export GCP_CREDENTIALS_B64=$(base64 -w0 /path/to/service-account.json)

# Add to ~/.bashrc
echo "export GCP_CREDENTIALS_B64='$GCP_CREDENTIALS_B64'" >> ~/.bashrc
```

In SkyPilot YAML setup:
```yaml
envs:
  GCP_CREDENTIALS_B64: ${GCP_CREDENTIALS_B64}
  GCS_BUCKET: wrinklefree-checkpoints

setup: |
  # Decode GCP credentials
  echo "$GCP_CREDENTIALS_B64" | base64 -d > /tmp/gcp-creds.json
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-creds.json
```

---

## WandB Configuration

```bash
# Get API key from https://wandb.ai/authorize
export WANDB_API_KEY="your-wandb-key"

# Add to ~/.bashrc
echo 'export WANDB_API_KEY="your-wandb-key"' >> ~/.bashrc
```

---

## Auto-Shutdown: Managed Jobs vs Clusters

### Managed Jobs (Recommended) - Auto Cleanup

```bash
# Launches, runs, and tears down automatically
sky jobs launch skypilot/train.yaml -e MODEL=qwen3_4b -e STAGE=2

# Monitor
sky jobs queue
sky jobs logs <job_id>

# Cancel if needed
sky jobs cancel <job_id>
```

**Features:**
- Automatic cluster teardown when done
- Auto-recovery from spot preemptions
- No idle cost - cluster exists only during training
- "Serverless" style

### Cluster Mode - Manual with Autostop

```bash
# Launch with auto-teardown after 10 min idle
sky launch --down --idle-minutes-to-autostop 10 skypilot/train.yaml

# Or in YAML:
resources:
  autostop: 10m  # Tear down after 10 min idle
```

---

## Multi-Cloud Fallback

SkyPilot automatically falls back to available clouds:

```yaml
resources:
  accelerators: H100:4
  use_spot: true

# Try Nebius first, then RunPod
candidates:
  - cloud: nebius
  - cloud: runpod
```

Or specify cloud explicitly:

```bash
sky jobs launch skypilot/train.yaml --cloud nebius
sky jobs launch skypilot/train.yaml --cloud runpod  # Fallback
```

---

## InfiniBand Networking

Enable high-performance networking for distributed training:

```yaml
resources:
  accelerators: H100:4
  network_tier: best  # Provisions InfiniBand where available
```

Nebius and Lambda Labs support 400Gb/s InfiniBand.

---

## Storage Optimization

Use `MOUNT_CACHED` for checkpoint writes (9.6x faster):

```yaml
file_mounts:
  /checkpoint:
    name: wrinklefree-checkpoints
    store: gcs  # or s3
    mode: MOUNT_CACHED  # Fast local writes, async sync to cloud
```

---

## Complete Example: train.yaml

```yaml
name: wrinklefree-train-${MODEL}-stage${STAGE}

resources:
  accelerators: ${GPU:-H100:4}
  use_spot: true
  disk_tier: best
  network_tier: best

candidates:
  - cloud: nebius
  - cloud: runpod

workdir: ../WrinkleFree-1.58Quant

file_mounts:
  /tmp/gcp-creds.json: ~/.config/gcloud/wrinklefree-sa.json
  /checkpoint:
    name: wrinklefree-checkpoints
    store: gcs
    mode: MOUNT_CACHED

envs:
  MODEL: qwen3_4b
  STAGE: "2"
  GOOGLE_APPLICATION_CREDENTIALS: /tmp/gcp-creds.json
  WANDB_API_KEY: ${WANDB_API_KEY}
  WANDB_PROJECT: wrinklefree
  WANDB_RUN_ID: ${SKYPILOT_TASK_ID}

setup: |
  set -e
  cd ~/sky_workdir
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env
  uv sync

run: |
  set -e
  cd ~/sky_workdir
  source $HOME/.local/bin/env

  case $STAGE in
    1)   CONFIG="stage1_subln" ;;
    1.9) CONFIG="stage1_9_layerwise" ;;
    2)   CONFIG="stage2_pretrain" ;;
    3)   CONFIG="stage3_distill" ;;
  esac

  uv run python scripts/train_lightning.py \
    model=${MODEL} \
    training=${CONFIG} \
    output_dir=/checkpoint/${MODEL}/stage${STAGE} \
    gcs.enabled=true \
    gcs.bucket=wrinklefree-checkpoints
```

---

## Troubleshooting

### "Nebius credentials not found"
```bash
./nebius-setup.sh  # Re-run setup
sky check nebius
```

### "RunPod API key invalid"
```bash
# Verify key at https://www.runpod.io/console/user/settings
echo $RUNPOD_API_KEY
sky check runpod
```

### "GCS permission denied"
```bash
# Verify service account has Storage Admin role
gcloud projects get-iam-policy wrinklefree-481904 \
  --format='table(bindings.role,bindings.members)'
```

### "No GPUs available"
```bash
# Check availability across clouds
sky show-gpus --all

# Try different cloud
sky jobs launch train.yaml --cloud runpod
```

---

## Sources

- [SkyPilot + Nebius Setup](https://docs.skypilot.co/en/latest/cloud-setup/cloud-permissions/nebius.html)
- [Nebius SkyPilot Integration](https://docs.nebius.com/3p-integrations/skypilot)
- [SkyPilot Managed Jobs](https://docs.skypilot.co/en/latest/examples/managed-jobs.html)
- [SkyPilot Autostop/Autodown](https://docs.skypilot.co/en/latest/reference/auto-stop.html)
