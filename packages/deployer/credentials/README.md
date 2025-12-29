# Credentials Setup

This folder contains credentials for WrinkleFree cloud training. **Never commit actual credentials.**

## Quick Start

```bash
# 1. Copy example files
cp .env.example .env
cp gcp-service-account.example.json gcp-service-account.json

# 2. Fill in your credentials
# Edit .env and gcp-service-account.json with your values

# 3. Verify SkyPilot can see your clouds
source .env
sky check
```

## Required Credentials

### 1. Weights & Biases
**Location:** `credentials/.env` → `WANDB_API_KEY`

```bash
# Get key from: https://wandb.ai/authorize
export WANDB_API_KEY=your_key_here
```

### 2. GCP Service Account (for GCS checkpoints + Docker auth)
**Location:** `credentials/gcp-service-account.json`

This file is used for:
- GCS checkpoint persistence (all clouds)
- Docker image authentication on non-GCP clouds (Nebius, RunPod)

**Setup:**
1. Go to [GCP Console → IAM → Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Create or select a service account
3. Grant roles: `Storage Object Admin`, `Artifact Registry Reader`
4. Create key → JSON → Download
5. Save as `credentials/gcp-service-account.json`

**For Docker auth on Nebius/RunPod:**
The CLI automatically uses this file to authenticate with Google Artifact Registry.
No additional setup needed - just ensure the file exists.

### 3. RunPod API Key
**Location:** `credentials/.env` → `RUNPOD_API_KEY`

```bash
# Get key from: https://www.runpod.io/console/user/settings
export RUNPOD_API_KEY=your_key_here
```

## Cloud-Specific Setup

### Nebius (Cheapest H100s - $1.99/hr)
**Location:** `~/.nebius/credentials.json`

```bash
# Run the setup script (requires browser for OAuth)
./scripts/setup-nebius.sh

# Or manually:
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
nebius profile create
# Follow prompts to create service account and access key
```

### RunPod
```bash
pip install runpod
runpod config  # Enter API key when prompted
```

## Optional Credentials

### HuggingFace Token (for private models)
**Location:** `credentials/.env` → `HF_TOKEN`

```bash
# Get token from: https://huggingface.co/settings/tokens
export HF_TOKEN=your_token_here
```

### Lambda Labs
**Location:** `~/.lambda_cloud/lambda_keys`

```bash
# Get key from: https://cloud.lambdalabs.com/api-keys
mkdir -p ~/.lambda_cloud
echo "api_key = YOUR_KEY" > ~/.lambda_cloud/lambda_keys
```

## Credential Locations Summary

| Credential | Location | Used By |
|------------|----------|---------|
| W&B API | `credentials/.env` | Training (via --secret) |
| GCP JSON | `credentials/gcp-service-account.json` | GCS uploads, Docker auth |
| RunPod API | `credentials/.env` or `runpod config` | SkyPilot |
| Nebius | `~/.nebius/credentials.json` | SkyPilot |
| HuggingFace | `credentials/.env` | Model downloads |

## Launching with Secrets (Canonical Approach)

SkyPilot's `secrets` feature redacts credentials from logs and the dashboard:

```bash
# 1. Set secrets in your environment
source credentials/.env

# 2. Launch with wf CLI (auto-prepares Docker auth)
wf train -m qwen3_4b -s 2 --cloud nebius

# Or manually with sky + --secret flags:
export SKYPILOT_DOCKER_PASSWORD=$(cat credentials/gcp-service-account.json)
sky launch skypilot/train.yaml --secret WANDB_API_KEY --secret SKYPILOT_DOCKER_PASSWORD
```

## Docker Image on Non-GCP Clouds

When running on Nebius or RunPod, the CLI automatically handles Docker registry auth:

1. Reads `credentials/gcp-service-account.json`
2. Sets `SKYPILOT_DOCKER_PASSWORD` (raw JSON for Nebius, base64 for RunPod)
3. Passes via SkyPilot's secrets feature (redacted from logs)

The Docker image is stored in Google Artifact Registry:
```
us-docker.pkg.dev/wrinklefree-481904/wf-train/wf-train:latest
```

## Verify Setup

```bash
# Check which clouds are enabled
source credentials/.env
sky check

# Expected output:
# ✓ GCP [compute, storage]
# ✓ RunPod [compute]
# ✓ Nebius [compute, storage]  (if configured)
```

## Security Notes

- All files in `credentials/` except `*.example*` are gitignored
- Never commit API keys or service account files
- Secrets passed via `--secret` are redacted from SkyPilot logs/dashboard
- Use environment variables for CI/CD, not file-based credentials
