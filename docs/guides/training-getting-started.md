# Training Getting Started

This guide covers cloud-based training using SkyPilot. For local development, see the [quick start](../quick-start.md).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager installed
- SkyPilot configured (`sky check` passes for at least one cloud)
- GCP service account for GCS checkpoint uploads
- WandB account for experiment tracking

## Quick Start

```bash
# 1. Navigate to deployer and load credentials
cd packages/deployer
source credentials/.env

# 2. Run a smoke test to verify setup (~5 min)
wf smoke

# 3. Start training
wf train -m smollm2_135m -t base
```

That's it! Your training job will launch on the cloud and upload checkpoints to GCS.

## Training Configs

| Config | Purpose | Use Case |
|--------|---------|----------|
| `base` | CE + DLM combined | Default training (recommended) |
| `bitdistill_full` | Knowledge distillation | Distill from teacher model |
| `lrc_run` | Low-Rank Correction | Post-quantization error recovery |
| `salient_run` | AWQ-style salient columns | Selective precision |
| `sft_run` | Supervised fine-tuning | Instruction-following |

## Models

| Model | Size | Notes |
|-------|------|-------|
| `smollm2_135m` | 135M | Good for testing |
| `qwen3_4b` | 4B | Production |

## Common Commands

```bash
# Preview without launching
wf train -m qwen3_4b -t base --dry-run

# Train with larger GPU configuration
wf train -m qwen3_4b -t bitdistill_full --scale large  # 4x H100

# Smoke test specific objective
wf smoke -o bitdistill
wf smoke -o lrc

# Check logs
wf logs <job_name>

# List recent jobs
wf runs
```

## GPU Scales

| Scale | GPUs | Use Case |
|-------|------|----------|
| `dev` | 1x A10G | Smoke tests |
| `small` | 1x H100 | Small models |
| `medium` | 2x H100 | Medium models |
| `large` | 4x H100 | Large models, fast training |

## Monitoring

- **SkyPilot logs**: `sky logs <cluster_name>` or `wf logs <job_name>`
- **WandB dashboard**: https://wandb.ai/umd-leans-well/wrinklefree
- **GCS checkpoints**: `gs://wrinklefree-checkpoints/checkpoints/`

## Common Workflows

### Smoke Test Before Full Run

Always run a smoke test to verify the training pipeline works:

```bash
wf smoke -o ce        # Test cross-entropy objective
wf smoke -o bitdistill  # Test distillation
```

Expected: Loss decreases from ~10-12 to ~6-8 over 20 steps.

### Resume from Checkpoint

Training auto-resumes from GCS checkpoints. If a job fails:

```bash
# Same command will find and resume from latest checkpoint
wf train -m qwen3_4b -t base

# Or specify checkpoint explicitly
wf train -m qwen3_4b -t base --resume gs://bucket/path/to/checkpoint.ckpt
```

### Clean Before Retry

If a job fails with stale state:

```bash
sky exec <cluster> "rm -rf /tmp/checkpoints/*"
```

## Troubleshooting

### "sky check" fails

```bash
# Check specific cloud
sky check nebius
sky check runpod

# Re-authenticate if needed
sky check
```

### gsutil 401 errors

```bash
# gsutil needs separate config from gcloud
./scripts/setup-gcs.sh
```

### WandB not logging

```bash
# Verify API key is set
echo $WANDB_API_KEY

# Pass explicitly
wf train -m qwen3_4b -t base --secret WANDB_API_KEY=$WANDB_API_KEY
```

## Next Steps

- **Full deployment reference**: See [packages/deployer/CLAUDE.md](../../packages/deployer/CLAUDE.md)
- **Cloud setup details**: See [Cloud Deployment Guide](cloud-deployment.md)
- **Training configs**: See `packages/training/configs/training/`
