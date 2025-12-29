# Training Guide

This guide covers training 1.58-bit quantized models using the WrinkleFree training pipeline.

## Overview

WrinkleFree uses a multi-stage training pipeline to convert and train standard LLMs into efficient 1.58-bit quantized models:

1. **Stage 1**: SubLN Insertion - Model architecture conversion (no training)
2. **Stage 1.9**: Layer-wise Distillation - Knowledge distillation from teacher model
3. **Stage 2**: Continue Pretrain - Additional pretraining on large-scale data
4. **Stage 3**: Distillation Fine-tuning - Task-specific fine-tuning

## Backends

| Backend | Default | Best For |
|---------|---------|----------|
| **Modal** | Yes | Easy AI tool control, automatic scaling, pay-per-use |
| **SkyPilot** | No | Spot instances, multi-cloud, custom infrastructure |

**Modal advantages:**
- Automatic shutdown when training completes (pay only for compute time)
- Fingerprint-based auto-resume (same config = resume from checkpoint)
- Persistent volumes for checkpoints and HF cache
- Simple Python/CLI interface for AI tools

**SkyPilot advantages:**
- Spot instance support (up to 70% cheaper)
- Multi-cloud orchestration (RunPod, AWS, GCP)
- Custom cloud storage (S3, GCS, R2)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Full Precision Model                     1.58-bit Quantized Model
    (e.g., Qwen3-4B)                        (WrinkleFree-Qwen3-4B)
           │                                            ▲
           │                                            │
           ▼                                            │
    ┌──────────────┐                            ┌──────────────┐
    │   Stage 1    │                            │   Stage 3    │
    │              │                            │              │
    │ SubLN Insert │                            │  Distill FT  │
    │  (No Train)  │                            │  (Task Data) │
    └──────┬───────┘                            └──────▲───────┘
           │                                           │
           │ Modified Architecture                     │
           ▼                                           │
    ┌──────────────┐                            ┌─────────────┐
    │  Stage 1.9   │                            │   Stage 2   │
    │              │────────────────────────────►              │
    │  Layer-wise  │    Quantized Weights       │   Continue  │
    │  Distill     │                            │  Pretrain   │
    └──────────────┘                            └─────────────┘
         500-2000 steps                          1000-10k steps
         ~$10-50 on A100                         ~$50-200 on H100
```

### Stage Details

| Stage | Purpose | Training | Duration | Cost (Est.) |
|-------|---------|----------|----------|-------------|
| **1** | Insert SubLN layers | ✗ | Seconds | $0 |
| **1.9** | Distill knowledge layer-by-layer | ✓ | 500-2000 steps | $10-50 |
| **2** | Continue pretraining (10B+ tokens) | ✓ | 1k-10k steps | $50-200 |
| **3** | Fine-tune for specific tasks | ✓ | 500-2k steps | $10-50 |

## Quick Start with Modal (Recommended)

### Prerequisites

```bash
# Install and authenticate
uv add modal
uv run modal setup

# Create secrets
modal secret create wandb-api-key WANDB_API_KEY=<your-key>
modal secret create huggingface-token HF_TOKEN=<your-token>

# Install WrinkleFree-Deployer
cd WrinkleFree-Deployer
uv sync
```

### Launch Training

**CLI:**
```bash
# Stage 1.9: Layer-wise distillation (recommended starting point)
uv run wf train -m qwen3_4b -s 1.9

# With limited steps (for testing)
uv run wf train -m smollm2_135m -s 1.9 training.max_steps=100

# Multi-GPU with scale profiles
uv run wf train -m qwen3_4b -s 1.9 --scale large  # 4x H100

# Smoke test
uv run wf smoke
```

**Python API:**
```python
from wf_deployer import train

# Basic usage
run_id = train("qwen3_4b", stage=1.9)

# With scale and overrides
run_id = train("qwen3_4b", stage=1.9, scale="large", overrides=["training.lr=1e-4"])

# Wait for completion
run_id = train("qwen3_4b", stage=1.9, detach=False)
```

**Monitor:**
```bash
uv run wf runs                # List runs
uv run wf logs <run_id>       # View logs
uv run wf cancel <run_id>     # Cancel run
```

### Key Features

**Automatic Shutdown:**
Modal functions shut down automatically when training completes. You only pay for actual compute time.

**Fingerprint-Based Resume:**
Runs are identified by a SHA256 hash of (config + git commit). Re-running the same config automatically resumes from the last checkpoint:
```bash
# First run - trains from scratch
uv run wf train -m qwen3_4b -s 1.9

# Same command later - resumes from checkpoint
uv run wf train -m qwen3_4b -s 1.9

# Force fresh start
uv run wf train -m qwen3_4b -s 1.9 --skip-recovery
```

**Persistent Volumes:**
- `wrinklefree-checkpoints`: Stores all training checkpoints
- `wrinklefree-hf-cache`: Caches HuggingFace models/datasets

---

## Quick Start with SkyPilot (Alternative)

### Prerequisites

1. **Install SkyPilot**:
   ```bash
   uv add "skypilot[runpod,aws,gcp]"
   sky check
   ```

2. **Configure Cloud Credentials**:
   ```bash
   # RunPod (primary GPU provider)
   export RUNPOD_API_KEY=your_key_here

   # AWS (for checkpoint storage)
   aws configure

   # Or GCP
   gcloud auth login
   ```

3. **Create Checkpoint Bucket**:
   ```bash
   # AWS S3
   aws s3 mb s3://wrinklefree-checkpoints

   # Or GCP GCS
   gsutil mb gs://wrinklefree-checkpoints
   ```

### Launch Training

```bash
# Navigate to the deployer directory
cd WrinkleFree-Deployer

# Stage 2: Continue Pretrain (most common)
sky jobs launch skypilot/train.yaml \
  -e MODEL=qwen3_4b \
  -e STAGE=2 \
  -e CHECKPOINT_BUCKET=wrinklefree-checkpoints

# Monitor training
sky jobs queue
sky jobs logs <job_id>
```

## Training Configurations

### Small Model (SmolLM2-135M)

Fast iteration for testing and development:

```bash
sky jobs launch skypilot/train.yaml \
  -e MODEL=smollm2_135m \
  -e STAGE=1.9 \
  -e ACCELERATOR=A100:1
```

**Specs:**
- GPU: 1x A100-80GB
- Duration: ~4 hours
- Cost: ~$10
- Use case: Testing pipeline, debugging

### Medium Model (Qwen3-4B)

Balanced performance and cost:

```bash
sky jobs launch skypilot/train.yaml \
  -e MODEL=qwen3_4b \
  -e STAGE=2 \
  -e ACCELERATOR=H100:4
```

**Specs:**
- GPU: 4x H100
- Duration: ~24 hours (10B tokens)
- Cost: ~$200
- Use case: Production models

### Custom Configuration

Override any parameter:

```bash
sky jobs launch skypilot/train.yaml \
  -e MODEL=custom_model \
  -e STAGE=2 \
  -e ACCELERATOR=A100:8 \
  -e CHECKPOINT_BUCKET=my-bucket \
  -e CHECKPOINT_STORE=gcs \
  -e WANDB_PROJECT=my-project
```

## Smoke Testing

Quick validation before running expensive training:

```bash
# Launch smoke test (5 minutes, ~$1)
sky launch skypilot/smoke_test.yaml -y --cluster smoke-test

# Monitor
sky logs smoke-test --follow

# Cleanup
sky down smoke-test -y
```

**What it tests:**
- Stage 1 model conversion
- Stage 1.9 training loop (100 steps)
- Checkpoint saving
- GCS upload (if configured)
- W&B logging

**Expected output:**
```
[Stage 1] Converting model with SubLN insertion...
[Stage 1] Complete! Saved to /tmp/checkpoints/stage1_model

[Stage 1.9] Running layer-wise distillation (limited to 100 steps)...
Step 10: loss=2.45, lr=5e-5
Step 20: loss=2.32, lr=4.8e-5
...
Step 100: loss=1.98, lr=2e-5

Smoke Test Complete!
Checkpoints: /tmp/checkpoints/smoke_test/checkpoint-100/
[Upload] Uploading to gs://wrinklefree-checkpoints/checkpoints/smoke-test/
```

## Monitoring

### Job Status

```bash
# List all jobs
sky jobs queue

# Output:
# ID  NAME                      STATUS   STARTED      DURATION
# 1   wrinklefree-train-qwen3   RUNNING  2 hours ago  2:14:33
# 2   wrinklefree-train-smol    DONE     1 day ago    4:23:11
```

### Real-time Logs

```bash
# Stream logs
sky jobs logs <job_id> --follow

# Last 100 lines
sky jobs logs <job_id> --tail 100
```

### Weights & Biases

All training runs are logged to W&B:

```bash
# View in browser
# https://wandb.ai/your-team/wrinklefree/runs/<run_id>
```

**Key metrics tracked:**
- Training loss
- Learning rate
- GPU utilization
- Tokens per second
- Memory usage
- Checkpoint sizes

### GPU Utilization

Training jobs automatically log GPU stats every 60-300 seconds:

```
=== GPU Utilization 2025-12-21 10:00:00 ===
utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W]
98 %, 92 %, 73728 MiB, 81920 MiB, 450.00 W
```

**Troubleshooting:**
- **Low GPU util (<70%)**: Check data loading, increase batch size
- **Low memory util (<60%)**: Increase model size or batch size
- **Power draw low**: GPU throttling, check temperature

## Checkpoint Management

### Automatic Saving

Checkpoints are automatically saved to your configured bucket:

```
s3://wrinklefree-checkpoints/
├── qwen3_4b/
│   ├── stage1.9/
│   │   ├── checkpoint-500/
│   │   ├── checkpoint-1000/
│   │   └── checkpoint-1500/
│   ├── stage2/
│   │   ├── checkpoint-1000/
│   │   ├── checkpoint-2000/
│   │   └── final/
│   └── stage3/
└── smollm2_135m/
    └── stage2/
```

### Download Checkpoints

```bash
# List available checkpoints
aws s3 ls s3://wrinklefree-checkpoints/qwen3_4b/stage2/

# Download specific checkpoint
aws s3 sync s3://wrinklefree-checkpoints/qwen3_4b/stage2/checkpoint-1000/ \
  ./checkpoints/qwen3_4b/

# Or for GCS
gsutil -m rsync -r gs://wrinklefree-checkpoints/qwen3_4b/stage2/checkpoint-1000/ \
  ./checkpoints/qwen3_4b/
```

### Resume from Checkpoint

Training automatically resumes from the latest checkpoint if a job is preempted:

```yaml
# In train.yaml
run: |
  # Check for existing checkpoint to resume from
  CHECKPOINT_DIR="/checkpoint/${MODEL}/stage${STAGE}"
  if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
    echo "Resuming from checkpoint..."
    RESUME_FLAG="training.resume_from_checkpoint=$CHECKPOINT_DIR"
  fi
```

Manual resume:

```bash
# Resume specific job
sky jobs launch skypilot/train.yaml \
  -e MODEL=qwen3_4b \
  -e STAGE=2 \
  -e RESUME_FROM=s3://wrinklefree-checkpoints/qwen3_4b/stage2/checkpoint-1000
```

## Advanced Features

### Checkpoint Storage Modes

SkyPilot supports multiple storage backends:

```yaml
file_mounts:
  /checkpoint:
    name: wrinklefree-checkpoints
    store: s3  # or gcs, r2, azure
    mode: MOUNT_CACHED  # High-performance writes (9.6x faster than MOUNT)
```

**Storage options:**
- `s3`: AWS S3 (default)
- `gcs`: Google Cloud Storage
- `r2`: Cloudflare R2 (S3-compatible, cheaper egress)
- `azure`: Azure Blob Storage

**Mount modes:**
- `MOUNT`: Direct mount (slower writes)
- `MOUNT_CACHED`: Async uploads (9.6x faster, recommended)
- `COPY`: Copy on start/end (simplest, slower)

### Job Recovery

Managed jobs automatically restart on failures:

```yaml
resources:
  job_recovery:
    max_restarts_on_errors: 3  # Retry up to 3 times
```

**Recovery scenarios:**
- Spot preemption
- NCCL timeout errors
- Driver crashes
- Network interruptions

### Multi-GPU Training

Use FSDP (Fully Sharded Data Parallel) for large models:

```bash
# 8x H100 for faster training
sky jobs launch skypilot/train.yaml \
  -e MODEL=qwen3_70b \
  -e STAGE=2 \
  -e ACCELERATOR=H100:8
```

The training script automatically configures FSDP:

```yaml
run: |
  uv run python scripts/train.py \
    distributed=fsdp_multi \  # Automatically uses all GPUs
    ...
```

### Wandb Integration

Configure Weights & Biases tracking:

```bash
# Set API key
export WANDB_API_KEY=your_key_here

# Launch with custom project
sky jobs launch skypilot/train.yaml \
  -e MODEL=qwen3_4b \
  -e STAGE=2 \
  -e WANDB_PROJECT=my-project \
  -e WANDB_API_KEY=$WANDB_API_KEY
```

**Environment variables:**
- `WANDB_API_KEY`: Authentication (required)
- `WANDB_PROJECT`: Project name (default: `wrinklefree`)
- `WANDB_RUN_ID`: Run identifier (default: `$SKYPILOT_TASK_ID`)

## Cloud Provider Notes

### RunPod (Recommended)

**Advantages:**
- Competitive GPU pricing
- Good availability for A100/H100
- Fast provisioning

**Limitations:**
- **40GB disk limit** for containers
- Use external storage (GCS/S3) for checkpoints

```yaml
resources:
  cloud: runpod
  disk_size: 40  # Maximum
  accelerators: A100-80GB:4
```

### GCP

**Setup for GCS storage:**

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
source ~/.bashrc

# Authenticate
gcloud auth activate-service-account \
  --key-file=credentials/gcp-service-account.json

# Enable Cloud Resource Manager API (required for GCS)
# https://console.developers.google.com/apis/api/cloudresourcemanager.googleapis.com

# Verify
sky check  # Should show "GCP: enabled [storage]"
```

### AWS

**Setup for S3 storage:**

```bash
# Configure credentials
aws configure

# Verify
aws s3 ls s3://wrinklefree-checkpoints/
```

## Cost Optimization

### Spot Instances

Enable spot for up to 70% cost savings:

```yaml
resources:
  use_spot: true
  spot_recovery: FAILOVER  # Auto-migrate on preemption
```

**Trade-offs:**
- **Pro**: 50-70% cheaper
- **Con**: Can be preempted (jobs auto-resume from checkpoint)

### Right-sizing GPUs

| Model Size | Recommended GPU | Cost/Hour | Notes |
|------------|----------------|-----------|-------|
| <500M params | RTX4090 (24GB) | ~$0.40 | Testing only |
| <2B params | A100-40GB | ~$1.10 | Development |
| <7B params | A100-80GB | ~$1.80 | Production |
| <13B params | H100 (80GB) | ~$4.00 | Production |
| 13B-70B | 4-8x H100 | ~$16-32 | Production |

### Checkpoint Frequency

Balance cost vs. fault tolerance:

```python
# In scripts/train.py config
training:
  checkpoint:
    save_interval: 500  # Steps between checkpoints
```

**Trade-offs:**
- **High frequency** (every 100 steps): More fault tolerant, higher storage cost
- **Low frequency** (every 1000 steps): Lower storage cost, more lost work on failure

Recommended: **500 steps** (sweet spot)

## Troubleshooting

### Job Won't Start

```bash
# Check SkyPilot status
sky check

# Verify cloud credentials
sky status --refresh

# Check quota
sky show-gpus --region us-east-1
```

### Out of Memory (OOM)

```bash
# Reduce batch size or model size
# Edit config in WrinkleFree-1.58Quant/configs/training/stage2_pretrain.yaml

# Or use more GPUs
sky jobs launch skypilot/train.yaml \
  -e ACCELERATOR=H100:8  # Double GPUs
```

### Slow Data Loading

Check GPU utilization. If low (<70%), data loading is the bottleneck:

```python
# Increase dataloader workers in config
training:
  dataloader:
    num_workers: 8  # Increase (default: 4)
    prefetch_factor: 4  # Increase (default: 2)
```

### Checkpoint Upload Failures

For RunPod's 40GB disk limit:

```yaml
# Use MOUNT_CACHED to avoid filling disk
file_mounts:
  /checkpoint:
    mode: MOUNT_CACHED  # Async upload, doesn't fill disk
```

For GCS authentication issues:

```bash
# Verify credentials
gcloud auth list
gcloud config set project YOUR_PROJECT_ID

# Re-authenticate
gcloud auth activate-service-account \
  --key-file=credentials/gcp-service-account.json
```

### W&B Not Logging

```bash
# Verify API key is set
echo $WANDB_API_KEY

# Check logs for W&B errors
sky jobs logs <job_id> | grep wandb
```

## Best Practices

1. **Always run smoke test first** before expensive training runs
2. **Monitor GPU utilization** - aim for >80% to ensure efficient use
3. **Use spot instances** for non-urgent training (70% cost savings)
4. **Set appropriate save intervals** (500 steps = good balance)
5. **Use MOUNT_CACHED** for checkpoint storage (9.6x faster uploads)
6. **Pre-download large models** in setup phase to avoid timeouts
7. **Enable W&B logging** for all production runs
8. **Tag experiments** for easy comparison

## Experiments & Ablations

WrinkleFree includes configurations for systematic ablation studies to measure the impact of training techniques.

### Haar-BitNet Ablation

Compares standard BitNet quantization vs. Haar wavelet-enhanced quantization.

**Hypothesis**: Haar wavelets smooth activation patterns, reducing quantization error.

```bash
# Set W&B API key
source credentials/.env

# Launch baseline (Standard BitNet)
sky jobs launch skypilot/ablation_haar.yaml \
  --env EXPERIMENT=baseline \
  --env HAAR_ENABLED=false \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  -y

# Launch experiment (Haar-BitNet)
sky jobs launch skypilot/ablation_haar.yaml \
  --env EXPERIMENT=haar \
  --env HAAR_ENABLED=true \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  -y
```

**Configuration:**
- Model: SmolLM2-135M (fast iteration)
- Pipeline: Stage 1 → Stage 1.9 (2000 steps) → Stage 2 (1000 steps)
- GPU: 1x A100-80GB
- Duration: ~4-6 hours, Cost: ~$10-15 per run
- Expected improvement: 3-6% better perplexity

### Saliency Curriculum Ablation

Compares standard training vs. saliency-based curriculum learning.

**Hypothesis**: Training on high-saliency tokens first improves convergence.

```bash
# Launch baseline (no saliency)
sky jobs launch skypilot/ablation_saliency.yaml \
  --env EXPERIMENT=baseline \
  --env SALIENCY_ENABLED=false \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  -y

# Launch experiment (with saliency)
sky jobs launch skypilot/ablation_saliency.yaml \
  --env EXPERIMENT=saliency \
  --env SALIENCY_ENABLED=true \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  -y
```

**Configuration:**
- Model: Qwen3-4B (larger model for saliency impact)
- Pipeline: Stage 1 → Stage 1.9 (500 steps) → Stage 2 (1000 steps)
- GPU: 1x A100-80GB, Disk: 100GB
- Duration: ~8-12 hours, Cost: ~$20-30 per run
- Expected improvement: 4-5% better loss convergence

### Running Custom Ablations

Template for new experiments - see `skypilot/ablation_haar.yaml` for full example:

```yaml
name: wrinklefree-ablation-custom

resources:
  accelerators: A100-80GB:1
  use_spot: false  # On-demand for reliability
  cloud: runpod

envs:
  MODEL: smollm2_135m
  EXPERIMENT: baseline
  YOUR_FEATURE_FLAG: "false"
  WANDB_PROJECT: wrinklefree-ablation
```

### Experiment Best Practices

1. **Always tag runs** in W&B for easy filtering
2. **Use consistent naming** (`baseline`, `experiment`)
3. **Run multiple seeds** for publishable results (3+ recommended)
4. **Use on-demand instances** (spot preemption adds noise)
5. **Compare on same hardware** (A100 vs H100 affects results)

## Next Steps

- [Architecture Guide](reference/architecture.md) - Training infrastructure details
- [Testing Guide](reference/testing.md) - Validate training pipelines
- [Tutorials](tutorials/training/) - Step-by-step training guides
