# SkyPilot Configuration

This directory contains SkyPilot configuration for:
- **Training**: Managed jobs for model training (`train.yaml`, smoke tests, ablations)
- **Inference**: SkyServe service for model serving (`service.yaml`)

## Overview

We use **SkyPilot Managed Jobs** for training and **SkyServe** for inference because they provide:
- Automatic spot recovery and checkpoint resumption
- Unified endpoint across all replicas
- Automatic load balancing and autoscaling
- Health checks and replica replacement
- Multi-infrastructure support (RunPod + Hetzner + AWS + GCP)

## Files

| File | Purpose |
|------|---------|
| `train.yaml` | Multi-stage training job (Stage 1, 1.9, 2, 3) |
| `dlm_train.yaml` | Fast-dLLM v2 SFT training for 2.5x faster inference |
| `smoke_test.yaml` | Quick validation (5 min, ~$1) |
| `ablation_haar.yaml` | Haar-BitNet ablation study |
| `ablation_saliency.yaml` | Saliency curriculum ablation study |
| `service.yaml` | SkyServe inference service definition |
| `ssh_node_pools.yaml.example` | Template for registering Hetzner servers |

---

## Training Jobs

Train WrinkleFree models on cloud GPUs using SkyPilot managed jobs. Managed jobs automatically restart on spot preemption and recover from checkpoints.

### Prerequisites

1. **RunPod API Key** (primary provider):
   ```bash
   # Add to ~/.bashrc or source before use
   export RUNPOD_API_KEY=your_key_here
   ```

2. **Checkpoint Bucket** (S3 or GCS):
   ```bash
   # Create bucket (example for AWS)
   aws s3 mb s3://wrinklefree-checkpoints

   # Or GCS
   gsutil mb gs://wrinklefree-checkpoints
   ```

3. **Verify SkyPilot setup**:
   ```bash
   sky check
   # Should show: RunPod: enabled, AWS/GCP: enabled (for bucket access)
   ```

### Launch Training

```bash
# Stage 2: Continue pretraining (most common)
sky jobs launch train.yaml -e MODEL=qwen3_4b -e STAGE=2

# Stage 3: Distillation fine-tuning
sky jobs launch train.yaml -e MODEL=qwen3_4b -e STAGE=3

# Custom GPU configuration
sky jobs launch train.yaml -e MODEL=qwen3_4b -e STAGE=2 -e ACCELERATOR=A100:8

# SmolLM2 (smaller model, cheaper GPUs)
sky jobs launch train.yaml -e MODEL=smollm2_135m -e STAGE=2 -e ACCELERATOR=A100:1
```

### Monitor Training

```bash
# List all jobs
sky jobs queue

# View logs (streaming)
sky jobs logs <job_id>

# View logs (last 100 lines)
sky jobs logs <job_id> --tail 100

# Cancel job
sky jobs cancel <job_id>
```

### Checkpoints

Checkpoints are automatically saved to your S3/GCS bucket at `/checkpoint/{MODEL}/stage{STAGE}/`.

```bash
# List checkpoints
aws s3 ls s3://wrinklefree-checkpoints/qwen3_4b/stage2/

# Download final checkpoint
aws s3 sync s3://wrinklefree-checkpoints/qwen3_4b/stage2/ ./checkpoints/
```

Training automatically resumes from the latest checkpoint if preempted.

### Cost Estimates (RunPod)

| Stage | GPUs | Duration | Cost |
|-------|------|----------|------|
| Stage 2 (10B tokens) | 4x H100 | ~24h | ~$200 |
| Stage 3 (distill) | 4x H100 | ~8h | ~$70 |
| Stage 2 (SmolLM2) | 1x A100 | ~4h | ~$10 |

*Costs are approximate and depend on spot availability.*

---

## Fast-dLLM v2 Training

Train models with Fast-dLLM v2 SFT recipe for 2.5x faster inference at generation time.

### What is Fast-dLLM v2?

Fast-dLLM v2 (arXiv:2509.26328) enables parallel token generation using block diffusion:
- Train with SFT on conversations (response-only loss)
- At inference, generate multiple tokens in parallel within blocks
- 2.5x speedup with no quality loss

### Launch DLM Training

```bash
# From WrinkleFree-Deployer directory
sky launch skypilot/dlm_train.yaml -y \
    --env MODEL=1bitLLM/bitnet_b1_58-large \
    --env TOKENS=1000000000 \
    --env WANDB_API_KEY

# With custom GPU config
sky launch skypilot/dlm_train.yaml -y \
    --accelerators H100:8 \
    --env MODEL=1bitLLM/bitnet_b1_58-large
```

### Monitor

```bash
# View logs
sky logs wf-dlm-train

# Check GCS for checkpoints
gsutil ls gs://wrinklefree-checkpoints/dlm/
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL` | `1bitLLM/bitnet_b1_58-large` | HuggingFace model path |
| `TOKENS` | `1000000000` | Total training tokens (1B) |
| `BLOCK_SIZE` | `32` | Block diffusion size (bd_size) |
| `BATCH_SIZE` | `4` | Per-GPU batch size |
| `GRAD_ACCUM` | `16` | Gradient accumulation steps |

### Cost Estimate

| Model | GPUs | Duration | Cost |
|-------|------|----------|------|
| BitNet 2B (1B tokens) | 1x H100 | ~4h | ~$15 |
| BitNet 2B (4B tokens) | 8x H100 | ~4h | ~$60 |

---

### Ablation Studies

Run controlled experiments to validate training techniques.

#### Haar-BitNet Ablation

Compare standard BitNet vs. Haar wavelet-enhanced quantization:

```bash
# Set W&B API key
source credentials/.env

# Launch baseline (standard BitNet)
sky jobs launch ablation_haar.yaml \
  --env EXPERIMENT=baseline \
  --env HAAR_ENABLED=false \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  -y

# Launch experiment (Haar-BitNet)
sky jobs launch ablation_haar.yaml \
  --env EXPERIMENT=haar \
  --env HAAR_ENABLED=true \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  -y
```

**What it tests:**
- Model: SmolLM2-135M (fast iteration)
- Pipeline: Stage 1 → Stage 1.9 (2000 steps) → Stage 2 (1000 steps)
- GPU: 1x A100-80GB
- Duration: ~4-6 hours
- Cost: ~$10-15 per run
- Tracking: W&B project `wrinklefree-ablation`

**Expected improvement:** 3-6% better perplexity with Haar wavelets

#### Saliency Curriculum Ablation

Compare standard training vs. saliency-based curriculum learning:

```bash
# Set W&B API key
source credentials/.env

# Launch baseline (no saliency)
sky jobs launch ablation_saliency.yaml \
  --env EXPERIMENT=baseline \
  --env SALIENCY_ENABLED=false \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  -y

# Launch experiment (with saliency)
sky jobs launch ablation_saliency.yaml \
  --env EXPERIMENT=saliency \
  --env SALIENCY_ENABLED=true \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  -y
```

**What it tests:**
- Model: Qwen3-4B (larger model for saliency impact)
- Pipeline: Stage 1 → Stage 1.9 (500 steps) → Stage 2 (1000 steps)
- GPU: 1x A100-80GB
- Disk: 100GB (larger model)
- Duration: ~8-12 hours
- Cost: ~$20-30 per run
- Tracking: W&B project `wrinklefree-ablation`

**Expected improvement:** 4-5% better loss convergence with saliency curriculum

#### Monitor Ablations

```bash
# Check job status
sky jobs queue

# View logs
sky jobs logs <job_id> --follow

# View results in W&B
# https://wandb.ai/wrinklefree-ablation
```

See [Experiments Guide](../docs/experiments.md) for full ablation documentation.

---

## Inference (SkyServe)

## Quick Start

### 1. Configure Hetzner SSH Node Pool

```bash
# Copy and edit with your Hetzner server IPs
cp ssh_node_pools.yaml.example ~/.sky/ssh_node_pools.yaml

# Edit the file with your server IPs from Terraform output
vim ~/.sky/ssh_node_pools.yaml

# Initialize the node pool (installs SkyPilot runtime on nodes)
sky ssh up

# Verify nodes are available
sky check ssh
```

### 2. Deploy SkyServe Service

```bash
# Deploy the service (this starts SkyServe)
sky serve up service.yaml --name wrinklefree

# Wait for replicas to become healthy (may take 2-5 minutes)
watch sky serve status wrinklefree
```

### 3. Get Service Endpoint

```bash
# Get the endpoint URL
sky serve status wrinklefree

# Example output:
# Service: wrinklefree
# Endpoint: https://wrinklefree-xxxx.sky.serve
# Replicas: 3/3 ready
```

### 4. Test the Service

```bash
# Health check
curl https://wrinklefree-xxxx.sky.serve/health

# Inference request
curl https://wrinklefree-xxxx.sky.serve/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
```

## SkyServe Commands Reference

```bash
# Deploy/update service
sky serve up service.yaml --name wrinklefree

# Check service status
sky serve status wrinklefree

# Check all replicas (detailed view)
sky serve status wrinklefree --all

# View service logs
sky serve logs wrinklefree

# View logs for specific replica
sky serve logs wrinklefree --replica-id 0

# Scale service manually
sky serve update wrinklefree --min-replicas 5

# Tear down service
sky serve down wrinklefree

# Tear down all services
sky serve down --all
```

## Configuration

### Autoscaling

The `replica_policy` in `service.yaml` controls autoscaling:

```yaml
replica_policy:
  min_replicas: 3         # Always keep 3 replicas (Hetzner base)
  max_replicas: 20        # Scale up to 20 during peaks
  target_qps_per_replica: 5.0  # Target queries/second per replica
  upscale_delay_seconds: 60    # Wait 60s before adding replicas
  downscale_delay_seconds: 300 # Wait 5min before removing replicas
```

**Tuning tips:**
- `min_replicas`: Set to your Hetzner node count (always-on base)
- `max_replicas`: Set based on budget for cloud burst
- `target_qps_per_replica`: Lower = more replicas, better latency
- `downscale_delay_seconds`: Higher = more stable, but higher cost

### Resource Requirements

SkyServe places replicas on the cheapest infrastructure that meets requirements:

```yaml
resources:
  cpus: 16+
  memory: 128+
  disk_size: 100
  use_spot: true
```

**Placement priority:**
1. SSH Node Pool (Hetzner) - $0/hr marginal cost
2. Cheapest spot instances meeting requirements
3. On-demand if spot unavailable

### Readiness Probe

The readiness probe determines when replicas can receive traffic:

```yaml
readiness_probe:
  path: /health
  initial_delay_seconds: 120  # Wait for model to load
  timeout_seconds: 30
  period_seconds: 10
```

**If replicas aren't becoming healthy:**
- Increase `initial_delay_seconds` for larger models
- Check logs: `sky serve logs wrinklefree --replica-id 0`

## Monitoring

### Check Replica Distribution

```bash
# See which infrastructure each replica is on
sky serve status wrinklefree --all

# Example output:
# Replica 0: ssh/hetzner-base (10.100.1.1) - READY
# Replica 1: ssh/hetzner-base (10.100.1.2) - READY
# Replica 2: ssh/hetzner-base (10.100.1.3) - READY
# Replica 3: aws (us-east-1, r7a.xlarge) - READY  <- Burst replica
```

### Metrics

SkyServe exposes metrics at the controller:

```bash
# Get controller status
sky serve status

# Key metrics to watch:
# - QPS per replica
# - Replica count over time
# - Latency percentiles
```

## Troubleshooting

### Service won't start

```bash
# Check controller logs
sky serve logs wrinklefree --controller

# Check if SSH Node Pool is accessible
sky check ssh
```

### Replicas not becoming healthy

```bash
# Check replica logs
sky serve logs wrinklefree --replica-id 0

# SSH into replica for debugging
sky serve ssh wrinklefree --replica-id 0
```

### High latency

```bash
# Check if replicas are overloaded
sky serve status wrinklefree --all

# Scale up manually
sky serve update wrinklefree --min-replicas 5
```

### Spot interruptions

SkyServe automatically handles spot interruptions with `spot_recovery: FAILOVER`.
Check status to see if replicas are being replaced:

```bash
sky serve status wrinklefree --all
```

## Production Checklist

### Training
- [ ] Cloud credentials configured (RunPod, AWS/GCP)
- [ ] Checkpoint bucket created (S3 or GCS)
- [ ] W&B API key set for experiment tracking
- [ ] Smoke test passes (`smoke_test.yaml`)
- [ ] Training config reviewed (model, stage, steps)

### Serving
- [ ] Hetzner nodes registered in SSH Node Pool
- [ ] `sky check` shows all clouds enabled
- [ ] Model files accessible (local, S3, or HuggingFace)
- [ ] `min_replicas` set to Hetzner node count
- [ ] `target_qps_per_replica` tuned based on benchmarks
- [ ] Monitoring/alerting configured for SkyServe endpoint
- [ ] Cloudflare (optional) configured to proxy SkyServe endpoint

## Additional Resources

- [Training Guide](../docs/training.md) - Complete training documentation
- [Experiments Guide](../docs/training.md#experiments--ablations) - Ablation studies and research
- [Architecture Guide](../docs/reference/architecture.md) - Training + serving architecture
- [Testing Guide](../docs/reference/testing.md) - Validation and smoke tests
