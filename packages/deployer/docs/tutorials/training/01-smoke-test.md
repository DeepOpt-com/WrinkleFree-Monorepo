# Tutorial: Run Your First Training Smoke Test

**Time**: 15 minutes
**Cost**: ~$1
**Goal**: Validate the training pipeline works before running expensive jobs

## What You'll Learn

- How to launch a SkyPilot training job
- How to monitor training progress
- How to verify checkpoints are saved correctly

## Prerequisites

1. **SkyPilot installed and configured**:
   ```bash
   uv sync
   sky check  # Should show at least one cloud enabled
   ```

2. **RunPod API key** (or another GPU provider):
   ```bash
   export RUNPOD_API_KEY=your_key_here
   ```

3. **Checkpoint storage** (optional but recommended):
   ```bash
   # AWS S3
   aws s3 mb s3://your-checkpoint-bucket

   # Or GCS
   gsutil mb gs://your-checkpoint-bucket
   ```

## Step 1: Review the Smoke Test Config

The smoke test is defined in `skypilot/smoke_test.yaml`. Let's understand what it does:

```yaml
# Key settings
resources:
  accelerators: A100-80GB:1  # Single GPU
  disk_size: 40              # Minimal disk
  use_spot: false            # On-demand for reliability

# What it runs
run: |
  # Stage 1: Model conversion (seconds)
  # Stage 1.9: 100 training steps (~5 min)
  # Checkpoint upload (if configured)
```

The smoke test uses **SmolLM2-135M**, a tiny model that fits easily in memory and trains quickly.

## Step 2: Launch the Smoke Test

```bash
# Navigate to the deployer directory
cd WrinkleFree-Deployer

# Launch the smoke test
sky launch skypilot/smoke_test.yaml -y --cluster smoke-test
```

**What happens:**
1. SkyPilot finds an available A100 GPU (usually RunPod)
2. Provisions the instance (~1-2 min)
3. Runs setup (installs dependencies, ~2-3 min)
4. Runs the training script

## Step 3: Monitor Progress

```bash
# Stream logs in real-time
sky logs smoke-test --follow
```

**Expected output:**

```
=== Stage 1: SubLN Insertion ===
Loading model: HuggingFaceTB/SmolLM2-135M
Inserting SubLN layers...
Stage 1 complete! Saved to /tmp/checkpoints/stage1_model

=== Stage 1.9: Layer-wise Distillation ===
Starting distillation (100 steps)...
Step 10: loss=2.45, lr=5e-5
Step 20: loss=2.32, lr=4.8e-5
Step 30: loss=2.21, lr=4.5e-5
...
Step 100: loss=1.98, lr=2e-5

=== Smoke Test Complete! ===
Checkpoints saved to: /tmp/checkpoints/smoke_test/
```

## Step 4: Verify Checkpoints

If you configured checkpoint storage:

```bash
# List checkpoints (AWS)
aws s3 ls s3://your-checkpoint-bucket/checkpoints/smoke-test/

# Or (GCS)
gsutil ls gs://your-checkpoint-bucket/checkpoints/smoke-test/
```

You should see:
```
checkpoint-50/
checkpoint-100/
```

## Step 5: Clean Up

```bash
# Terminate the cluster
sky down smoke-test -y

# Verify it's gone
sky status
```

## Understanding the Output

### GPU Utilization Logs

During training, you'll see GPU stats:

```
=== GPU Utilization ===
utilization.gpu: 85%
memory.used: 45000 MiB / 81920 MiB
power.draw: 350 W
```

**What to look for:**
- **GPU util > 70%**: Good, training is efficient
- **GPU util < 50%**: Data loading bottleneck
- **Memory near max**: May need larger GPU or smaller batch

### Training Metrics

```
Step 100: loss=1.98, lr=2e-5, tokens/s=1200
```

- **loss**: Should decrease over time
- **lr**: Learning rate (follows schedule)
- **tokens/s**: Training throughput

## Troubleshooting

### "No resources found"

```bash
# Check available GPUs
sky show-gpus

# Try a different cloud
sky launch skypilot/smoke_test.yaml --cloud aws
```

### "Out of memory"

The smoke test uses a tiny model, so OOM is unlikely. If it happens:

```bash
# Check the logs for error details
sky logs smoke-test --tail 50
```

### "Checkpoint upload failed"

```bash
# Verify credentials
aws sts get-caller-identity  # For AWS
gcloud auth list             # For GCP

# Check bucket exists
aws s3 ls s3://your-bucket/
```

## What's Next?

Now that you've verified the pipeline works:

1. **Full training**: See [Training Guide](../../training.md) for Stage 2+ training
2. **Monitor with W&B**: Add `WANDB_API_KEY` for experiment tracking
3. **Try larger models**: Scale up to Qwen3-4B or larger

## Cost Summary

| Resource | Duration | Cost |
|----------|----------|------|
| A100-80GB (RunPod) | ~10 min | ~$0.50 |
| Instance startup | ~5 min | ~$0.25 |
| **Total** | ~15 min | **~$0.75-1.00** |
