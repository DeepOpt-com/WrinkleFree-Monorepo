# Experiments and Reproduction

How to reproduce key results and run experiments.

## Environment Setup

```bash
# From monorepo root
uv sync --all-packages
```

## Experiment 1: Unified Training (SmolLM2-135M)

The recommended training approach combining CE + DLM objectives.

### Quick Test (100 steps)

```bash
uv run python scripts/train_lightning.py \
    model=smollm2_135m \
    training=unified \
    training.max_steps=100
```

**Expected**: Loss should decrease from ~10-12 to ~6-8.

### Full Training

```bash
uv run python scripts/train_lightning.py \
    model=smollm2_135m \
    training=unified \
    training.auto_batch_size=true \
    gcs.enabled=true \
    gcs.bucket=wrinklefree-checkpoints
```

## Experiment 2: Influence-Based Data Remixing

Verify the impact of dynamic dataset weight optimization.

### With Influence (Recommended)

```bash
uv run python scripts/train_lightning.py \
    model=smollm2_135m \
    training=unified \
    data.config_name=mixed_pretrain \
    training.influence.enabled=true \
    training.influence.warmup_steps=1000 \
    training.influence.update_interval=5000
```

### Without Influence (Baseline)

```bash
uv run python scripts/train_lightning.py \
    model=smollm2_135m \
    training=unified \
    data.config_name=mixed_pretrain \
    training.influence.enabled=false
```

**Comparison**: The influence-guided run should achieve lower validation loss on target domains.

## Experiment 3: LRC Calibration

Post-quantization recovery using Low-Rank Correction.

```bash
uv run python scripts/train_lightning.py \
    model=smollm2_135m \
    training=lrc_calibration \
    training.max_steps=5000
```

**Expected**: Recovery of quantization errors without full retraining.

## Experiment 4: Knowledge Distillation

Teacher-student distillation using BitDistill objectives.

```bash
uv run python scripts/train_lightning.py \
    model=smollm2_135m \
    training=bitdistill_full
```

## Cloud Deployment

For larger experiments, use SkyPilot via the deployer package:

```bash
cd packages/deployer
source credentials/.env

# Launch smoke test
sky launch skypilot/smoke_test_lightning.yaml -y --cluster smoke

# Monitor
sky logs smoke

# Teardown
sky down smoke -y
```

## Troubleshooting

**OOM Errors**:
- Enable auto batch size: `training.auto_batch_size=true`
- Reduce batch size: `training.batch_size=1`
- Use FSDP: `distributed=fsdp_multi`

**Divergence**:
- Check learning rate (lower it)
- Ensure warmup steps are sufficient
- Try AdamW instead of MuonClip: `training.optimizer.type=adamw`

**Slow Training**:
- Enable auto batch size to maximize GPU utilization
- Use mixed_pretrain data config (pre-tokenized)
