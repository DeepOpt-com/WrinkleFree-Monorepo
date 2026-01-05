# Configuration Guide

This guide helps you navigate the configuration system. WrinkleFree uses [Hydra](https://hydra.cc/) for configuration management.

## Directory Structure

Configurations are located in `configs/`:

- **`model/`**: Model architectures (e.g., `smollm2_135m`, `qwen3_4b`)
- **`training/`**: Training configs (e.g., `base`, `bitdistill_full`, `lrc_calibration`)
- **`distributed/`**: Hardware strategies (e.g., `single_gpu`, `fsdp_multi`)
- **`data/`**: Dataset configs (points to data_handler package)

## Step 1: Choose Your Model (`model=...`)

| Model | Config | Params | VRAM | Use Case |
|-------|--------|--------|------|----------|
| **SmolLM2-135M** | `smollm2_135m` | 135M | ~4GB | **Start Here**. Testing, debugging |
| **Qwen2-0.5B** | `qwen2_0.5b` | 0.5B | ~8GB | Development |
| **Qwen3-4B** | `qwen3_4b` | 4B | ~24GB | **Recommended**. Production training |

## Step 2: Choose Training Config (`training=...`)

| Config | Purpose | When to Use |
|--------|---------|-------------|
| **`base`** | Combined STE + DLM training | **Recommended** - production training |
| `bitdistill_full` | Knowledge distillation | Teacher-student distillation |
| `lrc_calibration` | Low-rank correction | Post-quantization recovery |
| `stage2_pretrain` | Continue pretraining | Legacy stage-based training |
| `stage1_subln` | SubLN insertion | Model conversion only |
| `stage1_9_layerwise` | Layer-wise distillation | Hidden state alignment |

### Unified Training (Recommended)

Combines STE quantization with DLM objectives in a single pass:

```bash
uv run python scripts/train_lightning.py model=smollm2_135m training=base
```

**Features**:
- Auto-converts model to BitNet if needed
- Multi-task: LM loss + DLM masking loss
- Curriculum phases ramp up DLM weight
- MuonClip optimizer with QK clipping
- Influence-based data remixing (optional)

### LRC Calibration

Post-training recovery using Low-Rank Correction:

```bash
uv run python scripts/train_lightning.py model=smollm2_135m training=lrc_calibration
```

**Features**:
- Only U, V matrices train (~10% of hidden dim)
- All other params frozen
- Short calibration run (~50M tokens)

## Step 3: Choose Hardware Strategy (`distributed=...`)

| Config | Description |
|--------|-------------|
| `single_gpu` | Standard training. Use for SmolLM2 or debugging |
| `fsdp_multi` | Fully Sharded Data Parallel. Required for 4B+ models |

## Common Recipes

### Quick Test (Single GPU)

```bash
uv run python scripts/train_lightning.py \
    model=smollm2_135m \
    training=base \
    training.max_steps=100
```

### Production Training (Multi-GPU)

```bash
uv run python scripts/train_lightning.py \
    model=qwen3_4b \
    training=base \
    distributed=fsdp_multi \
    training.auto_batch_size=true
```

### With Influence-Based Data Remixing

```bash
uv run python scripts/train_lightning.py \
    model=smollm2_135m \
    training=base \
    data.config_name=mixed_pretrain \
    training.influence.enabled=true \
    training.influence.warmup_steps=1000
```

### Resume from Checkpoint

```bash
uv run python scripts/train_lightning.py \
    training=base \
    training.resume.checkpoint_path=gs://bucket/checkpoint.pt \
    training.resume.load_optimizer_state=false
```

## How to Override Defaults

Override any parameter via command line:

```bash
# Change learning rate and batch size
uv run python scripts/train_lightning.py \
    model=smollm2_135m \
    training.lr=5e-5 \
    training.batch_size=16

# Enable GCS checkpointing
uv run python scripts/train_lightning.py \
    model=smollm2_135m \
    training=base \
    gcs.enabled=true \
    gcs.bucket=wrinklefree-checkpoints

# Disable WandB logging
uv run python scripts/train_lightning.py \
    model=smollm2_135m \
    training.logging.wandb.enabled=false
```

## Key Config Options

### Training

```yaml
training:
  max_steps: 100000           # Total training steps
  batch_size: 32              # Per-GPU batch size
  auto_batch_size: true       # Auto-find max batch size
  gradient_accumulation_steps: 4
  lr: 2.4e-3                  # Learning rate

  optimizer:
    type: muonclip            # muonclip, adamw

  influence:
    enabled: true             # Enable data remixing
    warmup_steps: 1000
    update_interval: 5000
```

### Objectives (in unified.yaml)

```yaml
objectives:
  continue_pretrain:
    enabled: true
    weight: 1.0
  dlm:
    enabled: true
    weight: 0.5
    mask_ratio: 0.15
```

### Curriculum Phases

```yaml
curriculum:
  phases:
    - name: warmup
      end_step_fraction: 0.1
      objectives: {continue_pretrain: 1.0, dlm: 0.0}
    - name: main
      end_step_fraction: 0.8
      objectives: {continue_pretrain: 1.0, dlm: 0.5}
```

## FAQ

**Q: I'm getting Out of Memory (OOM) errors.**

1. Enable auto batch size: `training.auto_batch_size=true`
2. Switch to smaller model: `model=smollm2_135m`
3. Reduce batch size: `training.batch_size=1`
4. Use FSDP: `distributed=fsdp_multi`

**Q: Where are my checkpoints?**

Check `outputs/` or the GCS bucket if `gcs.enabled=true`.

**Q: How do I use a different dataset?**

Override `data.config_name` to use a different config from data_handler:

```bash
data.config_name=fineweb  # Use data_handler's fineweb.yaml
```

**Q: How do I disable DLM and just do CE training?**

```bash
training.objectives.dlm.enabled=false
```
