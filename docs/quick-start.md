# Quick Start Guide

## Getting Started Guides

| Guide | Description |
|-------|-------------|
| [Training Getting Started](guides/training-getting-started.md) | Cloud training with SkyPilot |
| [Inference Getting Started](guides/inference-getting-started.md) | Run inference with wf_server |
| [Cloud Deployment](guides/cloud-deployment.md) | Full SkyPilot and credentials setup |

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Git with LFS support

## Installation

### Clone the Repository

```bash
git clone --recurse-submodules git@github.com:DeepOpt-com/WrinkleFree-Monorepo.git
cd WrinkleFree-Monorepo
```

If you forgot `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

### Install Dependencies

```bash
# Install all packages
uv sync --all-packages

# Or install a specific package
uv sync --package wf-train
```

## Verify Installation

```bash
# Check imports work
uv run --package wf-train python -c "import wf_train; print('training: ok')"
uv run --package wf-data python -c "import wf_data; print('data: ok')"
uv run --package wf-arch python -c "import wf_arch; print('arch: ok')"

# Run tests
uv run pytest packages/data_handler/tests/ -v --tb=short
```

## Common Tasks

### Training a Model

#### PyTorch Lightning (Recommended)

```bash
# SmolLM2-135M with Lightning trainer
uv run --package wf-train python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=base

# With auto batch size scaling (single GPU only - not supported with DDP/FSDP!)
uv run --package wf-train python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=base \
  training.auto_batch_size=true

# With limited steps for smoke test
uv run --package wf-train python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=base \
  training.max_steps=100
```

### Running Distillation

Distillation is integrated into the training package via the objectives system:

```bash
# BitDistill distillation (logits + attention)
uv run --package wf-train python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=bitdistill_full

# LRC Calibration (post-quantization low-rank correction)
uv run --package wf-train python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=lrc_calibration
```

### Running Evaluation

```bash
uv run --package wf-eval python packages/eval/scripts/run_eval.py \
  --model-path outputs/smollm2_135m/checkpoint \
  --benchmark glue
```

### Deploying to Cloud

```bash
cd packages/deployer
source credentials/.env

# Run smoke test to verify setup
wf smoke

# Launch training on SkyPilot (default)
wf train -m smollm2_135m -t base

# With larger GPU configuration
wf train -m qwen3_4b -t base --scale large
```

See [Training Getting Started](guides/training-getting-started.md) for more details.

### Converting Models to GGUF

```bash
cd packages/inference

# Convert checkpoint to GGUF (I2_S recommended)
python scripts/convert_checkpoint_to_gguf.py \
    path/to/checkpoint \
    --outfile model.gguf \
    --outtype i2_s
```

**Warning**: Never use TQ2_0 for bf16 checkpoints - it corrupts weights!

See [Inference Getting Started](guides/inference-getting-started.md) for running the model.

## Package Commands

| Task | Command |
|------|---------|
| Install all | `uv sync --all-packages` |
| Install one | `uv sync --package wf-train` |
| Run in context | `uv run --package wf-train python ...` |
| Add dependency | `cd packages/training && uv add torch` |
| Run tests | `uv run pytest packages/training/tests/` |
| Type check | `uv run mypy packages/training/src/` |
| Lint | `uv run ruff check packages/` |

## Troubleshooting

### "Package not found" errors

```bash
# Reinstall all packages
uv sync --all-packages --reinstall
```

### Submodule issues

```bash
# Reset submodules
git submodule deinit -f .
git submodule update --init --recursive
```

### Import errors between packages

Ensure workspace dependencies are configured:
```toml
# In packages/training/pyproject.toml
[tool.uv.sources]
wf-data = { workspace = true }
wf-arch = { workspace = true }
```

## Next Steps

- **Training**: [Training Getting Started](guides/training-getting-started.md)
- **Inference**: [Inference Getting Started](guides/inference-getting-started.md)
- **Cloud Setup**: [Cloud Deployment Guide](guides/cloud-deployment.md)
- **Architecture**: [Architecture Overview](architecture.md)
- **Contributing**: [Development Guide](development.md)
