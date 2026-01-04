# Quick Start Guide

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
uv sync --package wrinklefree
```

## Verify Installation

```bash
# Check imports work
uv run --package wrinklefree python -c "import wrinklefree; print('training: ok')"
uv run --package data-handler python -c "import data_handler; print('data_handler: ok')"
uv run --package bitnet-arch python -c "import bitnet_arch; print('bitnet_arch: ok')"

# Run tests
uv run pytest packages/data_handler/tests/ -v --tb=short
```

## Common Tasks

### Training a Model

#### PyTorch Lightning (Recommended)

```bash
# SmolLM2-135M with Lightning trainer
uv run --package wrinklefree python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=unified

# With auto batch size scaling (finds max batch that fits GPU)
uv run --package wrinklefree python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=unified \
  training.auto_batch_size=true

# With limited steps for smoke test
uv run --package wrinklefree python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=unified \
  training.max_steps=100
```

### Running Distillation

Distillation is integrated into the training package via the objectives system:

```bash
# BitDistill distillation (logits + attention)
uv run --package wrinklefree python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=bitdistill_full

# LRC Calibration (post-quantization low-rank correction)
uv run --package wrinklefree python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=lrc_calibration
```

### Running Evaluation

```bash
uv run --package wrinklefree-eval python packages/eval/scripts/evaluate.py \
  --model-path outputs/smollm2_135m/checkpoint \
  --benchmark glue
```

### Deploying to Cloud

```bash
cd packages/deployer

# Launch training on Modal
wf train --model smollm2_135m --stage 2 --scale dev

# Launch on SkyPilot
wf sky launch --config skypilot/train.yaml
```

### Converting Models to GGUF

See the root `CLAUDE.md` for the correct DLM GGUF conversion workflow. Key point: use I2_S format for bf16 DLM checkpoints (NOT TQ2_0).

```bash
# Convert using the inference package converter
python packages/inference/scripts/convert_checkpoint_to_gguf.py \
    path/to/checkpoint \
    --outfile model.gguf \
    --outtype i2_s
```

## Package Commands

| Task | Command |
|------|---------|
| Install all | `uv sync --all-packages` |
| Install one | `uv sync --package wrinklefree` |
| Run in context | `uv run --package wrinklefree python ...` |
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
data-handler = { workspace = true }
bitnet-arch = { workspace = true }
```

## Next Steps

- Read [Architecture Overview](architecture.md) for system design
- See [Development Guide](development.md) for contributing
- Check individual package READMEs for detailed usage
