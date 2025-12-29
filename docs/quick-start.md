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
uv run --package cheapertraining python -c "import cheapertraining; print('cheapertraining: ok')"

# Run tests
uv run pytest packages/cheapertraining/tests/ -v --tb=short
```

## Common Tasks

### Training a Model

```bash
# SmolLM2-135M (smallest, good for testing)
uv run --package wrinklefree python packages/training/scripts/train.py \
  model=smollm2_135m \
  training=stage2_pretrain \
  data=fineweb

# With limited steps for smoke test
uv run --package wrinklefree python packages/training/scripts/train.py \
  model=smollm2_135m \
  training=stage2_pretrain \
  training.max_steps=100
```

### Running Evaluation

```bash
uv run --package wrinklefree_eval python packages/eval/scripts/evaluate.py \
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

### Converting Models

```bash
# Convert to DLM format
uv run --package wf_dlm_converter python packages/converter/scripts/train_dlm.py \
  model=smollm2_135m \
  source.path=outputs/checkpoint
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
cheapertraining = { workspace = true }
```

## Next Steps

- Read [Architecture Overview](architecture.md) for system design
- See [Development Guide](development.md) for contributing
- Check individual package READMEs for detailed usage
