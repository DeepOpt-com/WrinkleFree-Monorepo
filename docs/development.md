# Development Guide

## Setting Up Development Environment

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:DeepOpt-com/WrinkleFree-Monorepo.git
cd WrinkleFree-Monorepo

# Install all packages in development mode
uv sync --all-packages

# Verify setup
uv run pytest packages/data_handler/tests/ -v
```

## Adding a New Package

1. Create package directory:
```bash
mkdir -p packages/mypackage/src/mypackage
touch packages/mypackage/src/mypackage/__init__.py
```

2. Create `pyproject.toml`:
```toml
[project]
name = "mypackage"
version = "0.1.0"
description = "My new package"
requires-python = ">=3.10"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]
```

3. Add to workspace (automatic via `packages/*` glob in root `pyproject.toml`)

4. Sync dependencies:
```bash
uv sync --all-packages
```

## Adding Workspace Dependencies

To use another package from the workspace:

```toml
# In packages/mypackage/pyproject.toml
[project]
dependencies = [
    "data-handler",  # The package name
]

[tool.uv.sources]
data-handler = { workspace = true }  # Resolve from workspace
```

## Running Tests

```bash
# All tests
uv run pytest

# Package-specific
uv run --package wrinklefree pytest packages/training/tests/

# With coverage
uv run pytest packages/training/tests/ --cov=wrinklefree --cov-report=html

# Smoke tests only
uv run pytest -m smoke

# Skip GPU tests
uv run pytest -m "not gpu"
```

## Code Quality

```bash
# Linting
uv run ruff check packages/

# Auto-fix
uv run ruff check packages/ --fix

# Type checking
uv run mypy packages/training/src/

# Format check
uv run ruff format packages/ --check
```

## Git Workflow

### Commits

```bash
# Stage changes
git add packages/training/

# Commit with conventional format
git commit -m "feat(training): add Q-Sparse activation sparsity"
git commit -m "fix(inference): correct batch size calculation"
git commit -m "docs(readme): update installation instructions"
```

### Submodules

```bash
# Update submodules
git submodule update --remote --merge

# Check submodule status
git submodule status
```

## Package-Specific Development

### training (wrinklefree)

```bash
# Run Lightning training (recommended)
uv run --package wrinklefree python packages/training/scripts/train_lightning.py \
  model=smollm2_135m training=unified training.max_steps=10

# With auto batch size scaling
uv run --package wrinklefree python packages/training/scripts/train_lightning.py \
  model=smollm2_135m training=unified training.auto_batch_size=true

# Legacy trainer (still supported)
uv run --package wrinklefree python packages/training/scripts/train.py \
  model=smollm2_135m training=stage2_pretrain training.max_steps=10

# Run unit tests
uv run --package wrinklefree pytest packages/training/tests/unit/
```

### data_handler

```bash
# Run all tests
uv run --package data-handler pytest packages/data_handler/tests/

# Test data loading
uv run --package data-handler python -c "from data_handler.data import create_dataloader; print('ok')"
```

### architecture (bitnet-arch)

```bash
# Run all tests
uv run --package bitnet-arch pytest packages/architecture/tests/

# Test layer imports
uv run --package bitnet-arch python -c "from bitnet_arch.layers import BitLinear, BitLinearLRC; print('ok')"
```

### deployer

```bash
# Test CLI
cd packages/deployer
uv run wf --help

# Dry run deployment
uv run wf train --model smollm2_135m --dry-run
```

## CI/CD Considerations

### GitHub Actions Example

```yaml
name: Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - uses: astral-sh/setup-uv@v4

      - run: uv sync --all-packages

      - run: uv run pytest -m "not gpu"

      - run: uv run ruff check packages/
```

### Selective Testing

Only test affected packages:
```bash
# If only training changed
uv run --package wrinklefree pytest packages/training/tests/

# If data_handler changed (affects training)
uv run pytest packages/data_handler/tests/ packages/training/tests/

# If architecture changed (affects training)
uv run pytest packages/architecture/tests/ packages/training/tests/
```

## Debugging Tips

### Import Issues

```bash
# Check package is installed
uv run python -c "import wrinklefree; print(wrinklefree.__file__)"

# Check workspace resolution
uv tree --package wrinklefree | grep data-handler
```

### Dependency Conflicts

```bash
# Show dependency tree
uv tree

# Check for conflicts
uv lock --check
```

## Documentation

Each package should have:
- `README.md` - User-facing documentation
- `CLAUDE.md` - AI assistant context (key files, patterns)
- `docs/` - Additional documentation (optional)

Root-level docs:
- `README.md` - Monorepo overview
- `CLAUDE.md` - AI navigation guide
- `docs/architecture.md` - System design
- `docs/quick-start.md` - Getting started
- `docs/dependencies.md` - Dependency graph
- `docs/development.md` - This file
