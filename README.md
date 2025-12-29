# WrinkleFree Monorepo

Research platform for 1.58-bit (ternary) quantized LLM training, serving, and evaluation.

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:DeepOpt-com/WrinkleFree-Monorepo.git
cd WrinkleFree-Monorepo

# Install all packages
uv sync --all-packages

# Run tests
uv run pytest

# Run training (example)
uv run --package wrinklefree python packages/training/scripts/train.py model=smollm2_135m training=stage2_pretrain
```

## Structure

```
WrinkleFree-Monorepo/
├── packages/
│   ├── training/          # 1.58-bit training (BitDistill) - App
│   ├── cheapertraining/   # Shared data layer & utilities - Library
│   ├── fairy2/            # Complex-valued quantization - App
│   ├── inference/         # Serving (sglang-bitnet) - App
│   ├── eval/              # Model evaluation - App
│   ├── deployer/          # Cloud deployment (Modal/SkyPilot) - App
│   └── converter/         # DLM format conversion - App
├── extern/
│   └── BitNet/            # Microsoft BitNet.cpp (submodule)
├── pyproject.toml         # Workspace root
└── uv.lock                # Unified lockfile
```

## Packages

| Package | Purpose | Key Entry |
|---------|---------|-----------|
| `training` | 1.58-bit quantization training pipeline (BitDistill) | `scripts/train.py` |
| `cheapertraining` | Data loading, influence functions, mixture optimization | Imported as library |
| `fairy2` | Complex-valued neural network quantization | `scripts/train.py` |
| `inference` | Model serving with sglang-bitnet | `demo/serve_sglang.py` |
| `eval` | GLUE, CNN/DailyMail benchmarks | `scripts/evaluate.py` |
| `deployer` | SkyPilot/Modal cloud deployment | `wf` CLI |
| `converter` | Convert models to Fast-dLLM format | `scripts/train_dlm.py` |

## Common Commands

```bash
# Install all dependencies
uv sync --all-packages

# Install single package
uv sync --package wrinklefree

# Run in package context
uv run --package wrinklefree python scripts/train.py

# Add dependency to package
cd packages/training && uv add torch

# Run tests for specific package
uv run --package wrinklefree pytest packages/training/tests/
```

## Shared Dependencies

`cheapertraining` is a shared library imported by:
- `training` - for data loading and influence-based optimization
- `fairy2` - for data loading utilities

Use workspace dependencies:
```toml
# In pyproject.toml
[tool.uv.sources]
cheapertraining = { workspace = true }
```

## Documentation

- [Quick Start Guide](docs/quick-start.md)
- [Architecture Overview](docs/architecture.md)
- [Development Guide](docs/development.md)
- [Dependency Graph](docs/dependencies.md)

Each package also has its own `README.md` and `CLAUDE.md` with package-specific details.

## Development

See [docs/development.md](docs/development.md) for:
- Adding new packages
- Adding workspace dependencies
- Cross-package testing
- CI/CD setup

## References

- [BitDistill Paper](https://arxiv.org/abs/2510.13998) - 1.58-bit training approach
- [Microsoft BitNet](https://github.com/microsoft/BitNet) - Inference engine
- [Fast-dLLM](https://arxiv.org/abs/2512.14067) - Diffusion LLM conversion
