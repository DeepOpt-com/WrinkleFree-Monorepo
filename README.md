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
│   ├── training/          # 1.58-bit training + distillation objectives - App
│   ├── architecture/      # BitNet layers (BitLinear, BitLinearLRC, SubLN) - Library
│   ├── data_handler/      # Shared data layer & utilities - Library
│   ├── inference/         # Serving (sglang-bitnet) - App
│   ├── eval/              # Model evaluation - App
│   ├── deployer/          # Cloud deployment (SkyPilot) - App
│   ├── mobile/            # Android inference - App
│   └── _legacy/           # Archived packages (do not use)
│       ├── distillation/      # (integrated into training)
│       ├── converter/         # (functionality moved)
│       └── cheapertraining/   # (renamed to data_handler)
├── extern/
│   └── BitNet/            # Microsoft BitNet.cpp (submodule)
├── pyproject.toml         # Workspace root
└── uv.lock                # Unified lockfile
```

## Packages

| Package | Purpose | Key Entry |
|---------|---------|-----------|
| `training` | 1.58-bit training pipeline + distillation objectives (BitDistill, LRC) | `scripts/train.py` |
| `architecture` | BitNet layers (BitLinear, BitLinearLRC, SubLN) & model conversion | Imported as library |
| `data_handler` | Data loading, influence functions, mixture optimization | Imported as library |
| `inference` | Model serving with sglang-bitnet | `demo/serve_sglang.py` |
| `eval` | GLUE, CNN/DailyMail benchmarks | `scripts/evaluate.py` |
| `deployer` | SkyPilot cloud deployment | `wf` CLI |
| `mobile` | Android inference with BitNet.cpp | Android app |

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

`data_handler` is a shared library imported by:
- `training` - for data loading and influence-based optimization

`architecture` provides BitNet components to:
- `training` - BitLinear, BitLinearLRC layers, SubLN, model conversion

Use workspace dependencies:
```toml
# In pyproject.toml
[tool.uv.sources]
data-handler = { workspace = true }
bitnet-arch = { workspace = true }
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
