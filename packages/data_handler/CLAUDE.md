# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

data_handler is a shared library in the WrinkleFree monorepo providing:
- **Data Loading**: Streaming datasets, sequence packing, data mixing
- **Mixture Optimization**: Dynamic dataset weight optimization

> **Note**: DataInf-based influence functions are now in `_legacy/`. Use `training.meta_optimization.odm` instead (O(1) complexity via EXP3 bandit). See [arxiv:2312.02406](https://arxiv.org/abs/2312.02406).

## Monorepo Integration

This is a **shared library** imported by the training package:

```
data_handler (this package)
    │
    └──► packages/training (wrinklefree)
            Uses: data_handler.data, data_handler.influence
```

> **Note**: The legacy `distillation` package (now in `_legacy/`) also used this library.
> Distillation is now integrated into the training package via objectives.

**Workspace dependency** (in consumer's pyproject.toml):
```toml
[project]
dependencies = ["data-handler"]

[tool.uv.sources]
data-handler = { workspace = true }
```

**Running from monorepo root**:
```bash
uv run --package data-handler python -c "import data_handler; print('ok')"
```

## Quick Start

```bash
# From monorepo root
uv sync --all-packages

# Run tests
uv run --package data-handler pytest packages/data_handler/tests/

# Import in Python
from data_handler.data import get_loader
from data_handler.influence import InfluenceAwareOptimizer
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `data_handler.data` | Data loading, streaming, packing |
| `data_handler.data.mixing` | MixedDataset with dynamic weights |
| `data_handler._legacy.influence_datainf` | DEPRECATED: DataInf influence (use ODM) |

## Architecture

```
src/data_handler/
├── data/                    # Data loading & processing
│   ├── tokenization.py         # TokenizerWrapper
│   ├── mixing.py               # MixedDataset, PackedDataset
│   └── datasets/
│       ├── pretrain.py         # PretrainDataset with packing
│       └── sft.py              # SFTDataset with chat templates
│
├── _legacy/                 # Deprecated code
│   └── influence_datainf/      # DataInf influence (replaced by ODM)
│       ├── tracker.py          # InfluenceTracker callback
│       └── mixture_calculator.py
│
└── distributed/             # Distributed training
    ├── fsdp2.py                # FSDP2 wrapping & checkpointing
    └── parallelism.py          # Tensor/pipeline parallelism
```

## Configuration

Configs are in `configs/` using Hydra:
- `data/` - Data mixture configs (mixed_pretrain, fineweb, downstream)
- `model/` - Model configs
- `distributed/` - FSDP/DDP settings

## Data Configs (Used by training & distillation)

| Config | Description | Usage |
|--------|-------------|-------|
| `mixed_pretrain` | Multi-source with influence | Stage 2 pre-training |
| `fineweb` | Single-source FineWeb-Edu | Simple pre-training |
| `downstream` | SFT/finetuning tasks | Distillation |

**Selecting data config from training package**:
```bash
uv run --package wrinklefree python scripts/train.py data.config_name=mixed_pretrain
```

## Development

```bash
# Run tests
uv run --package data-handler pytest

# Type check
uv run mypy packages/data_handler/src/

# Lint
uv run ruff check packages/data_handler/
```

## Notes

- Changes to this package affect training and distillation
- Run tests in all dependent packages after modifications
- Data configs are shared - update carefully
