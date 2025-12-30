# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

data_handler is a shared library in the WrinkleFree monorepo providing:
- **Data Loading**: Streaming datasets, sequence packing, data mixing
- **Influence Functions**: Data influence computation for optimizing training data
- **Mixture Optimization**: Dynamic dataset weight optimization

## Monorepo Integration

This is a **shared library** imported by other packages:

```
data_handler (this package)
    │
    ├──► packages/training (wrinklefree)
    │       Uses: data_handler.data, data_handler.influence
    │
    └──► packages/distillation
            Uses: data_handler.data, data_handler.influence
```

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
| `data_handler.influence` | Influence function computation |
| `data_handler.mixing` | Dataset mixing and weight optimization |

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
├── influence/               # Influence-based optimization
│   ├── calculator.py           # DataInfCalculator
│   └── optimizer.py            # InfluenceAwareOptimizer
│
├── distillation/            # Knowledge distillation
│   ├── teacher.py              # TeacherWrapper, CachedTeacher
│   └── losses.py               # KL divergence losses
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
