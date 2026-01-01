# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

CheaperTraining is a shared library in the WrinkleFree monorepo providing:
- **Data Loading**: Streaming datasets, sequence packing, data mixing
- **Influence Functions**: Data influence computation for optimizing training data
- **Mixture Optimization**: Dynamic dataset weight optimization

## Monorepo Integration

This is a **shared library** imported by other packages:

```
cheapertraining (this package)
    │
    ├──► packages/training (wrinklefree)
    │       Uses: cheapertraining.data, cheapertraining.influence
    │
    └──► packages/fairy2
            Uses: cheapertraining.data
```

**Workspace dependency** (in consumer's pyproject.toml):
```toml
[project]
dependencies = ["cheapertraining"]

[tool.uv.sources]
cheapertraining = { workspace = true }
```

**Running from monorepo root**:
```bash
uv run --package cheapertraining python -c "import cheapertraining; print('ok')"
```

## Quick Start

```bash
# From monorepo root
uv sync --all-packages

# Run tests
uv run --package cheapertraining pytest packages/cheapertraining/tests/

# Import in Python
from cheapertraining.data import get_loader
from cheapertraining.influence import InfluenceAwareOptimizer
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `cheapertraining.data` | Data loading, streaming, packing |
| `cheapertraining.influence` | Influence function computation |
| `cheapertraining.mixing` | Dataset mixing and weight optimization |

## Architecture

```
src/cheapertraining/
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

## Data Configs (Used by training & fairy2)

| Config | Description | Usage |
|--------|-------------|-------|
| `mixed_pretrain` | Multi-source with influence | Stage 2 pre-training |
| `fineweb` | Single-source FineWeb-Edu | Simple pre-training |
| `downstream` | SFT/finetuning tasks | Stage 3 distillation |

**Selecting data config from training package**:
```bash
uv run --package wrinklefree python scripts/train.py data.config_name=mixed_pretrain
```

## Development

```bash
# Run tests
uv run --package cheapertraining pytest

# Type check
uv run mypy packages/cheapertraining/src/

# Lint
uv run ruff check packages/cheapertraining/
```

## Notes

- Changes to this package affect training and fairy2
- Run tests in all dependent packages after modifications
- Data configs are shared - update carefully
