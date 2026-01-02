# WrinkleFree Training

> Part of [WrinkleFree Monorepo](https://github.com/DeepOpt-com/WrinkleFree-Monorepo) - 1.58-bit quantization training.

BitNet 1.58-bit LLM training framework using PyTorch Lightning.

## Overview

WrinkleFree trains language models with **1.58-bit (ternary) weights** {-1, 0, 1}, achieving significant memory savings and faster inference while maintaining performance comparable to full-precision models.

### Key Features

- **PyTorch Lightning Trainer**: Clean, maintainable training with auto batch size scaling
- **ObjectiveManager**: Multi-task training combining CE, DLM, distillation objectives
- **Curriculum Learning**: Phase-based weight transitions for training objectives
- **Influence-Based Data Remixing**: Dynamic dataset weight optimization (MobileLLM-R1 methodology)
- **FSDP Support**: Seamless distributed training with MuonClip optimizer

## Quick Start

```bash
# Install dependencies (from monorepo root)
uv sync --all-packages

# Run unified training (recommended)
uv run python scripts/train_lightning.py model=smollm2_135m training=unified

# With auto batch size scaling
uv run python scripts/train_lightning.py model=smollm2_135m training=unified \
    training.auto_batch_size=true
```

## Training Configurations

| Config | Purpose | Use Case |
|--------|---------|----------|
| `unified` | Combined STE + DLM training | **Recommended** - production training |
| `bitdistill_full` | Knowledge distillation | Teacher-student distillation |
| `lrc_calibration` | Low-rank correction | Post-quantization recovery |
| `stage2_pretrain` | Continue pretraining | Legacy stage-based training |

### Unified Training

The recommended approach combines STE quantization with DLM (Diffusion Language Model) objectives:

```bash
uv run python scripts/train_lightning.py model=smollm2_135m training=unified

# With influence-based data remixing
uv run python scripts/train_lightning.py model=smollm2_135m training=unified \
    data.config_name=mixed_pretrain \
    training.influence.enabled=true
```

## Available Models

| Model | Config | Params | Use Case |
|-------|--------|--------|----------|
| SmolLM2-135M | `model=smollm2_135m` | 135M | Testing, debugging |
| Qwen3-4B | `model=qwen3_4b` | 4B | Production |
| Qwen2-0.5B | `model=qwen2_0.5b` | 0.5B | Development |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PyTorch Lightning                        │
│  train_lightning.py → WrinkleFreeLightningModule           │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │           ObjectiveManager (multi-task)                 ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              ││
│  │  │ CE Loss  │  │   DLM    │  │   LRC    │  ...         ││
│  │  └──────────┘  └──────────┘  └──────────┘              ││
│  └─────────────────────────────────────────────────────────┘│
│         ↓                                                   │
│  Callbacks: BatchSizeFinder, GCS, ZClip, TokenCount        │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Directory | Purpose |
|-----------|---------|
| `lightning/` | PyTorch Lightning module, datamodule, callbacks |
| `objectives/` | ObjectiveManager + individual objectives (CE, DLM, distillation, LRC) |
| `training/` | Auto-setup, FSDP wrapper, legacy trainer |
| `models/` | BitLinear, SubLN, attention, FFN layers |
| `quantization/` | STE, weight/activation quantization |
| `serving/` | GGUF conversion, BitNet.cpp integration |
| `_experimental/` | MoE, TensorParallel, FP8 (not production-ready) |

## Monorepo Dependencies

This package depends on:
- **data_handler**: Data loading and influence-based optimization
- **bitnet_arch**: BitNet layers (BitLinear, BitLinearLRC, SubLN) and model conversion

## Configuration

All configs use [Hydra](https://hydra.cc/) and are in `configs/`:

```bash
# Common overrides
training.max_steps=100              # Limit steps for testing
training.auto_batch_size=true       # Auto-find max batch size
training.optimizer.type=adamw       # Use AdamW instead of MuonClip
gcs.enabled=true gcs.bucket=wrinklefree-checkpoints
```

## Documentation

- **[CLAUDE.md](./CLAUDE.md)**: Comprehensive guide with all training options
- **[docs/architecture.md](./docs/architecture.md)**: System architecture details
- **[docs/configuration_guide.md](./docs/configuration_guide.md)**: Configuration reference
- **[docs/experiments.md](./docs/experiments.md)**: Reproduction steps

## References

- [BitDistill: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation](https://arxiv.org/abs/2510.13998)
- [Microsoft BitNet](https://github.com/microsoft/BitNet)
