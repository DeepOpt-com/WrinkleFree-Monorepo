# WrinkleFree Monorepo Architecture

## Package Overview

```
                    ┌─────────────────┐
                    │   data_handler   │  (Shared Library)
                    │  - Data loading  │
                    │  - Influence     │
                    │  - Mixture opt   │
                    └────────┬────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │    training     │  (Application)
                   │  1.58-bit QAT   │
                   │  + Distillation │  ← BitDistill, TCS, LRC objectives
                   │  (Stages 1-3)   │
                   └────────┬────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  architecture   │  (Shared Library)
                  │  - BitLinear    │
                  │  - BitLinearLRC │
                  │  - SubLN        │
                  └─────────────────┘
                            │
                            │ produces checkpoints
                            ▼
                  ┌─────────────────┐           ┌─────────────────┐
                  │    inference    │───────────│      eval       │
                  │  sglang-bitnet  │           │  GLUE/CNN eval  │
                  └─────────────────┘           └─────────────────┘

    ┌─────────────────┐          ┌─────────────────┐
    │    deployer     │          │     mobile      │
    │    SkyPilot     │          │ Android BitNet  │
    └─────────────────┘          └─────────────────┘
```

## Package Types

| Package | Type | Description |
|---------|------|-------------|
| `data_handler` | **Library** | Shared data loading, influence functions |
| `architecture` | **Library** | BitNet layers (BitLinear, BitLinearLRC, SubLN) & model conversion |
| `training` | Application | 1.58-bit training pipeline (Stages 1-3) + distillation objectives |
| `inference` | Application | Model serving application |
| `eval` | Application | Evaluation scripts |
| `deployer` | Application | CLI tool for cloud deployment |
| `mobile` | Application | Android inference with BitNet.cpp |

> **Note**: Legacy packages (`distillation`, `converter`, `cheapertraining`) are archived in `packages/_legacy/`.

## Data Flow

### Training Pipeline

The training package supports two approaches:

#### PyTorch Lightning (Recommended)
The new Lightning-based trainer provides a cleaner, more maintainable training loop:
- Auto batch size scaling via `BatchSizeFinder`
- Built-in DDP/FSDP support
- All objectives work unchanged (DLM, LRC, distillation)

```bash
uv run python scripts/train_lightning.py model=smollm2_135m training=base
```

#### Legacy Stages (Still Supported)
1. **Stage 1**: SubLN insertion (training package)
2. **Stage 1.9**: Layer-wise distillation (training package)
3. **Stage 2**: Continue pre-training with QAT (training package)
4. **Stage 3**: Knowledge distillation via objectives (training package)
5. **LRC**: Post-quantization low-rank correction (training package)
6. **Export**: Convert to GGUF (see root CLAUDE.md for workflow)
7. **Serve**: Deploy for inference (inference package)
8. **Evaluate**: Run benchmarks (eval package)

### Distillation Objectives (via Training Package)

The training package supports multiple distillation modes via the objectives system:
- **BitDistill** (`training=bitdistill_full`): Logits + attention relation distillation
- **TCS** (`tcs_distill` objective): Target Concrete Score for DLM students
- **LRC** (`training=lrc_calibration`): Low-Rank Correction for post-quantization recovery
- **Logits-only** (`logits_distill` objective): KL divergence without attention

### Deployment Flow

```
Local Development
       │
       ▼
┌──────────────┐
│   deployer   │ ── wf train ───────► SkyPilot cloud training
│              │ ── wf serve ───────► Cloud GPU serving
│              │ ── wf eval  ───────► Batch evaluation
└──────────────┘

Note: Distillation is now done via training objectives:
  training=bitdistill_full  → BitDistill (logits + attention)
  training=lrc_calibration  → LRC post-quant recovery
```

## External Dependencies

### Git Submodules

| Submodule | Location | Purpose |
|-----------|----------|---------|
| BitNet | `extern/BitNet/` | Microsoft BitNet.cpp inference engine |
| sglang-bitnet | `packages/inference/extern/sglang-bitnet/` | SGLang fork for BitNet |

### Key Libraries

- **torch**: Core training framework
- **pytorch-lightning**: Training loop, auto batch size, distributed
- **transformers**: Model architectures (Llama, Qwen)
- **hydra-core**: Configuration management
- **datasets**: HuggingFace data loading
- **wandb**: Experiment tracking
- **vLLM**: Remote teacher backend (optional, for distillation)

## Configuration

All packages use Hydra for configuration:

```
packages/{name}/configs/
├── model/          # Model architecture configs
├── training/       # Training hyperparameters
├── data/           # Dataset configs
├── distillation/   # Distillation settings (distillation package)
└── distributed/    # FSDP/DDP settings
```

Shared data configs are in the `data_handler` package.

## Build & Test

```bash
# Install all packages
uv sync --all-packages

# Run all tests
uv run pytest

# Run package-specific tests
uv run --package wrinklefree pytest packages/training/tests/
uv run --package data-handler pytest packages/data_handler/tests/
uv run --package bitnet-arch pytest packages/architecture/tests/

# Type checking
uv run mypy packages/training/src/
uv run ruff check packages/
```
