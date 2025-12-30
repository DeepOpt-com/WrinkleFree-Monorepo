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
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              │              ▼
    ┌─────────────────┐      │    ┌─────────────────┐
    │    training     │      │    │  distillation   │
    │  1.58-bit QAT   │      │    │  Teacher-student│
    │  (BitDistill)   │      │    │  (Stage 3+)     │
    └────────┬────────┘      │    └────────┬────────┘
             │               │             │
             │               │             │
    ┌────────┴───────────────┘             │
    │                                      │
    ▼                                      │
┌─────────────────┐                        │
│  architecture   │  (Shared Library)      │
│  - BitLinear    │                        │
│  - SubLN        │                        │
│  - Conversion   │                        │
└─────────────────┘                        │
                                           │
             ┌─────────────────────────────┘
             │ produces checkpoints
             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │   converter     │───────────│    inference    │
    │  DLM format     │           │  sglang-bitnet  │
    └─────────────────┘           └────────┬────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │      eval       │
                                  │  GLUE/CNN eval  │
                                  └─────────────────┘

    ┌─────────────────┐
    │    deployer     │  (Orchestrates cloud training/serving/distillation)
    │  SkyPilot/Modal │
    └─────────────────┘
```

## Package Types

| Package | Type | Description |
|---------|------|-------------|
| `data_handler` | **Library** | Shared data loading, influence functions |
| `architecture` | **Library** | BitNet layers (BitLinear, SubLN) & model conversion |
| `training` | Application | 1.58-bit training pipeline (Stages 1, 1.9, 2) |
| `distillation` | Application | Knowledge distillation (Stage 3+, BitDistill, TCS) |
| `inference` | Application | Model serving application |
| `eval` | Application | Evaluation scripts |
| `deployer` | Application | CLI tool for cloud deployment |
| `converter` | Application | Model format conversion |

## Data Flow

### Training Pipeline

1. **Stage 1**: SubLN insertion (training package)
2. **Stage 1.9**: Layer-wise distillation (training package)
3. **Stage 2**: Continue pre-training with QAT (training package)
4. **Stage 3**: Knowledge distillation (**distillation package**)
5. **Export**: Convert to GGUF/DLM (converter package)
6. **Serve**: Deploy for inference (inference package)
7. **Evaluate**: Run benchmarks (eval package)

### Distillation Options

The distillation package supports multiple modes:
- **BitDistill**: Logits + attention relation distillation
- **TCS**: Target Concrete Score for DLM (diffusion) students
- **Logits-only**: KL divergence without attention

### Deployment Flow

```
Local Development
       │
       ▼
┌──────────────┐
│   deployer   │ ── wf train ───────► Modal/SkyPilot
│              │ ── wf distill ─────► Distillation jobs
│              │ ── wf tcs-distill ─► TCS distillation
│              │ ── wf serve ───────► Cloud GPU
│              │ ── wf eval  ───────► Batch eval
└──────────────┘
```

## External Dependencies

### Git Submodules

| Submodule | Location | Purpose |
|-----------|----------|---------|
| BitNet | `extern/BitNet/` | Microsoft BitNet.cpp inference engine |
| sglang-bitnet | `packages/inference/extern/sglang-bitnet/` | SGLang fork for BitNet |

### Key Libraries

- **torch**: Core training framework
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
uv run --package wrinklefree-distillation pytest packages/distillation/tests/
uv run --package bitnet-arch pytest packages/architecture/tests/

# Type checking
uv run mypy packages/training/src/
uv run ruff check packages/
```
