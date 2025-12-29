# WrinkleFree Monorepo Architecture

## Package Overview

```
                    ┌─────────────────┐
                    │  cheapertraining │  (Shared Library)
                    │  - Data loading  │
                    │  - Influence     │
                    │  - Mixture opt   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │    training     │           │     fairy2      │
    │  1.58-bit QAT   │           │  Complex quant  │
    │  (BitDistill)   │           │                 │
    └────────┬────────┘           └─────────────────┘
             │
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
    │    deployer     │  (Orchestrates cloud training/serving)
    │  SkyPilot/Modal │
    └─────────────────┘
```

## Package Types

| Package | Type | Description |
|---------|------|-------------|
| `cheapertraining` | **Library** | Shared code imported by other packages |
| `training` | Application | Executable training pipeline |
| `fairy2` | Application | Executable training for complex-valued models |
| `inference` | Application | Model serving application |
| `eval` | Application | Evaluation scripts |
| `deployer` | Application | CLI tool for cloud deployment |
| `converter` | Application | Model format conversion |

## Data Flow

### Training Pipeline

1. **Stage 1**: SubLN insertion (training package)
2. **Stage 1.9**: Layer-wise distillation (training package)
3. **Stage 2**: Continue pre-training with QAT (training package)
4. **Stage 3**: Distillation fine-tuning (training package)
5. **Export**: Convert to GGUF/DLM (converter package)
6. **Serve**: Deploy for inference (inference package)
7. **Evaluate**: Run benchmarks (eval package)

### Deployment Flow

```
Local Development
       │
       ▼
┌──────────────┐
│   deployer   │ ── wf train ──► Modal/SkyPilot
│              │ ── wf serve ──► Cloud GPU
│              │ ── wf eval  ──► Batch eval
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

## Configuration

All packages use Hydra for configuration:

```
packages/{name}/configs/
├── model/          # Model architecture configs
├── training/       # Training hyperparameters
├── data/           # Dataset configs
└── distributed/    # FSDP/DDP settings
```

Root-level configs can be shared via the `cheapertraining` package.

## Build & Test

```bash
# Install all packages
uv sync --all-packages

# Run all tests
uv run pytest

# Run package-specific tests
uv run --package wrinklefree pytest packages/training/tests/
uv run --package cheapertraining pytest packages/cheapertraining/tests/

# Type checking
uv run mypy packages/training/src/
uv run ruff check packages/
```
