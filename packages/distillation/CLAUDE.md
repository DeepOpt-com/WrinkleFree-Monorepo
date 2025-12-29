# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree-Distillation implements knowledge distillation for quantized LLMs:
- **Algorithm**: BitDistill-style distillation (logits + attention)
- **Training**: Teacher-student distillation with temperature scaling
- **Config**: Hydra
- **Package management**: uv
- **License**: Apache 2.0 (commercially friendly)

## Monorepo Integration

This package is part of the WrinkleFree monorepo and depends on:
- **cheapertraining**: Shared data loading utilities, influence-based dataset rebalancing

**Related packages**:
| Package | Relationship |
|---------|--------------|
| `cheapertraining` | Data loading, influence functions (shared library) |
| `training` | Produces quantized models that can be distilled |
| `fairy2` | Produces Fairy2 models that can be distilled |
| `deployer` | Cloud deployment |

**Running from monorepo root**:
```bash
uv run --package wrinklefree-distillation python packages/distillation/scripts/distill.py \
  student.checkpoint_path=outputs/stage2/checkpoint.pt
```

## Key Concepts

### Distillation Loss (BitDistill)

Combined loss with three components:
```
L = L_CE + lambda_logits * L_LD + gamma_attention * L_AD

Where:
- L_CE: Cross-entropy on ground truth labels
- L_LD: KL(P_teacher || P_student) * T^2 at temperature T
- L_AD: Attention relation distillation (single layer)
```

### Teacher Backends

1. **LocalTeacher**: In-process HuggingFace model (frozen, eager attention)
2. **VLLMTeacher**: Remote vLLM server (logits-only, no attention)
3. **CachedTeacher**: Pre-computed teacher outputs (future)

## Quick Start

```bash
# Install dependencies
uv sync

# Distill a BitNet model (default: use original teacher)
uv run python scripts/distill.py \
  student.checkpoint_path=outputs/stage2/checkpoint.pt

# Distill with different teacher
uv run python scripts/distill.py \
  student.checkpoint_path=outputs/stage2/checkpoint.pt \
  teacher.model_name=meta-llama/Llama-3.2-3B

# Logits-only (no attention distillation)
uv run python scripts/distill.py \
  student.checkpoint_path=outputs/stage2/checkpoint.pt \
  distillation=logits_only

# Run tests
uv run pytest
```

## Configuration

All configs in `configs/` using Hydra:
- `distillation/` - Distillation configs (bitdistill, logits_only, classification)
- `training/` - Training hyperparameters

### Key Config Options

```yaml
# Student settings
student:
  checkpoint_path: ???  # Required

# Teacher settings
teacher:
  model_name: null      # null = infer from student's original model
  use_vllm: false       # Use vLLM server
  vllm_url: "http://localhost:8000"

# Loss coefficients
distillation:
  lambda_logits: 10.0   # Logits distillation weight
  gamma_attention: 1e-5 # Attention distillation weight (0 = disabled)
  temperature: 5.0      # KL temperature

# Data (from cheapertraining)
data:
  config_name: mixed_pretrain

# Influence-based rebalancing
influence:
  enabled: true
  update_interval: 1000
```

## Architecture

### Core Modules
- `src/distillation/losses/` - Loss functions (logits, attention, combined)
- `src/distillation/teachers/` - Teacher backends (local, vLLM)
- `src/distillation/training/` - Trainer and config

### Entry Point
- `scripts/distill.py` - Main Hydra entry point

## Data (via CheaperTraining)

Uses cheapertraining's data infrastructure:
- `create_dataloader()` - Multi-source mixed datasets
- `MixedDataset.update_weights_from_influence()` - Dynamic rebalancing
- `InfluenceDistillation` - Compute optimal mixture weights

## Testing

```bash
# All tests
uv run pytest

# Smoke tests only
uv run pytest -m smoke

# With coverage
uv run pytest --cov=distillation
```

## References

- Paper: [BitDistill](https://arxiv.org/abs/2510.13998)
- Related: [BitNet](https://arxiv.org/abs/2310.11453), [Fairy2i](https://arxiv.org/abs/2512.02901)
