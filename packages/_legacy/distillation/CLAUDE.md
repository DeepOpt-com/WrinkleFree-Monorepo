# CLAUDE.md (DEPRECATED)

> **WARNING: This package has been archived to `packages/_legacy/`.**
>
> Distillation functionality is now integrated into `packages/training/src/wrinklefree/objectives/`.
>
> **For distillation, use the training package:**
> ```bash
> # BitDistill (logits + attention distillation)
> uv run python scripts/train.py training=bitdistill_full
>
> # LRC Calibration (post-quantization recovery)
> uv run python scripts/train.py training=lrc_calibration
> ```
>
> See `packages/training/CLAUDE.md` for current documentation.

---

*Original documentation preserved below for reference only*

---

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree-Distillation implements knowledge distillation for quantized LLMs:
- **Algorithm**: BitDistill-style distillation (logits + attention)
- **Training**: Teacher-student distillation with temperature scaling
- **Config**: Hydra
- **Package management**: uv
- **License**: Apache 2.0 (commercially friendly)

## MUST-DO Rules

1. **Always use MuonClip + AdamW**: Set `optimizer.type=muon` (NOT adamw). If it fails, fix it.
2. **Always enable QK-clipping**: Set `enable_clipping=True` in MuonConfig. Prevents loss spikes and training instability.
3. **Always use seq_len=2048**: Never reduce sequence length. If OOM, use more GPUs or gradient checkpointing.
4. **Always use max_seq_length=2048 in data config**: This is the standard for LLM training.
5. **Always pass WANDB_API_KEY**: Add `--env WANDB_API_KEY=<key>` when launching jobs. Without it, you CANNOT monitor training losses and detect instabilities. A VERY LOUD warning will print if missing.

## Monorepo Integration

This package is part of the WrinkleFree monorepo and depends on:
- **data_handler**: Shared data loading utilities, influence-based dataset rebalancing

**Related packages**:
| Package | Relationship |
|---------|--------------|
| `data_handler` | Data loading, influence functions (shared library) |
| `training` | Produces quantized models that can be distilled |
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

### TCS Distillation Loss (for DLM Students)

Target Concrete Score (TCS) distillation for DLM (Diffusion Language Model) students:
```
L = L_CE + lambda_tcs * L_TCS + gamma_attn * L_BlockAttn

Where:
- L_CE: Cross-entropy on ground truth labels (NO SHIFT for DLM!)
- L_TCS: KL(softmax(teacher_topk/T) || softmax(student[topk_indices]/T)) * T^2
- L_BlockAttn: Block-wise attention relation distillation (within bd_size blocks)
```

**Key differences from BitDistill**:
1. **No logit shifting** - DLM predicts masked tokens, not next tokens
2. **Top-K estimation** - Sparse distribution matching for efficiency
3. **Block-wise attention** - Only matches attention within blocks where both AR teacher and DLM student use bidirectional attention

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

# TCS distillation for DLM students (block-wise attention enabled)
uv run python scripts/distill.py \
  distillation=tcs \
  student.checkpoint_path=gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/ \
  student.type=dlm

# TCS with explicit teacher
uv run python scripts/distill.py \
  distillation=tcs \
  student.checkpoint_path=gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/ \
  student.type=dlm \
  teacher.model_name=1bitLLM/bitnet_b1_58-2B

# Run tests
uv run pytest
```

## Configuration

All configs in `configs/` using Hydra:
- `distillation/` - Distillation configs (bitdistill, logits_only, tcs, classification)
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

# Data (from data_handler)
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

## Data (via data_handler)

Uses data_handler's data infrastructure:
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
- TCS: [Apple TCSM (ICML 2025)](https://machinelearning.apple.com/research/target-concrete)
- DLM: [DDLM](https://openreview.net/forum?id=xfw92pDy2u), [Fast-dLLM v2](https://arxiv.org/abs/2509.26328)
- Related: [BitNet](https://arxiv.org/abs/2310.11453), [Fairy2i](https://arxiv.org/abs/2512.02901)
