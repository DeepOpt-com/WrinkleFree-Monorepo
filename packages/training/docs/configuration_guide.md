# Configuration Guide

This guide helps you navigate the configuration system and choose the right settings for your training needs. WrinkleFree uses [Hydra](https://hydra.cc/) for configuration management, allowing you to compose configurations from modular YAML files.

## Directory Structure

Configurations are located in `configs/`:

- **`model/`**: Defines model architectures (e.g., `smollm2_135m`, `qwen3_4b`).
- **`training/`**: Defines the training stage (e.g., `stage1_subln`, `stage3_distill`).
- **`distributed/`**: Hardware and distribution strategies (e.g., `single_gpu`, `fsdp_multi`).
- **`data/`**: Dataset definitions (e.g., `fineweb`, `downstream`).
- **`distillation/`**: Loss coefficients for Stage 3 (e.g., `classification`).

## Step 1: Choose Your Model (`model=...`)

Select a model based on your available hardware and goals.

| Model | Config Name | Params | VRAM (Training) | Use Case |
|-------|-------------|--------|-----------------|----------|
| **SmolLM2-135M** | `smollm2_135m` | 135M | < 12GB (Single GPU) | **Start Here**. Testing, debugging, learning the codebase. |
| **Qwen3-4B** | `qwen3_4b` | 4B | ~24GB (Stage 3) | **Recommended**. Good balance of performance and efficiency. |
| **LLaMA-3B** | `llama_3b` | 3B | ~20GB | Development alternative to Qwen. |
| **LLaMA-7B** | `llama_7b` | 7B | > 40GB (Multi-GPU) | Production-grade training. Requires FSDP. |

**Default**: `llama_7b` (in `config.yaml`), but **we recommend overriding to `smollm2_135m` for your first run**.

## Step 2: Choose Your Training Stage (`training=...`)

The training process is sequential. You must complete earlier stages to generate checkpoints for later ones.

1.  **Stage 1: SubLN Insertion** (`stage1_subln`)
    *   **What it does**: Modifies a pretrained model to insert SubLN layers and replace Linear layers with BitLinear.
    *   **Input**: HuggingFace pretrained model name.
    *   **Output**: An initialized (untrained) BitNet model.
    *   **Time**: Seconds/Minutes.

2.  **Stage 1.9: Layer-wise Distillation** (`stage1_9_layerwise`)
    *   **What it does**: Aligns hidden states of the BitNet model with the original float model.
    *   **Input**: Stage 1 checkpoint.
    *   **Time**: ~1-2 hours.

3.  **Stage 2: Continue Pretraining** (`stage2_pretrain`)
    *   **What it does**: Trains the model on a large corpus with quantization warmup.
    *   **Input**: Stage 1 or 1.9 checkpoint.
    *   **Time**: Hours/Days.

4.  **Stage 3: Distillation** (`stage3_distill` / `stage3_distill_smollm2`)
    *   **What it does**: Fine-tunes the model using knowledge distillation from a teacher.
    *   **Input**: Stage 2 checkpoint.
    *   **Time**: Hours.

## Step 3: Choose Hardware Strategy (`distributed=...`)

| Config Name | Description |
|-------------|-------------|
| `single_gpu` | Standard PyTorch DDP (or no DDP if 1 device). Use for SmolLM2 or debugging. |
| `fsdp_multi` | Fully Sharded Data Parallel for multi-GPU setups. Required for 3B/4B/7B models. |
| `fsdp_large` | FSDP optimized for large scale (8+ GPUs, cross-node). |

## Step 4: Choose Data (`data=...`)

| Config Name | Description |
|-------------|-------------|
| `fineweb` | Large-scale pretraining dataset (HuggingFaceFW/fineweb). Used in Stage 1.9 and Stage 2. |
| `downstream` | Task-specific datasets (e.g., Glue, SuperGlue) for Stage 3 distillation. |

## Common Recipes

### 1. The "Hello World" (Run this first)
Verify everything works with a tiny model on a single GPU.

```bash
# Insert SubLN
uv run python scripts/train.py model=smollm2_135m training=stage1_subln distributed=single_gpu

# Run a few steps of Stage 2 training
uv run python scripts/train.py \
    model=smollm2_135m \
    training=stage2_pretrain \
    distributed=single_gpu \
    training.max_steps=10
```

### 2. The "Efficient Standard" (Qwen3-4B)
Train a capable model using FSDP.

```bash
# Stage 1
uv run python scripts/train.py model=qwen3_4b training=stage1_subln distributed=single_gpu

# Stage 2 (Multi-GPU)
uv run python scripts/train.py \
    model=qwen3_4b \
    training=stage2_pretrain \
    distributed=fsdp_multi
```

## How to Override Defaults

You can override any parameter from the command line using dot notation.

**Example**: Change batch size and learning rate:
```bash
uv run python scripts/train.py \
    model=smollm2_135m \
    training.batch_size=32 \
    training.lr=1e-4
```

**Example**: Change specific model parameters:
```bash
uv run python scripts/train.py \
    model=smollm2_135m \
    model.quantization.weight_bits=1.58
```

## FAQ

**Q: I'm getting Out of Memory (OOM) errors.**
A:
1.  Switch to a smaller model (`model=smollm2_135m`).
2.  Reduce batch size (`training.batch_size=1` or `2`).
3.  Enable gradient accumulation (`training.gradient_accumulation_steps=16`).
4.  Ensure `distributed=fsdp_multi` is used for larger models.

**Q: Where are my checkpoints?**
A: Check `outputs/bitdistill_<model>_<stage>/`.

**Q: Which default should I change in `config.yaml`?**
A: Avoid changing `configs/config.yaml` directly if possible. Instead, create a `local.yaml` or just use command line overrides to keep your git status clean.
