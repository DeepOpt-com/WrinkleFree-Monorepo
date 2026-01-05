# WrinkleFree Deployer

> Part of [WrinkleFree Monorepo](https://github.com/DeepOpt-com/WrinkleFree-Monorepo) - Cloud deployment orchestrator.

**Simple launcher for training 1.58-bit LLMs.**

```bash
# Clone the monorepo
git clone --recurse-submodules git@github.com:DeepOpt-com/WrinkleFree-Monorepo.git
cd WrinkleFree-Monorepo

# Install all packages
uv sync --all-packages

# Set credentials
cp .env.example .env  # Add WANDB_API_KEY

# Setup Nebius ($1.99/hr H100)
pip install "skypilot[nebius]"
sky check nebius

# Train
uv run wf train -m qwen3_4b -s 1.9
```

That's it. See [Getting Started](docs/getting-started.md) for more.

## What This Does

This is a **launcher**, not a training framework. The training code lives in [WrinkleFree-1.58Quant](../training/).

We just:
1. Launch training jobs on SkyPilot (Nebius, RunPod, GCP, AWS)
2. Pass through any config overrides to Hydra
3. Provide `wf logs` / `wf runs` / `wf cancel` for monitoring

## Usage

### CLI

```bash
# Stage 1: Model conversion
uv run wf train -m qwen3_4b -s 1

# Stage 1.9: Layer-wise distillation
uv run wf train -m qwen3_4b -s 1.9

# Stage 2: Continue pretraining (main training phase)
uv run wf train -m qwen3_4b -s 2

# Multi-GPU: use --scale for different GPU configs
uv run wf train -m qwen3_4b -s 2 --scale xlarge  # 8x H100

# With Hydra overrides (anything after the flags)
uv run wf train -m qwen3_4b -s 1.9 training.lr=1e-4 training.batch_size=8

# Wait for completion
uv run wf train -m qwen3_4b -s 1.9 --no-detach

# Monitor
uv run wf logs wrinklefree-train      # View logs
uv run wf logs wrinklefree-train -f   # Follow logs
uv run wf runs                        # List all jobs
uv run wf cancel wrinklefree-train    # Cancel job

# Smoke test
uv run wf smoke
```

### Scale Profiles

| Scale | GPUs | Use case |
|-------|------|----------|
| `dev` | 1x H100 | Testing |
| `small` | 1x H100 | Default |
| `xlarge` | 8x H100 | Production |

Note: Nebius only has 1 or 8 GPU configs.

### Python

```python
from wf_deploy import train

run_id = train("qwen3_4b", stage=1)      # Stage 1: conversion
run_id = train("qwen3_4b", stage=1.9)    # Stage 1.9: distillation
run_id = train("qwen3_4b", stage=1.9, scale="xlarge")  # 8x H100
```

## Models & Stages

| Model | Config | For |
|-------|--------|-----|
| `smollm2_135m` | SmolLM2-135M | Testing (small, fast) |
| `qwen3_4b` | Qwen3-4B | Production |

| Stage | Purpose | Est. Cost (Nebius) |
|-------|---------|-------------------|
| 1 | Model conversion | ~$0 |
| 1.9 | Layer-wise distillation | $5-25 |
| 2 | Continue pretraining | $30-120 |
| 3 | Fine-tuning | $5-25 |

## Configuration

Configs live in `packages/training/configs/`. Override any value:

```bash
uv run wf train -m qwen3_4b -s 1.9 training.lr=1e-4
uv run wf train -m qwen3_4b -s 1.9 training.max_steps=100
uv run wf train -m qwen3_4b -s 1.9 training.logging.wandb.enabled=false
```

## Docs

- [Getting Started](docs/getting-started.md) - Get running in 5 minutes
- [Training Guide](docs/training.md) - Full training documentation
- [Serving Guide](docs/serving.md) - Deploy for inference
