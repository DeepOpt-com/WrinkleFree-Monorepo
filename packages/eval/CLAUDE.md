# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree-Eval is the evaluation framework for WrinkleFree BitNet models:
- **Framework**: lm-eval harness
- **Benchmarks**: BitDistill paper benchmarks (GLUE subset)
- **Models**: HuggingFace models + BitNet.cpp optimized inference
- **Config**: Hydra
- **Package management**: uv

## Monorepo Integration

This package is part of the WrinkleFree monorepo.

**Related packages**:
| Package | Relationship |
|---------|--------------|
| `training` | Produces models to evaluate |
| `inference` | Provides optimized inference backend |
| `deployer` | Cloud deployment for batch evaluation |

**Running from monorepo root**:
```bash
uv run --package wf_eval python packages/eval/scripts/run_eval.py model_path=path/to/model benchmark=bitdistill
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run smoke test (sst2 only, 10 samples)
uv run python scripts/run_eval.py --smoke-test

# Run full BitDistill benchmark
uv run python scripts/run_eval.py --model-path path/to/model --benchmark bitdistill

# Use optimized BitNet inference (requires running server)
INFERENCE_URL=http://localhost:8080 uv run python scripts/run_eval.py --use-bitnet

# Evaluate DLM (diffusion) models with Monte Carlo masking
uv run python scripts/run_eval.py --model-path path/to/dlm-checkpoint --use-dlm --mc-iterations 128
```

## Evaluation API

### Simple Python API
```python
from wf_eval import evaluate

# Basic evaluation
results = evaluate("path/to/model", benchmark="bitdistill")

# With W&B logging
results = evaluate(
    "microsoft/BitNet-b1.58-2B-4T",
    benchmark="bitdistill",
    wandb_project="wf-eval"
)

# Smoke test
results = evaluate("path/to/model", smoke_test=True)

# DLM (diffusion) model evaluation
# Uses Monte Carlo masking for loglikelihood computation
results = evaluate(
    "path/to/dlm-checkpoint",
    benchmark="bitdistill",
    use_dlm=True,
    mc_iterations=128,  # Monte Carlo iterations (default: 128)
)
```

### Available Benchmarks
| Benchmark | Tasks | Description |
|-----------|-------|-------------|
| `bitdistill` | mnli, qnli, sst2 | GLUE subset from BitDistill paper |
| `glue` | mnli, qnli, sst2 | Same as bitdistill |
| `smoke_test` | sst2 | Fast validation |

## BitNet Inference Integration

For optimized inference, use WrinkleFree-Inference-Engine:

### Start Inference Server
```bash
cd ../inference

# Serve model
uv run wf-infer serve -m model.gguf -c 4096 --port 8080
```

### Use with Eval
```bash
# Set INFERENCE_URL to use BitNet.cpp
export INFERENCE_URL=http://localhost:8080

# Run eval with optimized inference
uv run python scripts/run_eval.py --use-bitnet
```

### Remote Inference (RunPod)
```bash
# Deploy to RunPod via WrinkleFree-Deployer
cd ../deployer
sky launch ../inference/skypilot/runpod_cpu.yaml

# Get endpoint
ENDPOINT=$(sky status inference-engine --endpoint 8080)

# Run eval against remote server
INFERENCE_URL=$ENDPOINT uv run python scripts/run_eval.py --use-bitnet
```

### Remote Evaluation via SkyPilot
```bash
cd ../deployer

# Evaluate standard model
sky jobs launch skypilot/eval.yaml \
  -e MODEL_PATH=microsoft/BitNet-b1.58-2B-4T \
  -e BENCHMARK=bitdistill

# Evaluate DLM (diffusion) checkpoint from GCS
sky jobs launch skypilot/eval.yaml \
  -e MODEL_PATH=gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-5000/ \
  -e BENCHMARK=bitdistill \
  -e USE_DLM=true

# Monitor
sky jobs queue
sky jobs logs <job_id>
```

## Architecture

```
src/wf_eval/
├── api.py           # Simple evaluate() API
├── cli.py           # CLI entry point
├── models/
│   ├── hf_model.py      # HuggingFace model wrapper
│   └── bitnet_model.py  # BitNet optimized wrapper
└── tasks/
    └── utils.py         # Task utilities
```

## Configuration

Configs are in `configs/` using Hydra:
- `eval/bitdistill.yaml` - BitDistill benchmark settings
- `model/default.yaml` - Default model settings

### Hydra Overrides
```bash
# Change batch size
uv run python scripts/run_eval.py batch_size=8

# Limit samples
uv run python scripts/run_eval.py benchmark.limits.default=100

# Enable W&B
uv run python scripts/run_eval.py wandb.enabled=true wandb.project=my-project
```

## W&B Integration

```python
from wf_eval import evaluate

# Auto-logging to W&B
results = evaluate(
    model_path="path/to/model",
    wandb_project="wf-eval",
    wandb_run_name="my-eval-run",
)
```

## Development

```bash
# Run tests
uv run pytest

# Smoke test
uv run pytest tests/test_smoke.py -v

# Upload results to HF Hub
uv run python scripts/upload_results.py --results-path outputs/results.json
```

## Notes

- Uses lm-eval harness for standardized evaluation
- BitNet inference fallback to HuggingFace if server unavailable
- Results are JSON-compatible for easy logging
- W&B integration for experiment tracking
