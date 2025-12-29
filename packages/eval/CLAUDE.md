# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree-Eval is the evaluation framework for WrinkleFree BitNet models:
- **Framework**: lm-eval harness
- **Benchmarks**: BitDistill paper benchmarks (GLUE subset)
- **Models**: HuggingFace models + BitNet.cpp optimized inference
- **Config**: Hydra
- **Package management**: uv

## Quick Start

```bash
# Install dependencies
uv sync

# Run smoke test (sst2 only, 10 samples)
uv run python scripts/run_eval.py --smoke-test

# Run full BitDistill benchmark
uv run python scripts/run_eval.py model_path=path/to/model benchmark=bitdistill

# Use optimized BitNet inference (requires running server)
INFERENCE_URL=http://localhost:8080 uv run python scripts/run_eval.py --use-bitnet
```

## Evaluation API

### Simple Python API
```python
from wrinklefree_eval import evaluate

# Basic evaluation
results = evaluate("path/to/model", benchmark="bitdistill")

# With W&B logging
results = evaluate(
    "microsoft/BitNet-b1.58-2B-4T",
    benchmark="bitdistill",
    wandb_project="wrinklefree-eval"
)

# Smoke test
results = evaluate("path/to/model", smoke_test=True)
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
cd ../WrinkleFree-Inference-Engine

# Serve model
uv run wrinklefree-inference serve -m model.gguf -c 4096 --port 8080
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
cd ../WrinkleFree-Deployer
sky launch ../WrinkleFree-Inference-Engine/skypilot/runpod_cpu.yaml

# Get endpoint
ENDPOINT=$(sky status inference-engine --endpoint 8080)

# Run eval against remote server
INFERENCE_URL=$ENDPOINT uv run python scripts/run_eval.py --use-bitnet
```

## Architecture

```
src/wrinklefree_eval/
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
from wrinklefree_eval import evaluate

# Auto-logging to W&B
results = evaluate(
    model_path="path/to/model",
    wandb_project="wrinklefree-eval",
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
