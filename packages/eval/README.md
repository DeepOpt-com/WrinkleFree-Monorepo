# WrinkleFree-Eval

> Part of [WrinkleFree Monorepo](https://github.com/DeepOpt-com/WrinkleFree-Monorepo) - Evaluation harness for quantized LLMs.

Clean evaluation harness for BitDistill models. Evaluates quantized LLMs on the benchmarks from the [BitDistill paper](https://arxiv.org/abs/2510.13998).

## Benchmarks

| Benchmark | Tasks | Metrics |
|-----------|-------|---------|
| **GLUE** | MNLI, QNLI, SST-2 | Accuracy |
| **CNN/DailyMail** | Summarization | ROUGE-1, ROUGE-2, ROUGE-L, BLEU |

## Installation

```bash
# Clone the monorepo
git clone --recurse-submodules git@github.com:DeepOpt-com/WrinkleFree-Monorepo.git
cd WrinkleFree-Monorepo

# Install all packages
uv sync --all-packages
```

## Quick Start

### Python API (Recommended)

```python
from wrinklefree_eval import evaluate

# Full BitDistill benchmarks
results = evaluate("path/to/model", benchmark="bitdistill")

# Quick smoke test (10 samples per task)
results = evaluate("path/to/model", benchmark="smoke_test")

# GLUE only
results = evaluate("path/to/model", benchmark="glue")

# With W&B logging
results = evaluate("path/to/model", benchmark="bitdistill",
                   wandb_project="my-project")

# DLM (diffusion) model evaluation
# For Fast-dLLM v2 trained models, use Monte Carlo masking
results = evaluate("path/to/dlm-checkpoint", benchmark="bitdistill",
                   use_dlm=True, mc_iterations=128)

# Access results
print(results["sst2"]["acc"])  # 0.92
print(results["mnli"]["acc"])  # 0.85
```

### CLI with Hydra

```bash
# Full BitDistill benchmarks
uv run python -m wrinklefree_eval \
    model_path=HuggingFaceTB/SmolLM2-135M \
    benchmark=bitdistill

# Smoke test (quick validation)
uv run python -m wrinklefree_eval \
    model_path=path/to/checkpoint \
    benchmark=smoke_test

# DLM (diffusion) model evaluation
uv run python scripts/run_eval.py \
    --model-path path/to/dlm-checkpoint \
    --benchmark bitdistill \
    --use-dlm \
    --mc-iterations 128

# GLUE only with custom settings
uv run python -m wrinklefree_eval \
    model_path=path/to/model \
    benchmark=glue \
    batch_size=16 \
    device=cuda \
    output_dir=./my_results
```

## Benchmark Presets

| Preset | Tasks | Description |
|--------|-------|-------------|
| `bitdistill` | MNLI, QNLI, SST-2, CNN/DM | Full paper benchmarks |
| `glue` | MNLI, QNLI, SST-2 | Classification only (faster) |
| `summarization` | CNN/DailyMail | Generation + ROUGE |
| `smoke_test` | SST-2, CNN/DM (10 samples) | Quick pipeline validation |

## Configuration

All settings are YAML-configurable via Hydra:

```yaml
# configs/eval.yaml
model_path: null  # Required
benchmark: bitdistill
device: cuda
dtype: bfloat16
batch_size: auto
smoke_test: false
output_dir: ./eval_results
```

Override via CLI:
```bash
uv run python -m wrinklefree_eval \
    model_path=/path/to/model \
    benchmark=smoke_test \
    dtype=float16 \
    +smoke_test=true
```

## Evaluating BitNet Models

For models trained with WrinkleFree-1.58Quant:

```python
from wrinklefree_eval import evaluate

# Stage 2 checkpoint
results = evaluate(
    "outputs/bitdistill_smollm2_stage2/checkpoint",
    benchmark="bitdistill",
    use_bitnet=True,  # Use BitNet kernels if available
)
```

## Backend

Built on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) by EleutherAI. Uses **real benchmarks**, not approximations.

## Remote Evaluation (via Deployer)

Run evaluations on remote GPUs using WrinkleFree-Deployer:

```bash
cd ../deployer

# Evaluate HuggingFace model
sky jobs launch skypilot/eval.yaml \
  -e MODEL_PATH=HuggingFaceTB/SmolLM2-135M \
  -e BENCHMARK=smoke_test

# Evaluate GCS checkpoint with W&B logging
sky jobs launch skypilot/eval.yaml \
  -e MODEL_PATH=gs://bucket/checkpoint \
  -e BENCHMARK=bitdistill \
  -e WANDB_PROJECT=wrinklefree

# Evaluate DLM (diffusion) model with Monte Carlo masking
sky jobs launch skypilot/eval.yaml \
  -e MODEL_PATH=gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-5000/ \
  -e BENCHMARK=bitdistill \
  -e USE_DLM=true \
  -e MC_ITERATIONS=128

# Monitor jobs
sky jobs queue
sky jobs logs <job_id>
```

See [WrinkleFree-Deployer](../deployer) for full deployment documentation.

## Optional Dependencies

```bash
# W&B logging support
uv sync --extra wandb

# GCS upload support
uv sync --extra gcs

# All optional features
uv sync --extra all
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run smoke test
uv run pytest tests/test_smoke.py -v
```
