# WrinkleFree-Eval

Clean evaluation harness for BitDistill models. Evaluates quantized LLMs on the benchmarks from the [BitDistill paper](https://arxiv.org/abs/2510.13998).

## Benchmarks

| Benchmark | Tasks | Metrics |
|-----------|-------|---------|
| **GLUE** | MNLI, QNLI, SST-2 | Accuracy |
| **CNN/DailyMail** | Summarization | ROUGE-1, ROUGE-2, ROUGE-L, BLEU |

## Installation

```bash
cd WrinkleFree-Eval
uv sync
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

# Access results
print(results["glue_sst2"]["acc"])           # 0.92
print(results["cnn_dailymail_summarization"]["rouge1"])  # 0.45
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
cd ../WrinkleFree-Deployer

# Evaluate HuggingFace model
sky launch skypilot/eval.yaml \
  --env MODEL_PATH=HuggingFaceTB/SmolLM2-135M \
  --env BENCHMARK=smoke_test

# Evaluate GCS checkpoint with W&B logging
sky launch skypilot/eval.yaml \
  --env MODEL_PATH=gs://bucket/checkpoint \
  --env BENCHMARK=bitdistill \
  --env WANDB_PROJECT=wrinklefree
```

See [WrinkleFree-Deployer](../WrinkleFree-Deployer) for full deployment documentation.

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
