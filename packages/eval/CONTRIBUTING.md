# Contributing to Eval (wrinklefree-eval)

> Contributor guide for navigating and understanding the eval package codebase.

## Quick Orientation

### What This Package Does
Evaluation framework for WrinkleFree BitNet models using lm-eval harness with support for GLUE benchmarks, DLM (diffusion) models, and WandB logging.

### Dependencies

| Depends On | What For |
|------------|----------|
| lm-eval | Evaluation harness |
| transformers | Model loading |
| `inference` | Optimized BitNet inference backend |

| Relationship | Package |
|--------------|---------|
| Evaluates | `training` outputs |
| Uses | `inference` for fast evaluation |
| Deploys via | `deployer` for cloud evaluation |

---

## Codebase Architecture

### Directory Structure

```
packages/eval/
├── scripts/
│   └── run_eval.py          # CLI entry point
│
├── src/wrinklefree_eval/
│   ├── __init__.py          # Package exports
│   ├── api.py               # Simple evaluate() API
│   ├── cli.py               # CLI implementation
│   ├── models/
│   │   ├── hf_model.py      # HuggingFace model wrapper for lm-eval
│   │   └── bitnet_model.py  # BitNet.cpp optimized wrapper
│   └── tasks/
│       └── utils.py         # Task utilities
│
└── configs/
    ├── eval/
    │   └── bitdistill.yaml  # Benchmark settings
    └── model/
        └── default.yaml     # Default model settings
```

### Key Components

| File | Purpose |
|------|---------|
| `api.py` | Main `evaluate()` function |
| `models/hf_model.py` | lm-eval wrapper for HuggingFace models |
| `models/bitnet_model.py` | lm-eval wrapper for BitNet inference |
| `cli.py` | Command-line interface |

---

## Code Flow

### Evaluation Flow

```
evaluate(model_path, benchmark="bitdistill")
│
├─► Resolve model wrapper:
│   ├─► If use_bitnet: BitNetModel (uses inference server)
│   ├─► If use_dlm: DLMEvalHarness (Monte Carlo masking)
│   └─► Default: HFModel (standard HuggingFace)
│
├─► Load lm-eval tasks from BENCHMARK_PRESETS
│
├─► lm_eval.evaluator.simple_evaluate()
│   └─► Runs tasks with model wrapper
│
├─► Format results
│
└─► Optional: Log to WandB
```

### DLM Evaluation Flow

```
DLM models use Monte Carlo masking for loglikelihood:

evaluate(..., use_dlm=True, mc_iterations=128)
│
├─► Create DLMEvalHarness
│
└─► For each loglikelihood request:
    │
    ├─► Run mc_iterations times:
    │   ├─► Random mask positions
    │   ├─► Forward pass
    │   └─► Collect logits at mask positions
    │
    └─► Average log probabilities
```

---

## Entry Points

| Task | Start Here |
|------|------------|
| Add new benchmark | `api.py:BENCHMARK_PRESETS` |
| Add new model wrapper | `models/` directory |
| Modify evaluation logic | `api.py:evaluate()` |
| Add CLI option | `cli.py` |
| Modify task handling | `tasks/utils.py` |

---

## Patterns & Conventions

### Benchmark Presets

```python
# api.py
BENCHMARK_PRESETS = {
    "bitdistill": ["mnli", "qnli", "sst2"],
    "glue": ["mnli", "qnli", "sst2"],
    "smoke_test": ["sst2"],
}
```

### lm-eval Model Wrapper Pattern

```python
# models/hf_model.py
class HFModel(lm_eval.api.model.LM):
    def loglikelihood(self, requests):
        # Compute log probabilities for requests
        ...

    def generate_until(self, requests):
        # Generate completions
        ...
```

### WandB Integration Pattern

```python
results = evaluate(
    model_path="path/to/model",
    wandb_project="wrinklefree-eval",
    wandb_run_name="my-eval",
)
# Automatically logs metrics to WandB
```

---

## Testing

### Running Tests

```bash
# All tests
uv run --package wrinklefree-eval pytest packages/eval/tests/ -v

# Smoke test
uv run --package wrinklefree-eval pytest packages/eval/tests/test_smoke.py -v
```

### Manual Testing

```bash
# Quick smoke test (10 samples)
uv run --package wrinklefree-eval python packages/eval/scripts/run_eval.py --smoke-test

# Full evaluation
uv run --package wrinklefree-eval python packages/eval/scripts/run_eval.py \
  --model-path HuggingFaceTB/SmolLM2-135M --benchmark bitdistill
```

---

## Common Tasks

### Adding a New Benchmark

1. Add task list to `api.py:BENCHMARK_PRESETS`:
   ```python
   BENCHMARK_PRESETS["my_benchmark"] = ["task1", "task2"]
   ```

2. Update documentation

### Adding a New Model Wrapper

1. Create `models/my_model.py`
2. Implement lm-eval `LM` interface (`loglikelihood`, `generate_until`)
3. Add to model resolution logic in `api.py`
4. Add tests

### Using with Inference Server

```bash
# Start inference server (from inference package)
cd packages/inference
./scripts/launch_rust_gateway.sh --model models/model.gguf

# Run eval with BitNet backend
INFERENCE_URL=http://localhost:30000 \
  uv run --package wrinklefree-eval python packages/eval/scripts/run_eval.py --use-bitnet
```

---

## Gotchas & Tips

- **lm-eval Task Renaming**: Recent lm-eval versions renamed `glue_*` tasks. The `TASK_MAPPING` in api.py handles this.

- **DLM Monte Carlo**: DLM evaluation requires `mc_iterations` parameter. More iterations = more accurate but slower. Default 128 is reasonable.

- **Batch Size**: Large batch sizes can OOM on smaller GPUs. Use `batch_size` override for memory-constrained environments.

- **Inference Fallback**: If `INFERENCE_URL` is not set or server is unavailable, evaluation falls back to HuggingFace inference.

- **Results Format**: Results are JSON-serializable for easy logging and comparison.

- **WandB API Key**: Set `WANDB_API_KEY` environment variable for automatic logging.
