# Ax Bayesian Optimization Benchmarks

Hyperparameter optimization for 1.58-bit training using [Ax](https://ax.dev/).

## Quick Start

### Stage 1.9 (Layer-wise Distillation)

```bash
uv run python scripts/benchmark_stage1_9.py --num-trials 10
```

Search space:
- Learning rate (1e-5 to 1e-3)
- Loss type (mse_normalized, cosine, kl)
- LM loss weight (0.0 to 0.5)
- Temperature (1.0 to 4.0)

### Stage 2 (Pre-training)

```bash
uv run python scripts/benchmark_stage2.py --num-trials 10
```

Search space:
- Optimizer (muon, adamw_8bit, apollo)
- Learning rate (1e-5 to 1e-2)
- Batch size (4 to 64)
- Distillation params (lambda_logits, gamma_attention, temperature)

## Options

```bash
# Use different model
uv run python scripts/benchmark_stage2.py --model Qwen/Qwen3-4B-Base

# More trials
uv run python scripts/benchmark_stage2.py --num-trials 30

# Resume from Stage 1 checkpoint
uv run python scripts/benchmark_stage2.py --stage1-checkpoint ./outputs/stage1_checkpoint
```

## Output

Results are saved to `./benchmark_results/{stage}/`:
- `experiment.json` - Ax experiment state (can resume)
- `trials.csv` - All trial results

## API

```python
from benchmark import BenchmarkRunner, BenchmarkAxClient, BenchmarkMetrics

# Create Ax client with search space
ax_client = BenchmarkAxClient(search_space_config, experiment_name="my_exp")

# Get next trial parameters
params, trial_idx = ax_client.get_next_trial()

# Run benchmark
runner = BenchmarkRunner(model_name="HuggingFaceTB/SmolLM2-135M")
metrics = runner.run_trial(params, trial_id=trial_idx)

# Report results
ax_client.complete_trial(trial_idx, metrics)

# Get best parameters
best = ax_client.get_best_parameters()
```

## Cost Estimate

H100 on RunPod @ $2.89/hr:
- 10 trials × 300 steps × ~1 sec/step ≈ 50 min ≈ **~$2.50**
