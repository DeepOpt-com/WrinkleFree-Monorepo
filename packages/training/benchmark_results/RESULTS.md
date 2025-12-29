# BitNet 1.58-bit Training Optimizer Benchmark Results

**Date:** December 19, 2024
**Hardware:** NVIDIA RTX 4090 (24GB)
**Trials:** 50 (Sobol initialization + BoTorch Bayesian optimization)
**Metric:** Convergence Efficiency = (initial_loss - final_loss) / wall_time / peak_memory_gb

## Executive Summary

After 50 trials of Bayesian hyperparameter optimization, **Apollo optimizer** emerged as the clear winner for BitNet 1.58-bit training, achieving a convergence efficiency of **0.0289 loss/sec/GB**.

## Best Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **optimizer_type** | `apollo` | Clear winner over Muon, AdamW 8bit, Apollo Mini |
| **learning_rate** | 7.51e-05 | Mid-range, ~75 micros |
| **gradient_accumulation_steps** | 1 | No accumulation works best |
| **lambda_logits** | 15.60 | Distillation logits weight |
| **gamma_attention** | 4.29e-07 | Very small attention loss weight |
| **temperature** | 9.49 | High temperature for distillation |
| **influence_enabled** | True | Influence tracking helps |
| **influence_lambda_reg** | 3.14e-06 | Small regularization |
| **influence_threshold** | -0.12 | Negative threshold |

## Performance Metrics

**Best convergence efficiency:** 0.0289 loss/sec/GB

### Optimization Progress

The Bayesian optimizer found progressively better configurations:

| Trial | Best Convergence | Improvement |
|-------|-----------------|-------------|
| 0-3 | 0.0072 | Baseline (Muon) |
| 4-10 | 0.0118 | +64% (Apollo Mini) |
| 11-21 | 0.0251 | +113% (Apollo) |
| 22-50 | 0.0289 | +15% (Apollo, tuned) |

**Total improvement:** 4x from initial to best configuration

## Optimizer Comparison

The Bayesian optimizer explored all 4 optimizers and heavily favored Apollo:

| Optimizer | Trials Explored | Best Convergence | Notes |
|-----------|-----------------|------------------|-------|
| **Apollo** | 36 (72%) | 0.0289 | Winner - most explored |
| Muon | 6 (12%) | 0.0072 | Fast but less efficient |
| Apollo Mini | 4 (8%) | 0.0118 | Good but Apollo better |
| AdamW 8bit | 4 (8%) | 0.0067 | Memory efficient but slower |

## Key Findings

### 1. Apollo Dominates for BitNet Training
Apollo optimizer consistently outperformed alternatives. The Bayesian optimizer naturally converged to exploring Apollo configurations, indicating strong signal that Apollo works best for ternary weight training.

### 2. No Gradient Accumulation Preferred
All top configurations used `gradient_accumulation_steps=1`. For this model/batch size, direct updates outperform accumulated gradients.

### 3. High Distillation Temperature
Optimal temperature around 9.5 suggests soft targets with high entropy work best for knowledge distillation to 1.58-bit models.

### 4. Minimal Attention Loss Weight
The near-zero `gamma_attention` (4.29e-07) suggests attention distillation provides minimal benefit for this setup.

### 5. Influence Tracking Helps
Enabling influence tracking with small regularization improved convergence.

## Training Details

- **Model:** BitNet Llama (512 hidden, 4 layers, 8 heads)
- **Sequence Length:** 512 tokens
- **Dataset:** FineWeb-Edu (sample-10BT)
- **Steps per Trial:** 500
- **Target Memory:** 20GB (auto batch sizing)

## Recommended Configuration

For production BitNet 1.58-bit training on similar hardware:

```yaml
optimizer:
  type: apollo
  learning_rate: 7.5e-05

training:
  gradient_accumulation_steps: 1
  batch_size: auto  # Let system optimize for memory

distillation:
  lambda_logits: 15.0
  gamma_attention: 0.0  # Can disable
  temperature: 9.5

influence:
  enabled: true
  lambda_reg: 3.0e-06
  threshold: -0.12
```

## Files

- `analysis.json` - Structured results data
- `experiment.json` - Full Ax experiment state (can resume)
- `plots/optimization_trace.png` - Convergence over trials

## Reproducibility

To reproduce or continue optimization:

```bash
# Resume from checkpoint
uv run python -m benchmark.run_benchmark \
  --config benchmark/config/search_space.yaml \
  --resume benchmark_results/experiment.json
```
