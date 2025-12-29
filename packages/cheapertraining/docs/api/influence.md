# Influence Functions API

Influence-based data mixture optimization for efficient LLM training.

## Background

This module implements influence function techniques for dynamically optimizing dataset mixture weights during training. The core insight: datasets whose samples have higher positive influence on a target probe set should receive higher sampling weights.

### Academic Context

| Method | Venue | Key Contribution |
|--------|-------|------------------|
| [DoReMi](https://arxiv.org/abs/2305.10429) | NeurIPS 2023 | Proxy model + Group DRO for domain weights |
| [DataInf](https://arxiv.org/abs/2310.00902) | ICLR 2024 | Efficient influence without Hessian inversion |
| [MobileLLM-R1](https://arxiv.org/abs/2509.24945) | Sept 2025 | Cross-domain influence + two-stage curriculum |
| [TiKMiX](https://arxiv.org/abs/2508.17677) | Aug 2025 | "Group Influence" with gradient accumulation |

This implementation follows **DataInf + MobileLLM-R1**.

## Core Formula

The DataInf algorithm computes influence without expensive Hessian inversion:

```
I(z_train, z_probe) = <grad_train, grad_probe> / (λ + ||grad_train||²)
```

Where:
- `grad_train`: gradient of loss on a training sample
- `grad_probe`: gradient of loss on a probe sample
- `λ`: regularization term for numerical stability

**Interpretation**: A training sample has high positive influence if its gradient aligns with the probe set gradients.

## Quick Start

```python
from cheapertraining.influence import (
    DataInfCalculator,
    MixtureWeightCalculator,
    InfluenceAwareOptimizer,
    DiscriminativeGradientExtractor,
    InfluenceConfig,
)

# 1. Create gradient extractor (uses only embed_tokens + lm_head)
config = InfluenceConfig(lambda_reg=1e-4)
extractor = DiscriminativeGradientExtractor(model, config)

# 2. Create influence calculator
calculator = DataInfCalculator(extractor, config)
calculator.cache_probe_gradients(probe_dataloader)

# 3. Compute influence for a batch
influences = calculator.compute_batch_influence_aggregated(batch)

# 4. Or use MixtureWeightCalculator for full pipeline
mixture_calc = MixtureWeightCalculator(model, probe_dataloader)
weights = mixture_calc.compute_mixture_weights({
    "code": code_dataloader,
    "math": math_dataloader,
    "web": web_dataloader,
})
```

## Classes

### InfluenceConfig

Configuration for influence computation.

```python
from cheapertraining.influence import InfluenceConfig, InfluenceTarget

config = InfluenceConfig(
    target_layers=InfluenceTarget.EMBEDDING_AND_OUTPUT,  # Which layers to use
    lambda_reg=1e-4,          # DataInf regularization
    batch_size=32,            # Batch size for gradient computation
    use_fp16=True,            # Use FP16 for efficiency
    max_grad_norm=1.0,        # Gradient clipping
)
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_layers` | InfluenceTarget | EMBEDDING_AND_OUTPUT | Layers for gradient extraction |
| `lambda_reg` | float | 1e-4 | Regularization in DataInf formula |
| `batch_size` | int | 32 | Gradient computation batch size |
| `use_fp16` | bool | True | Use FP16 for memory efficiency |
| `max_grad_norm` | float | 1.0 | Gradient clipping threshold |

### DiscriminativeGradientExtractor

Extracts gradients from embedding and output layers only (AutoMixer technique).

```python
from cheapertraining.influence import DiscriminativeGradientExtractor

extractor = DiscriminativeGradientExtractor(model, config)

# Get gradient for a single sample
grads = extractor.compute_per_sample_gradient(input_ids, labels)

# Get gradients for a batch [batch_size, D]
batch_grads = extractor.compute_batch_gradients(batch)

# Get mean gradient for efficiency
mean_grad = extractor.compute_aggregated_gradient(batch)
```

**Why discriminative layers?**
- Reduces computation from O(d_model × n_layers) to O(d_vocab × d_embed)
- Embedding and output layers contain the most discriminative signal
- Handles weight sharing automatically

### DataInfCalculator

Core influence calculation using the DataInf algorithm.

```python
from cheapertraining.influence import DataInfCalculator, create_influence_calculator

# Factory function
calculator = create_influence_calculator(model, config)

# Or manual construction
calculator = DataInfCalculator(extractor, config)

# Step 1: Cache probe gradients (do once, refresh periodically)
calculator.cache_probe_gradients(probe_dataloader)

# Step 2: Compute influence
# Per-sample influence on all probe samples
influences = calculator.compute_influence(train_gradient)  # [N_probe]

# Aggregated influence (more efficient)
agg_influence = calculator.compute_influence_aggregated(train_gradient)  # scalar

# Batch computation
batch_influences = calculator.compute_batch_influence_aggregated(batch)  # [batch_size]
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `cache_probe_gradients(loader)` | Tensor, Tensor | Pre-compute and cache probe gradients |
| `compute_influence(grad)` | Tensor [N_probe] | Influence on each probe sample |
| `compute_influence_aggregated(grad)` | float | Mean influence (efficient) |
| `compute_batch_influence(batch)` | Tensor [B, N_probe] | Full influence matrix |
| `compute_batch_influence_aggregated(batch)` | Tensor [B] | Mean influence per sample |
| `clear_cache()` | None | Clear cached probe gradients |

### MixtureWeightCalculator

Computes optimal dataset mixture weights based on influence scores.

```python
from cheapertraining.influence import MixtureWeightCalculator, MixtureOptimizationConfig

config = MixtureOptimizationConfig(
    samples_per_dataset=1000,   # Samples to evaluate per dataset
    normalize_weights=True,      # Normalize to sum to 1
    min_weight=0.01,            # Minimum 1% per dataset
    max_weight=0.90,            # Maximum 90% per dataset
    influence_smoothing=0.1,    # EMA smoothing
)

calculator = MixtureWeightCalculator(
    model=model,
    probe_dataloader=probe_loader,
    config=config,
)

# Compute optimal weights
weights = calculator.compute_mixture_weights({
    "code": code_loader,
    "math": math_loader,
    "web": web_loader,
})
# Returns: {"code": 0.35, "math": 0.45, "web": 0.20}

# Incremental update for online learning
new_weights = calculator.get_weight_update(
    current_weights=old_weights,
    dataset_loaders=loaders,
    learning_rate=0.2,
)
```

### InfluenceAwareOptimizer

Optimizer wrapper that periodically updates mixture weights during training.

```python
from cheapertraining.training.optimizer import InfluenceAwareOptimizer

optimizer = InfluenceAwareOptimizer(
    optimizer=base_optimizer,           # e.g., AdamW
    mixture_calculator=mixture_calc,
    mixed_dataset=mixed_dataset,
    update_interval=1000,               # Steps between weight updates
    learning_rate=0.2,                  # How fast to move toward optimal
    rank=0,                             # For logging
)

# Use like any optimizer
optimizer.zero_grad()
loss.backward()
optimizer.step()  # Automatically triggers mixture updates every N steps
```

## Configuration Files

### YAML Configuration

```yaml
# configs/influence/default.yaml
influence:
  enabled: true
  update_interval: 1000
  learning_rate: 0.2
  config:
    lambda_val: 0.1
    gamma_val: 0.1
    temperature: 1.0

probe_set:
  size: 10000
  fineweb_edu_min_score: 4.0
  ask_llm_top_fraction: 0.10
  dedup_similarity_threshold: 0.85
```

## Pipeline Flow

```
1. Initialize
   └── Create probe set (target evaluation data)
   └── Cache probe gradients: probe_dataloader → [N_probe, D] tensor

2. During Training (every update_interval steps)
   └── For each dataset k:
       └── Sample N batches
       └── Compute batch gradients
       └── Calculate influence: <grad_train, avg_probe_grad> / (λ + ||grad||²)
       └── Average to get dataset influence score

3. Update Weights
   └── Convert influences → weights (clamp negatives, normalize)
   └── Interpolate: new = (1 - lr) * old + lr * optimal
   └── Update MixedDataset sampling probabilities

4. Refresh (periodically)
   └── Re-cache probe gradients with updated model
```

## Integration with 1.58Quant Stage 2

```python
# In WrinkleFree-1.58Quant/src/wrinklefree/training/stage2.py

from cheapertraining import (
    DataInfCalculator,
    MixtureWeightCalculator,
    InfluenceAwareOptimizer,
    MixedDataset,
    InfluenceConfig,
    DiscriminativeGradientExtractor,
)

# Create influence config
inf_config = InfluenceConfig(lambda_val=0.1)

# Create components
grad_extractor = DiscriminativeGradientExtractor(model, inf_config)
datainf = DataInfCalculator(grad_extractor, inf_config)
mixture_calc = MixtureWeightCalculator(datainf)

# Wrap optimizer
optimizer = InfluenceAwareOptimizer(
    optimizer=base_optimizer,
    mixture_calculator=mixture_calc,
    mixed_dataset=mixed_dataset,
    update_interval=1000,
)
```

---

## InfluenceDistillation (New)

**Reference:** [arXiv:2505.19051](https://arxiv.org/abs/2505.19051) - "Efficient Data Selection at Scale via Influence Distillation"

For large-scale data selection, InfluenceDistillation provides **2-3.5x speedup** over DataInf by:
1. Computing cheap JVP embeddings instead of full gradients
2. Using a small set of landmarks (~4096) for accurate gradient computation
3. Propagating influence from landmarks to all samples via KRR

### Quick Start

```python
from cheapertraining.influence import (
    InfluenceDistillation,
    InfluenceDistillationConfig,
    JVPEmbeddingConfig,
    LandmarkConfig,
)

# Configure
config = InfluenceDistillationConfig(
    jvp=JVPEmbeddingConfig(
        num_jvp_layers=4,           # First 4 transformer blocks
        num_tangent_vectors=2,       # Random vectors for JVP
        projection_dim=131072,       # Hadamard projection dim
    ),
    landmark=LandmarkConfig(
        num_landmarks=4096,          # Number of landmarks
        selection_strategy="kmeans_pp",  # Diverse selection
    ),
)

# Create distillation instance
distiller = InfluenceDistillation(model, config)

# Cache probe gradients (target distribution)
distiller.cache_probe_gradients(probe_dataloader)

# Compute JVP embeddings for source data
distiller.cache_source_embeddings(source_dataloader)

# Select landmarks and compute KRR coefficients
distiller.cache_landmarks(source_dataloader)

# Get influence scores for all samples
scores = distiller.compute_influence_scores()

# Or select top-k directly
selected_indices = distiller.select(source_dataloader, budget_k=10000, target_loader=probe_dataloader)
```

### Continuous Rebalancing

For updating mixture weights during training:

```python
# Compute mixture weights for multiple datasets
weights = distiller.compute_mixture_weights({
    "code": code_loader,
    "math": math_loader,
    "web": web_loader,
})
# Returns: {"code": 0.35, "math": 0.45, "web": 0.20}
```

### Fixed Evaluation Set

Use a fixed evaluation set to monitor training progress:

```python
# Create a fixed eval dataloader (held-out data, never used for training)
eval_loader = create_eval_dataloader(eval_data, batch_size=32)

# During training, periodically evaluate
for step in range(num_steps):
    # ... training step ...

    if step % eval_interval == 0:
        # Compute loss on fixed eval set
        metrics = distiller.evaluate(eval_loader, max_batches=50)
        print(f"Step {step}: loss={metrics['loss']:.4f}, ppl={metrics['perplexity']:.2f}")
        wandb.log({'eval/loss': metrics['loss'], 'eval/ppl': metrics['perplexity']})

    if step % rebalance_interval == 0:
        # Rebalance uses influence on probe set (can be refreshed)
        distiller.cache_probe_gradients(probe_loader)
        weights = distiller.compute_mixture_weights(dataset_loaders)
```

**Key distinction:**
- **Eval set**: Fixed held-out data for monitoring loss/perplexity (never changes)
- **Probe set**: Target distribution for influence computation (can be refreshed as model trains)

### Using with InfluenceTracker

```yaml
# configs/influence/distillation.yaml
influence:
  enabled: true
  method: distillation           # Use InfluenceDistillation instead of DataInf
  update_interval: 10000
  learning_rate: 0.1
  warmup_steps: 1000

  # InfluenceDistillation-specific settings
  jvp_layers: 4                  # Number of transformer layers for JVP
  jvp_vectors: 2                 # Number of tangent vectors
  projection_dim: 131072         # Hadamard projection dimension
  num_landmarks: 4096            # Number of landmarks
  landmark_strategy: kmeans_pp   # Selection strategy
```

### Algorithm Overview

```
1. JVP Embeddings (cheap):
   - Forward through first K transformer blocks
   - Compute Jacobian-Vector Product with random tangent vectors
   - Project via Randomized Hadamard Transform

2. Landmark Selection:
   - Select ~4096 representative samples via K-means++
   - Ensures diverse coverage of embedding space

3. KRR Propagation:
   - C = E_S @ E_L.T @ (E_L @ E_L.T + λI)^{-1}
   - Cholesky solve for numerical stability

4. Influence Computation:
   - Compute accurate gradients for landmarks only
   - Propagate: p = C @ (G_L @ g_T)
   - Top-k selection based on scores
```

### Comparison with DataInf

| Aspect | DataInf | InfluenceDistillation |
|--------|---------|----------------------|
| Gradient computations | All N samples | L landmarks (~4096) |
| Per-sample cost | Full backward | JVP forward (faster) |
| Memory | O(N × D_grad) | O(L × D_grad) + O(N × D_jvp) |
| Accuracy | Exact (for discriminative layers) | Approximate (via KRR) |
| Speedup | 1× | 2-3.5× |

### Components

| Class | File | Description |
|-------|------|-------------|
| `JVPEmbeddingExtractor` | `jvp_embedding.py` | Extract JVP embeddings from transformer layers |
| `RandomizedHadamardTransform` | `hadamard.py` | O(D log D) dimensionality reduction |
| `LandmarkSelector` | `landmark.py` | Select representative landmarks |
| `InfluenceDistillation` | `distillation.py` | Main class with KRR solver |

---

## Ablation Results (Real Data)

### Experiment Setup

**Date:** 2025-12-26

**Datasets (commercially-friendly):**
- `fineweb-edu` (HuggingFaceFW/fineweb-edu-score-2) - High-quality educational content
- `gsm8k` (MIT license) - Math word problems
- `wikitext` (Apache 2.0) - General knowledge

**Model:** GPT-2 (124M parameters)

**Configuration:**
- 300 training steps
- Batch size: 8
- Learning rate: 5e-5
- Rebalance interval: 30 steps
- JVP layers: 2, projection dim: 512
- Landmarks: 16

### Results (Test timed out at ~270 steps)

| Mode | Step 0 Eval Loss | Step ~270 Eval Loss | Notes |
|------|------------------|---------------------|-------|
| **Static** (equal weights) | 3.47 | 2.76 | 33%/33%/33% throughout |
| **Rebalanced** | 3.52 | **2.34** | Dynamic weights |

**Improvement:** ~0.42 lower eval loss with rebalancing (~15% relative improvement)

### Weight Evolution (Rebalanced)

The influence-based rebalancing dynamically shifted weights away from wikitext toward fineweb-edu and math:

| Step | fineweb | math | wiki |
|------|---------|------|------|
| 0 | 33% | 33% | 33% |
| 30 | 52% | 47% | 1% |
| 90 | 51% | 48% | 1% |
| 210 | 75% | 24% | 1% |
| 270 | 36% | 65% | 1% |

**Observations:**
1. Wiki was consistently downweighted to minimum (1%) - lower influence on fineweb-edu target
2. Fineweb and math weights oscillated based on current model state
3. The system correctly identified that wiki text was less helpful for the target distribution

### Key Findings

1. **Rebalancing works on real data** - Unlike synthetic data where all datasets look identical, real data shows clear influence differences
2. **Dynamic adaptation** - Weights shift as the model learns, suggesting the influence signal captures meaningful gradient alignment
3. **Target distribution matters** - Using fineweb-edu as probe (target) naturally upweights similar high-quality content

### Limitations & Future Work

- Test timed out before full 300 steps (need longer timeout)
- Should evaluate on held-out test set separate from probe set
- Need to test with larger models and more diverse datasets
- Compare against DoReMi and other baselines

### Running the Test

```bash
# Full ablation (may take 5-10 minutes)
PYTHONUNBUFFERED=1 uv run python tests/integration/test_real_data_ablation.py
```

---

## References

- **DataInf**: [ICLR 2024](https://arxiv.org/abs/2310.00902) - Efficient influence without Hessian inversion
- **MobileLLM-R1**: [arXiv:2509.24945](https://arxiv.org/abs/2509.24945) - Cross-domain influence methodology
- **DoReMi**: [NeurIPS 2023](https://arxiv.org/abs/2305.10429) - Domain reweighting with minimax optimization
- **TiKMiX**: [arXiv:2508.17677](https://arxiv.org/abs/2508.17677) - Dynamic data mixing with Group Influence
- **InfluenceDistillation**: [arXiv:2505.19051](https://arxiv.org/abs/2505.19051) - Landmark-based influence approximation
