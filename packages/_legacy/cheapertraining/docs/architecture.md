# Architecture

High-level system design for CheaperTraining, an influence-based data selection library for efficient LLM training.

## System Overview

CheaperTraining provides influence-based data mixture optimization for LLM training. Rather than implementing a full training pipeline, it focuses on computing optimal dataset mixture weights using influence functions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CheaperTraining                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Influence Functions Pipeline                    │   │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│   │  │   Probe     │───▶│   DataInf   │───▶│  Mixture Weight         │  │   │
│   │  │   Set       │    │ Calculator  │    │  Calculator             │  │   │
│   │  └─────────────┘    └─────────────┘    └─────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│        ┌───────────────────────────┼───────────────────────────┐           │
│        │                           │                           │           │
│  ┌─────▼─────┐              ┌──────▼──────┐             ┌──────▼──────┐   │
│  │Optimizer  │              │    Data     │             │  Gradient   │   │
│  │Factory   │              │   Mixing    │             │  Extractor  │   │
│  └───────────┘              └─────────────┘             └─────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Influence Functions Module (`src/cheapertraining/influence/`)

The heart of CheaperTraining. Implements influence-based data mixture optimization following established research (DataInf, MobileLLM-R1, AutoMixer).

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Influence Functions Module                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   config.py                                                                  │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  InfluenceConfig         │  Target layers, lambda_reg, caching     │    │
│   │  InfluenceTarget         │  EMBEDDING_ONLY | OUTPUT_ONLY | BOTH    │    │
│   │  MixtureOptimizationConfig│  Weight constraints, update intervals  │    │
│   │  ProbeSetConfig          │  Size, quality filtering thresholds     │    │
│   │  SelfBoostingConfig      │  Rejection sampling parameters          │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   gradient.py                                                                │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  DiscriminativeGradientExtractor                                    │    │
│   │  ──────────────────────────────                                     │    │
│   │  • Extracts gradients from embed_tokens and lm_head only           │    │
│   │  • Handles weight sharing (common in LLMs)                         │    │
│   │  • Per-sample and batch gradient computation                        │    │
│   │  • O(d_vocab × d_embed) instead of O(d_model × n_layers)           │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   datainf.py                                                                 │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  DataInfCalculator                                                  │    │
│   │  ─────────────────                                                  │    │
│   │  • Tractable influence without Hessian inversion                    │    │
│   │  • I(z_train, z_probe) = <g_train, g_probe> / (λ + ||g_train||²)   │    │
│   │  • Probe gradient caching for efficiency                            │    │
│   │  • Per-sample and aggregated influence computation                  │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   mixture_calculator.py                                                      │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  MixtureWeightCalculator                                            │    │
│   │  ───────────────────────                                            │    │
│   │  • Compute optimal dataset weights using influence on probe set    │    │
│   │  • Higher influence → higher sampling weight                        │    │
│   │  • EMA smoothing for stable weight updates                          │    │
│   │  • Min/max weight constraints                                       │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### DataInf Algorithm (Core Innovation)

The DataInf algorithm enables tractable influence calculation without expensive Hessian inversion:

```
I(z_train, z_probe) = <grad_train, grad_probe> / (λ + ||grad_train||²)

Where:
- grad_train: Gradient of loss on training sample
- grad_probe: Gradient of loss on probe sample
- λ: Regularization parameter (default: 1e-4)
```

This approximation:
- Avoids O(n³) Hessian inversion
- Uses diagonal approximation for tractability
- Regularization prevents numerical instability

#### Discriminative Layer Selection (AutoMixer)

Only uses embedding and output layers for gradient computation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer Selection Strategy                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Full Model Gradients (Expensive):                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  embed_tokens → layers[0..N] → final_norm → lm_head     │   │
│   │      ↓              ↓              ↓           ↓         │   │
│   │     grad          grad           grad        grad        │   │
│   │                                                          │   │
│   │  Complexity: O(d_model × n_layers × hidden_dim)          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Discriminative Layers Only (Efficient):                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  embed_tokens ─────────────────────────────── lm_head   │   │
│   │      ↓                                           ↓       │   │
│   │     grad                                        grad     │   │
│   │                                                          │   │
│   │  Complexity: O(vocab_size × embed_dim)                   │   │
│   │  Speedup: ~10-100x for typical LLMs                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Weight Sharing Detection:                                      │
│   - Automatically detects tied embed_tokens/lm_head weights     │
│   - Only computes gradient once when weights are shared          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Academic Context

| Method | Venue | Key Contribution |
|--------|-------|------------------|
| [DoReMi](https://arxiv.org/abs/2305.10429) | NeurIPS 2023 | Proxy model + Group DRO for domain weights |
| [DataInf](https://arxiv.org/abs/2310.00902) | ICLR 2024 | Efficient influence without Hessian inversion |
| [Data Mixing Laws](https://arxiv.org/abs/2403.16952) | 2024 | Scaling law interpolation for mixtures |
| [AutoMixer](https://arxiv.org/abs/2408.00417) | ACL 2025 | Discriminative layer selection |
| [MobileLLM-R1](https://arxiv.org/abs/2509.24945) | Sept 2025 | Cross-domain influence + two-stage curriculum |
| [TiKMiX](https://arxiv.org/abs/2508.17677) | Aug 2025 | "Group Influence" with gradient accumulation |

---

### 2. Data Pipeline (`src/cheapertraining/data/`)

Provides dynamic data mixing with influence-based weight updates.

#### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Data Pipeline                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   DatasetMixture (dataclass)                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  name: str           │  Dataset identifier                          │    │
│   │  weight: float       │  Initial sampling weight                     │    │
│   │  path: str           │  HuggingFace path or local path              │    │
│   │  subset: str?        │  Dataset subset/config                       │    │
│   │  split: str          │  Dataset split (default: "train")            │    │
│   │  text_column: str    │  Column containing text (default: "text")    │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│   MixedDataset (IterableDataset)                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  Core Features:                                                      │    │
│   │  • Weighted sampling from multiple HuggingFace datasets             │    │
│   │  • Streaming mode for large datasets (recommended)                   │    │
│   │  • Distributed sharding (rank/world_size aware)                     │    │
│   │  • Shuffle buffer for streaming randomization                        │    │
│   │                                                                      │    │
│   │  Dynamic Weight Updates:                                             │    │
│   │  • update_weights_from_influence(weights: dict)                     │    │
│   │  • get_current_weights() -> dict                                    │    │
│   │  • Weights can be updated during training based on influence        │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│   PackedDataset (IterableDataset)                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  • Concatenates documents into fixed-length chunks                  │    │
│   │  • Batched tokenization for performance (256 texts at once)         │    │
│   │  • Configurable separator token between documents                    │    │
│   │  • Eliminates padding waste for efficiency                          │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Data Flow Diagram                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   HuggingFace Datasets                                                       │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│   │  Code    │  │   Math   │  │  Wiki    │  │  Books   │                   │
│   │  w=0.3   │  │  w=0.2   │  │  w=0.25  │  │  w=0.25  │                   │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│        │             │             │             │                          │
│        └──────────┬──┴─────────────┴──┬──────────┘                          │
│                   │                   │                                      │
│                   ▼                   ▼                                      │
│         ┌─────────────────────────────────────┐                             │
│         │           MixedDataset              │                             │
│         │  • Multinomial sampling by weight   │                             │
│         │  • Stream from each source          │◀─── Influence Calculator   │
│         │  • Dynamic weight updates           │     updates weights         │
│         └─────────────────┬───────────────────┘                             │
│                           │                                                  │
│                           ▼                                                  │
│         ┌─────────────────────────────────────┐                             │
│         │          PackedDataset              │                             │
│         │  • Concatenate texts with <SEP>     │                             │
│         │  • Batch tokenization (256 texts)   │                             │
│         │  • Yield fixed-length chunks        │                             │
│         └─────────────────┬───────────────────┘                             │
│                           │                                                  │
│                           ▼                                                  │
│         ┌─────────────────────────────────────┐                             │
│         │           DataLoader                │                             │
│         │  • num_workers for parallelism      │                             │
│         │  • Worker init for unique seeds     │                             │
│         │  • Prefetching and pin_memory       │                             │
│         └─────────────────────────────────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Training Infrastructure (`src/cheapertraining/training/`)

Provides optimizer factory and influence-aware training utilities.

#### Supported Optimizers

| Optimizer | Memory | Speed | Use Case |
|-----------|--------|-------|----------|
| **Muon** (default) | ~50% of Adam | 2x efficiency | General pretraining |
| AdamW | Baseline | Baseline | Standard training |
| AdamW 8-bit | ~25% of Adam | ~Same | Memory-constrained |
| APOLLO | ~12.5% of Adam | ~Same | Extreme memory constraint |
| APOLLO-Mini | ~0.1% of Adam | ~Same | Maximum memory savings |
| SGD | Minimal | Fast | Fine-tuning |

#### Muon Optimizer Configuration

The Muon optimizer automatically separates parameters:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Muon Parameter Grouping                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Model Parameters                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                      │   │
│   │   Hidden Weights (2D tensors, not embed/head/bias/norm):            │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │  q_proj, k_proj, v_proj, o_proj                              │   │   │
│   │   │  gate_proj, up_proj, down_proj                               │   │   │
│   │   │  → Uses Muon with momentum=0.95, nesterov=True               │   │   │
│   │   │  → Learning rate: lr                                         │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │   Other Parameters (embed, head, bias, norm, 1D):                   │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │  embed_tokens, lm_head                                       │   │   │
│   │   │  all bias terms, all norm weights                           │   │   │
│   │   │  → Uses AdamW with betas=(0.9, 0.95)                        │   │   │
│   │   │  → Learning rate: lr × 0.1 (10% of main LR)                 │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### InfluenceAwareOptimizer

Wraps any optimizer to periodically update dataset mixture weights:

```python
┌─────────────────────────────────────────────────────────────────────────────┐
│                       InfluenceAwareOptimizer                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Initialization:                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  optimizer: Optimizer          # Base optimizer (Muon, AdamW, etc)  │   │
│   │  mixture_calculator: MixtureWeightCalculator                        │   │
│   │  mixed_dataset: MixedDataset   # Dataset to update weights for      │   │
│   │  update_interval: int = 1000   # Steps between weight updates       │   │
│   │  learning_rate: float = 0.2    # Weight update step size            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   step() Flow:                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  1. Call base optimizer.step()                                       │   │
│   │  2. Increment step_count                                             │   │
│   │  3. If step_count % update_interval == 0:                           │   │
│   │     a. Refresh probe gradients with current model                   │   │
│   │     b. Compute new optimal weights                                  │   │
│   │     c. Interpolate: new = (1-lr)*old + lr*optimal                   │   │
│   │     d. Update MixedDataset weights                                  │   │
│   │     e. Log weight changes                                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Basic Influence Calculation

```python
from cheapertraining import (
    InfluenceConfig,
    DiscriminativeGradientExtractor,
    DataInfCalculator,
    create_influence_calculator,
)

# Create influence calculator from model
calculator = create_influence_calculator(model)

# Cache probe set gradients (do this once)
calculator.cache_probe_gradients(probe_dataloader)

# Compute influence of training samples on probe set
influences = calculator.compute_batch_influence_aggregated(train_batch)
# Returns: tensor of shape [batch_size] with influence scores
```

### Mixture Weight Optimization

```python
from cheapertraining import MixtureWeightCalculator, create_mixture_calculator

# Create calculator
mixture_calc = create_mixture_calculator(model, probe_dataloader)

# Compute optimal weights for each dataset
dataset_loaders = {
    "code": code_dataloader,
    "math": math_dataloader,
    "wiki": wiki_dataloader,
}
weights = mixture_calc.compute_mixture_weights(dataset_loaders)
# Returns: {"code": 0.35, "math": 0.25, "wiki": 0.40}
```

### Dynamic Data Mixing

```python
from cheapertraining import MixedDataset, DatasetMixture, create_mixed_dataloader

# Define mixture
mixtures = [
    DatasetMixture(name="code", weight=0.3, path="bigcode/starcoderdata"),
    DatasetMixture(name="math", weight=0.2, path="EleutherAI/proof-pile-2"),
    DatasetMixture(name="wiki", weight=0.5, path="wikipedia", subset="20231101.en"),
]

# Create dataset
dataset = MixedDataset(mixtures, streaming=True, rank=0, world_size=1)

# Update weights dynamically during training
dataset.update_weights_from_influence({"code": 0.4, "math": 0.3, "wiki": 0.3})
```

### Full Training Loop with Influence

```python
from cheapertraining import (
    InfluenceAwareOptimizer,
    MixtureWeightCalculator,
    MixedDataset,
    create_optimizer,
)

# Create base optimizer
base_optimizer = create_optimizer(model, optimizer_type="muon", learning_rate=4e-3)

# Create mixture calculator
mixture_calc = MixtureWeightCalculator(model, probe_dataloader)
mixture_calc.cache_probe_gradients()

# Wrap with influence-aware optimizer
optimizer = InfluenceAwareOptimizer(
    optimizer=base_optimizer,
    mixture_calculator=mixture_calc,
    mixed_dataset=train_dataset,
    update_interval=1000,  # Update weights every 1000 steps
    learning_rate=0.2,     # Move 20% toward optimal weights each update
)

# Training loop
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()  # Automatically updates mixture weights periodically
    optimizer.zero_grad()
```

---

## Configuration

### Influence Configuration

```python
from cheapertraining import InfluenceConfig, InfluenceTarget

config = InfluenceConfig(
    # Layer selection: EMBEDDING_ONLY, OUTPUT_ONLY, or EMBEDDING_AND_OUTPUT
    target_layers=InfluenceTarget.EMBEDDING_AND_OUTPUT,

    # DataInf regularization (prevents division by zero)
    lambda_reg=1e-4,

    # Computation settings
    batch_size=32,
    use_fp16=True,

    # Gradient clipping during extraction
    max_grad_norm=1.0,
)
```

### Mixture Optimization Configuration

```python
from cheapertraining.influence import MixtureOptimizationConfig

config = MixtureOptimizationConfig(
    # Samples to evaluate from each dataset
    samples_per_dataset=1000,

    # Weight constraints
    min_weight=0.01,    # At least 1% from each dataset
    max_weight=0.90,    # At most 90% from any dataset
    normalize_weights=True,

    # Update frequency
    weight_update_interval=10000,

    # Smoothing (EMA)
    influence_smoothing=0.1,
)
```

### Probe Set Configuration

```python
from cheapertraining.influence import ProbeSetConfig

config = ProbeSetConfig(
    probe_set_size=10000,

    # Quality filtering (MobileLLM-R1 Phase I)
    fineweb_edu_min_score=4.0,
    ask_llm_top_fraction=0.10,

    # Deduplication
    dedup_similarity_threshold=0.85,
    dedup_method="minhash",

    # Domain balance
    domains=["code", "math", "knowledge"],
    samples_per_domain=3333,
)
```

---

## Integration with WrinkleFree-1.58Quant

CheaperTraining provides the data selection foundation:

```
┌─────────────────────┐      ┌─────────────────────┐
│  CheaperTraining    │      │ WrinkleFree-1.58    │
│                     │      │      Quant          │
│  • Influence calc   │─────▶│  • Full training    │
│  • Mixture weights  │      │  • Quantization     │
│  • Data mixing      │      │  • Model export     │
└─────────────────────┘      └─────────────────────┘
```

---

## Design Principles

1. **Focused Scope**: Influence functions and data mixing only - not a full training framework
2. **Efficiency First**: Discriminative layer selection for 10-100x speedup
3. **Dynamic Adaptation**: Weights update during training based on model state
4. **Research-Backed**: Implements DataInf, AutoMixer, and MobileLLM-R1 methodology
5. **Composable**: Works with any PyTorch optimizer and training loop

---

## References

- **DataInf**: [ICLR 2024](https://arxiv.org/abs/2310.00902) - Efficient influence without Hessian inversion
- **AutoMixer**: [ACL 2025](https://arxiv.org/abs/2408.00417) - Discriminative layer selection
- **MobileLLM-R1**: [arXiv:2509.24945](https://arxiv.org/abs/2509.24945) - Cross-domain influence methodology
- **DoReMi**: [NeurIPS 2023](https://arxiv.org/abs/2305.10429) - Domain reweighting with minimax optimization
- **TiKMiX**: [arXiv:2508.17677](https://arxiv.org/abs/2508.17677) - Dynamic data mixing with Group Influence
- **Muon**: [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) - Momentum Orthogonalized by Newton-Schulz
- **APOLLO**: [arXiv:2412.05270](https://arxiv.org/abs/2412.05270) - Memory-efficient optimizer
