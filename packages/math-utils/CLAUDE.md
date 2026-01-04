# math-utils

Pure mathematical utilities for influence functions and gradient computation.

## Overview

This package provides the core mathematical algorithms for:
- **DataInf**: Tractable influence computation without Hessian inversion
- **Influence Distillation**: Landmark-based influence approximation
- **Gradient Extraction**: Discriminative gradient computation from models
- **JVP Embeddings**: Jacobian-vector product extraction
- **Hadamard Transform**: Randomized projection for dimensionality reduction
- **Landmark Selection**: K-means++, farthest point sampling strategies

## Monorepo Integration

This is a **pure math library** with no dependencies on data loading or training:

```
math-utils (this package) - Pure algorithms
    │
    └──► data-handler
            Uses: math_utils.influence for training integration
            Provides: InfluenceTracker, MixtureWeightCalculator
                │
                └──► training (wrinklefree)
                        Uses: data_handler.influence
```

## Quick Start

```python
from math_utils.influence import (
    DataInfCalculator,
    InfluenceConfig,
    DiscriminativeGradientExtractor,
    InfluenceDistillation,
)

# Create gradient extractor
extractor = DiscriminativeGradientExtractor(
    model=model,
    config=InfluenceConfig(lambda_val=0.1, gamma_val=0.1),
)

# Create influence calculator
calculator = DataInfCalculator(
    model=model,
    config=config,
    gradient_extractor=extractor,
)

# Compute influence
influence = calculator.compute_influence(train_loader, probe_loader)
```

## Key Components

| Module | Purpose |
|--------|---------|
| `config.py` | Configuration dataclasses |
| `base.py` | Abstract interfaces (EmbeddingExtractor, InfluenceCalculator) |
| `gradient.py` | DiscriminativeGradientExtractor |
| `datainf.py` | DataInfCalculator (core algorithm) |
| `distillation.py` | InfluenceDistillation (landmark-based) |
| `jvp_embedding.py` | JVPEmbeddingExtractor |
| `hadamard.py` | RandomizedHadamardTransform |
| `landmark.py` | LandmarkSelector strategies |

## Future Use Case: Hyper-Gradient Descent

This package supports future hyper-gradient descent loops that:
- Rebalance data mixture weights dynamically
- Optimize hyperparameters based on validation gradients
- Compute parameter sensitivity via influence functions

## Development

```bash
# Run tests
uv run pytest packages/math-utils/tests/ -v

# Type check
uv run mypy packages/math-utils/src/
```

## Notes

- This package has NO dependencies on data_handler or training
- All algorithms are pure PyTorch operations
- Consumers should import via data_handler.influence for full integration
