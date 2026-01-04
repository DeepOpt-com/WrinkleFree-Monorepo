# Contributing to Data Handler

> Contributor guide for navigating and understanding the data-handler codebase.

## Quick Orientation

### What This Package Does
Shared library providing data loading (streaming, packing, mixing) and influence-based optimization for training data weights.

### Dependencies

| Depends On | What For |
|------------|----------|
| torch | Tensor ops, IterableDataset base |
| datasets | HuggingFace dataset loading |
| transformers | Tokenization |

| Used By | What For |
|---------|----------|
| `training` | MixedDataset, InfluenceTracker for training |
| `eval` | Data loading for evaluation |

---

## Codebase Architecture

### Directory Structure

```
src/data_handler/
├── __init__.py              # Public API exports
├── data/
│   ├── factory.py           # get_loader(), dataloader creation
│   ├── mixing.py            # MixedDataset, DatasetMixture
│   ├── packing.py           # Sequence packing utilities
│   └── config_loader.py     # Load data configs
├── influence/
│   ├── config.py            # InfluenceConfig dataclass
│   ├── tracker.py           # InfluenceTracker callback
│   ├── mixture_calculator.py # MixtureWeightCalculator
│   ├── datainf.py           # DataInfCalculator (main method)
│   ├── gradient.py          # DiscriminativeGradientExtractor
│   ├── landmark.py          # LandmarkCalculator (alternative method)
│   └── distillation.py      # Distillation-based influence
├── training/
│   ├── optimizer.py         # InfluenceAwareOptimizer
│   ├── gradient_clipping.py # Gradient clipping utilities
│   └── qk_clip.py           # QK clipping for attention
└── _legacy/                 # Archived code (do not modify)
```

### Key Abstractions

| Class | File | Purpose |
|-------|------|---------|
| `MixedDataset` | `data/mixing.py` | Weighted sampling from multiple sources with dynamic weight updates |
| `DatasetMixture` | `data/mixing.py` | Config for a single dataset in the mixture |
| `InfluenceTracker` | `influence/tracker.py` | Training callback for periodic weight updates |
| `MixtureWeightCalculator` | `influence/mixture_calculator.py` | Computes optimal dataset weights |
| `DataInfCalculator` | `influence/datainf.py` | DataInf algorithm (no Hessian inversion) |
| `InfluenceConfig` | `influence/config.py` | Configuration for influence calculation |

---

## Code Flow

### Data Loading Flow

```
training/configs/data/mixed_pretrain.yaml
│
├─► config_loader.py: load_data_config()
│   └─► Parse YAML, create DatasetMixture list
│
├─► mixing.py: MixedDataset(mixtures, streaming=True)
│   └─► Lazy-loads each dataset on first iteration
│
└─► factory.py: get_loader(dataset, batch_size, ...)
    └─► Returns DataLoader with proper collation
```

### Influence-Based Remixing Flow

```
training start
│
├─► InfluenceTracker.on_train_begin()
│   └─► Cache probe gradients for target distribution
│
└─► Each step:
    └─► InfluenceTracker.on_step_end(step, loss)
        │
        └─► if step % update_interval == 0:
            │
            ├─► MixtureWeightCalculator.get_weight_update()
            │   └─► DataInfCalculator.compute_influence()
            │       └─► Compute gradient similarity to probes
            │
            └─► MixedDataset.update_weights_from_influence(new_weights)
                └─► Update sampling probabilities
```

### MixedDataset Iteration

```
MixedDataset.__iter__()
│
├─► Sample source index from normalized_weights (torch.multinomial)
│
├─► Yield next item from selected source iterator
│
└─► If source exhausted:
    └─► Re-create iterator (infinite streaming)
```

---

## Entry Points

| Task | Start Here |
|------|------------|
| Add new dataset source | `data/mixing.py:DatasetMixture` |
| Modify sampling logic | `data/mixing.py:MixedDataset.__iter__()` |
| Add new influence method | `influence/base.py`, then implement in new file |
| Change weight update logic | `influence/tracker.py:on_step_end()` |
| Modify gradient extraction | `influence/gradient.py:DiscriminativeGradientExtractor` |

---

## Patterns & Conventions

### Self-Disabling Callback Pattern

InfluenceTracker checks config and no-ops if disabled:
```python
class InfluenceTracker:
    def __init__(self, config, ...):
        self.enabled = config.get("influence", {}).get("enabled", False)

    def on_step_end(self, step, loss):
        if not self.enabled:
            return  # No-op
        # ... actual logic
```

### Dynamic Weight Update Pattern

```python
# MixedDataset supports runtime weight changes
dataset.update_weights_from_influence({
    "fineweb": 0.4,
    "cosmopedia": 0.3,
    "starcoder": 0.3,
})
# Future samples drawn with new weights
```

### Config Cascade Pattern

Influence configs inherit from data configs:
```yaml
# data/mixed_pretrain.yaml
influence:
  enabled: true
  method: datainf
  update_interval: 10000
```

---

## Testing

### Running Tests

```bash
# All tests
uv run --package data-handler pytest packages/data_handler/tests/ -v

# Specific module
uv run --package data-handler pytest packages/data_handler/tests/test_mixing.py -v

# With coverage
uv run --package data-handler pytest packages/data_handler/tests/ --cov=data_handler
```

### Test Organization

| File | What's Tested |
|------|---------------|
| `test_mixing.py` | MixedDataset sampling, weight updates |
| `test_influence.py` | DataInfCalculator, MixtureWeightCalculator |
| `test_factory.py` | DataLoader creation, collation |

---

## Common Tasks

### Adding a New Data Source Type

1. Add new `DatasetMixture` subclass if needed in `data/mixing.py`
2. Or add new fields to existing `DatasetMixture` dataclass
3. Update `MixedDataset._load_dataset()` to handle new type
4. Add config example in `configs/data/`
5. Test with `pytest tests/test_mixing.py`

### Adding a New Influence Method

1. Create `influence/my_method.py`
2. Implement calculator class with `compute_influence(batch, probe_grads)` method
3. Register in `influence/__init__.py`
4. Add to `MixtureWeightCalculator` method selection
5. Add tests in `tests/test_influence.py`

### Modifying Weight Update Schedule

1. Edit `influence/tracker.py:on_step_end()`
2. Adjust `update_interval` or warmup logic
3. Test with training smoke test to verify behavior

---

## Gotchas & Tips

- **Tokenizer Requirement**: `InfluenceTracker` needs a tokenizer to create per-source dataloaders for gradient computation. Pass it in the constructor.

- **Self-Disabling**: When `config.influence.enabled=false`, InfluenceTracker methods are no-ops. This is by design - don't add additional enable checks in callers.

- **Streaming vs Map-Style**: `MixedDataset` is an `IterableDataset` (streaming). Don't try to index it directly.

- **Weight Normalization**: Weights are automatically normalized in `MixedDataset`. You can pass any positive values and they'll be converted to probabilities.

- **Legacy Code**: The `_legacy/` directory contains archived implementations. Don't modify these - they're kept for reference only.

- **Memory**: DataInf caches probe gradients at training start. For large probe sets, this can use significant GPU memory.

- **Test Both Packages**: Changes affect `training` package. Always run training smoke test:
  ```bash
  uv run --package wrinklefree python packages/training/scripts/train_lightning.py \
    model=smollm2_135m training=unified training.max_steps=10
  ```

- **Config Changes**: Data configs are shared between training and eval. Coordinate changes with both packages.
