# math-utils

Pure mathematical utilities for ML algorithms.

## DEPRECATED: Influence Functions

> **All influence code is now in `_legacy/influence/`**. Use `training.meta_optimization.odm` instead (O(1) complexity via EXP3 bandit).
>
> Reference: [arxiv:2312.02406](https://arxiv.org/abs/2312.02406) (ODM paper)

The DataInf-based influence algorithms have been replaced by ODM (Online Data Mixing) which:
- Uses EXP3 multi-armed bandit instead of gradient influence
- Has O(1) complexity instead of O(K)
- Requires no gradient caching or Hessian computation
- Has ~0% overhead vs ~5-10% for DataInf

## Legacy Influence Algorithms (in `_legacy/`)

For reference, the legacy influence code provides:
- **DataInf**: Tractable influence computation without Hessian inversion
- **Influence Distillation**: Landmark-based influence approximation
- **Gradient Extraction**: Discriminative gradient computation from models
- **JVP Embeddings**: Jacobian-vector product extraction
- **Hadamard Transform**: Randomized projection for dimensionality reduction
- **Landmark Selection**: K-means++, farthest point sampling strategies

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
- Legacy influence code emits deprecation warnings when imported
