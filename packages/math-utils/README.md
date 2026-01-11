# wf-math

Pure mathematical utilities for ML algorithms in the WrinkleFree monorepo.

## Status

> **Note**: Most influence functions have been deprecated in favor of ODM (Online Data Mixing).
> See `training.meta_optimization.odm` for the O(1) complexity alternative.

## Installation

```bash
# From monorepo root
uv sync --package wf-math
```

## Usage

```python
import wf_math
# Currently minimal API - legacy functions deprecated
```

## Development

```bash
# Run tests
uv run pytest packages/math-utils/tests/ -v

# Type check
uv run mypy packages/math-utils/src/
```

## Related

- [ODM Paper](https://arxiv.org/abs/2312.02406) - Online Data Mixing (replacement for influence)
- `packages/training/src/wf_train/meta/` - ODM implementation
