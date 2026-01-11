"""wf-data: Data handling for LLM training.

Active components:
- MixedDataset: Dataset with dynamic mixture weights
- Data loading and streaming utilities

Note: Legacy influence functions have been removed.
Use training.meta_optimization.odm instead (O(1) complexity via EXP3 bandit).
See https://arxiv.org/abs/2312.02406 for the ODM paper.
"""

__version__ = "0.1.0"

from wf_data.data.mixing import MixedDataset

__all__ = [
    "MixedDataset",
]
