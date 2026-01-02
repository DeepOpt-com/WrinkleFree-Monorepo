"""EXPERIMENTAL: These modules are not production-ready.

APIs may change without notice. Do not use in production training runs.

This directory contains:
- moe/: Mixture of Experts components (benchmark/testing only)
- tensor_parallel/: Tensor parallelism utilities (experimental)
"""

import warnings

warnings.warn(
    "Importing from wrinklefree._experimental - these APIs are unstable "
    "and may change without notice.",
    FutureWarning,
    stacklevel=2,
)
