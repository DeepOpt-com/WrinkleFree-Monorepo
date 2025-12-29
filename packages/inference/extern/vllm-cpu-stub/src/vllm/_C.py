"""CPU stub for vllm._C (native CUDA extension).

This stub exists to satisfy import checks but does not provide actual functionality.
For CPU-only inference, sgl-kernel should be used instead.
"""

# This module exists to satisfy "import vllm._C" checks
# The actual ops are provided by _custom_ops.py using pure PyTorch

_STUB_WARNING = (
    "vllm._C is a CPU stub - native CUDA operations are not available. "
    "Use sgl-kernel for optimized CPU inference."
)


def __getattr__(name):
    """Provide helpful error for any accessed attribute."""
    raise NotImplementedError(
        f"vllm._C.{name} is not available in CPU stub. "
        "Please ensure sgl-kernel is installed for CPU inference."
    )
