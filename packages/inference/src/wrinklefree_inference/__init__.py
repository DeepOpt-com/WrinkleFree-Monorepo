"""WrinkleFree Inference Engine - SGLang-BitNet serving with native SIMD kernels."""

__version__ = "0.1.0"

# Primary modules (SGLang-BitNet stack)
from wrinklefree_inference.client.bitnet_client import BitNetClient
from wrinklefree_inference.sglang_backend import BitNetConfig, BITNET_QUANT_TYPES

__all__ = ["BitNetClient", "BitNetConfig", "BITNET_QUANT_TYPES", "__version__"]
