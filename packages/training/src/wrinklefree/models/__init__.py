"""Model components for BitNet 1.58-bit models."""

from wrinklefree.models.attention import BitNetAttention, BitNetFlashAttention
from wrinklefree.models.bitlinear import (
    BitLinear,
    BitLinearNoActivationQuant,
    convert_linear_to_bitlinear,
)

# FP8 is experimental - use optional import
try:
    from wrinklefree._experimental.fp8 import (
        FP8BitLinear,
        convert_bitlinear_to_fp8,
    )
except ImportError:
    FP8BitLinear = None  # type: ignore
    convert_bitlinear_to_fp8 = None  # type: ignore

from wrinklefree.models.config import (
    BITNET_CONFIGS,
    BitNetConfig,
    BitNetDistributedConfig,
    BitNetTrainingConfig,
    get_config,
)
from wrinklefree.models.ffn import BitNetFFN, BitNetMLP
from wrinklefree.models.llama import BitNetLlama, BitNetLlamaForSequenceClassification
from wrinklefree.models.subln import RMSNorm, SubLN
from wrinklefree.models.transformer import BitNetDecoderLayer, BitNetTransformer

__all__ = [
    # Core layers
    "BitLinear",
    "BitLinearNoActivationQuant",
    "convert_linear_to_bitlinear",
    # FP8 acceleration (DeepSeek-V3 style)
    "FP8BitLinear",
    "convert_bitlinear_to_fp8",
    "SubLN",
    "RMSNorm",
    # Attention
    "BitNetAttention",
    "BitNetFlashAttention",
    # FFN
    "BitNetFFN",
    "BitNetMLP",
    # Transformer
    "BitNetDecoderLayer",
    "BitNetTransformer",
    # Full models
    "BitNetLlama",
    "BitNetLlamaForSequenceClassification",
    # Configs
    "BitNetConfig",
    "BitNetTrainingConfig",
    "BitNetDistributedConfig",
    "BITNET_CONFIGS",
    "get_config",
]
