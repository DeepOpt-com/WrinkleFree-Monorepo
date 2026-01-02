"""Model components for BitNet 1.58-bit models."""

from wrinklefree.models.attention import BitNetAttention, BitNetFlashAttention
from wrinklefree.models.bitlinear import (
    BitLinear,
    BitLinearNoActivationQuant,
    convert_linear_to_bitlinear,
)

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
