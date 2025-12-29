"""BitNet 1.58-bit ternary quantization for SGLang.

BitNet uses ternary weights {-1, 0, +1} packed as 2-bit values.
This provides ~10x compression over FP16 with minimal accuracy loss.

Weight format:
- 2 bits per weight: 00 = -1, 01 = 0, 10 = +1
- Block size: 128 elements (QK_I2_S)
- Activations quantized to INT8 during forward pass

NO FALLBACK: If SIMD kernels are unavailable, operations will FAIL LOUDLY.

References:
- https://arxiv.org/abs/2402.17764 (The Era of 1-bit LLMs)
- https://github.com/microsoft/BitNet
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Constants
BITNET_BLOCK_SIZE = 128  # QK_I2_S
BITS_PER_WEIGHT = 1.58


class BitNetConfig(QuantizationConfig):
    """Configuration for BitNet 1.58-bit ternary quantization.

    BitNet uses ternary weights {-1, 0, +1} which can be represented in 1.58 bits.
    Weights are packed as 2-bit values (4 weights per byte).

    Attributes:
        block_size: Weight block size (default 128)
        activation_bits: Activation quantization bits (default 8)
    """

    def __init__(
        self,
        block_size: int = BITNET_BLOCK_SIZE,
        activation_bits: int = 8,
    ):
        super().__init__()
        self.block_size = block_size
        self.activation_bits = activation_bits

    def __repr__(self) -> str:
        return (
            f"BitNetConfig(block_size={self.block_size}, "
            f"activation_bits={self.activation_bits})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "bitnet"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # CPU only - no GPU capability required
        # 0 means any capability is acceptable
        return 0

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # No external config files needed

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BitNetConfig":
        block_size = config.get("block_size", BITNET_BLOCK_SIZE)
        activation_bits = config.get("activation_bits", 8)
        return cls(block_size=block_size, activation_bits=activation_bits)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            return BitNetLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class BitNetLinearMethod(LinearMethodBase):
    """Linear method for BitNet 1.58-bit inference.

    Uses ternary weight multiplication via optimized SIMD kernels.
    NO FALLBACK: Will fail loudly if kernels are unavailable.
    """

    def __init__(self, quant_config: BitNetConfig):
        self.quant_config = quant_config
        self._kernel_checked = False
        self._kernel_available = False

    def _check_kernel(self):
        """Check kernel availability once."""
        if self._kernel_checked:
            return

        self._kernel_checked = True

        try:
            from sgl_kernel.quantization.bitnet import check_kernel_available
            self._kernel_available = check_kernel_available()
        except ImportError:
            self._kernel_available = False

        if not self._kernel_available:
            logger.warning(
                "BitNet native kernels NOT available. "
                "Operations will use Python fallback (slower) or FAIL."
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weight parameters for BitNet linear layer."""
        self._check_kernel()

        output_size_per_partition = sum(output_partition_sizes)

        # Validate input size alignment
        if input_size_per_partition % self.quant_config.block_size != 0:
            raise ValueError(
                f"BitNet: input_size_per_partition ({input_size_per_partition}) "
                f"must be divisible by block_size ({self.quant_config.block_size})"
            )

        # Packed weight shape: 4 weights per byte
        packed_input_size = input_size_per_partition // 4
        tensor_shape = (output_size_per_partition, packed_input_size)

        # Packed 2-bit weights
        qweight = Parameter(
            torch.empty(tensor_shape, dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": 4,
                "is_bitnet_weight": True,
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        # Scale factor (per-tensor)
        scale = Parameter(
            torch.ones(1, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(scale, {"is_bitnet_scale": True})
        set_weight_attrs(scale, extra_weight_attrs)
        layer.register_parameter("scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Process weights after loading from checkpoint."""
        # Validate weight format
        qweight = layer.qweight
        if qweight.dtype != torch.uint8:
            raise ValueError(
                f"BitNet: Expected qweight dtype uint8, got {qweight.dtype}. "
                "Model may not be properly quantized for BitNet."
            )

        logger.debug(
            f"BitNet weight loaded: shape={qweight.shape}, "
            f"size_mb={qweight.numel() / 1024 / 1024:.2f}"
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply BitNet linear operation.

        Args:
            layer: Linear layer with qweight and scale parameters
            x: Input tensor [batch, in_features]
            bias: Optional bias tensor

        Returns:
            Output tensor [batch, out_features]

        Raises:
            RuntimeError: If native kernels are unavailable (NO FALLBACK!)
        """
        qweight = layer.qweight
        scale = layer.scale

        out_features = qweight.shape[0]
        in_features = qweight.shape[1] * 4  # 4 weights per byte

        # Try native kernel first
        try:
            from sgl_kernel.quantization.bitnet import bitnet_gemm, quantize_activations_i8

            # Quantize activations to INT8
            x_int8, act_scale = quantize_activations_i8(x)

            # Call native GEMM
            out = bitnet_gemm(qweight, x_int8, scale.item())

            # Apply activation scale
            out = out * act_scale

        except (ImportError, NotImplementedError, RuntimeError) as e:
            # NO FALLBACK: Fail loudly!
            raise RuntimeError(
                f"BitNet apply FAILED: {e}. "
                "Native BitNet kernels are required for inference. "
                "NO FALLBACK is provided. "
                "Please ensure the sgl-kernel library is built with SIMD support."
            ) from e

        if bias is not None:
            out = out + bias

        return out


# Factory function for compatibility
def create_bitnet_config(**kwargs) -> BitNetConfig:
    """Create BitNet configuration."""
    return BitNetConfig(**kwargs)
