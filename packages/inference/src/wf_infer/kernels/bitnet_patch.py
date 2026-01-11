"""BitNet kernel patches for transformers models.

Replaces standard PyTorch linear operations with native SIMD kernels.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import native kernels
try:
    from sgl_kernel.quantization import (
        bitnet_check_kernel_available,
        bitnet_gemm,
        bitnet_gemv,
        bitnet_quantize_activations,
        BITNET_BLOCK_SIZE,
    )
    NATIVE_KERNELS_AVAILABLE = bitnet_check_kernel_available()
except ImportError:
    NATIVE_KERNELS_AVAILABLE = False
    BITNET_BLOCK_SIZE = 128


class BitNetLinearNative(nn.Module):
    """Linear layer using native BitNet SIMD kernels.

    Replaces packed 2-bit weight matmul with optimized AVX2/AVX512 kernels.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store packed weights and scale
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scale", weight_scale)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with native BitNet kernel."""
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_features)
        batch_size = x_flat.shape[0]

        # Quantize activations to INT8
        x_int8, act_scale = bitnet_quantize_activations(x_flat)

        # Combined scale
        scale = self.weight_scale.item() * act_scale

        if batch_size == 1:
            # Use GEMV for single token
            output = bitnet_gemv(self.packed_weight, x_int8.squeeze(0), scale)
            output = output.unsqueeze(0)
        else:
            # Use GEMM for batched
            output = bitnet_gemm(self.packed_weight, x_int8, scale)

        if self.bias is not None:
            output = output + self.bias

        return output.view(*batch_shape, self.out_features)


def pack_ternary_weights(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack ternary weights {-1, 0, +1} into 2-bit packed format.

    Args:
        weight: Float tensor with values in {-1, 0, +1}

    Returns:
        Tuple of (packed_weight uint8, scale float tensor)
    """
    out_features, in_features = weight.shape

    # Ensure in_features is divisible by 4 for packing
    if in_features % 4 != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by 4")

    # Round to ternary: {-1, 0, +1}
    # Encoding: -1 -> 0 (00), 0 -> 1 (01), +1 -> 2 (10)
    ternary = torch.round(weight.float()).clamp(-1, 1).to(torch.int8)
    encoded = (ternary + 1).to(torch.uint8)  # Now 0, 1, 2

    # Pack 4 values per byte
    packed_weight = torch.zeros(out_features, in_features // 4, dtype=torch.uint8)

    for i in range(4):
        packed_weight |= (encoded[:, i::4] << (i * 2))

    # Compute scale (using max abs value for proper scaling)
    scale = weight.abs().max().item()
    if scale < 1e-6:
        scale = 1.0

    return packed_weight, torch.tensor(scale, dtype=torch.float32)


def extract_packed_weight_and_scale(
    linear: nn.Module,
    force_repack: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Extract packed weights and scale from a BitNet linear layer.

    BitNet models store weights in packed uint8 format (4 values per byte).
    If force_repack=True, will attempt to pack float ternary weights.
    """
    # Check for BitNet-style packed weights
    if hasattr(linear, "weight"):
        weight = linear.weight.data

        # Check if weights are already packed (uint8)
        if weight.dtype == torch.uint8:
            packed_weight = weight
            # Look for weight scale
            if hasattr(linear, "weight_scale"):
                weight_scale = linear.weight_scale.data
            elif hasattr(linear, "scale"):
                weight_scale = linear.scale.data
            else:
                # Default scale
                weight_scale = torch.tensor(1.0)
        elif force_repack and weight.numel() > 0:
            # Check if weights are already ternary-ish
            unique_vals = torch.unique(torch.round(weight.float()))
            is_ternary = len(unique_vals) <= 3 and unique_vals.abs().max() <= 1.0

            if is_ternary:
                try:
                    packed_weight, weight_scale = pack_ternary_weights(weight)
                    logger.info(f"Repacked ternary weights for {linear}")
                except Exception as e:
                    logger.warning(f"Failed to repack weights for {linear}: {e}")
                    return None, None, None
            else:
                logger.warning(f"Non-ternary weights found (unique: {unique_vals.tolist()}), skipping {linear}")
                return None, None, None
        else:
            logger.warning(f"Non-packed weights found, skipping native kernel for {linear}")
            return None, None, None
    else:
        return None, None, None

    bias = linear.bias.data if hasattr(linear, "bias") and linear.bias is not None else None

    return packed_weight, weight_scale, bias


def patch_model_with_native_kernels(model: nn.Module) -> int:
    """Replace compatible linear layers with native BitNet kernels.

    Returns number of layers patched.
    """
    if not NATIVE_KERNELS_AVAILABLE:
        logger.warning("Native BitNet kernels not available, skipping patch")
        return 0

    patched = 0

    for name, module in model.named_modules():
        # Look for linear layers with packed weights
        if isinstance(module, nn.Linear) or "Linear" in type(module).__name__:
            packed_weight, weight_scale, bias = extract_packed_weight_and_scale(module)

            if packed_weight is not None:
                # Check if dimensions are compatible
                out_features = packed_weight.shape[0]
                in_features = packed_weight.shape[1] * 4  # 4 weights per byte

                if in_features % BITNET_BLOCK_SIZE == 0:
                    # Create native kernel layer
                    native_layer = BitNetLinearNative(
                        in_features=in_features,
                        out_features=out_features,
                        packed_weight=packed_weight,
                        weight_scale=weight_scale,
                        bias=bias,
                    )

                    # Replace in parent
                    parent_name = ".".join(name.split(".")[:-1])
                    attr_name = name.split(".")[-1]

                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model

                    setattr(parent, attr_name, native_layer)
                    patched += 1
                    logger.debug(f"Patched {name} with native BitNet kernel")

    logger.info(f"Patched {patched} layers with native BitNet kernels")
    return patched


def benchmark_kernels(
    in_features: int = 2048,
    out_features: int = 2048,
    batch_size: int = 1,
    num_iterations: int = 100,
) -> dict:
    """Benchmark native vs torch kernels."""
    import time

    if not NATIVE_KERNELS_AVAILABLE:
        return {"error": "Native kernels not available"}

    # Create test data
    packed_weight = torch.randint(0, 256, (out_features, in_features // 4), dtype=torch.uint8)
    weight_scale = torch.tensor(0.1)
    x = torch.randn(batch_size, in_features)

    # Warm up
    for _ in range(10):
        x_int8, act_scale = bitnet_quantize_activations(x)
        if batch_size == 1:
            _ = bitnet_gemv(packed_weight, x_int8.squeeze(0), weight_scale.item() * act_scale)
        else:
            _ = bitnet_gemm(packed_weight, x_int8, weight_scale.item() * act_scale)

    # Benchmark native kernel
    start = time.perf_counter()
    for _ in range(num_iterations):
        x_int8, act_scale = bitnet_quantize_activations(x)
        if batch_size == 1:
            _ = bitnet_gemv(packed_weight, x_int8.squeeze(0), weight_scale.item() * act_scale)
        else:
            _ = bitnet_gemm(packed_weight, x_int8, weight_scale.item() * act_scale)
    native_time = (time.perf_counter() - start) / num_iterations

    # Benchmark torch fallback
    # Unpack weights for comparison
    unpacked = torch.zeros(out_features, in_features)
    packed = packed_weight.to(torch.int32)
    for i in range(4):
        bits = (packed >> (i * 2)) & 0x03
        unpacked[:, i::4] = bits.float() - 1.0

    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = torch.matmul(x, unpacked.t())
    torch_time = (time.perf_counter() - start) / num_iterations

    return {
        "native_ms": native_time * 1000,
        "torch_ms": torch_time * 1000,
        "speedup": torch_time / native_time,
        "in_features": in_features,
        "out_features": out_features,
        "batch_size": batch_size,
    }
