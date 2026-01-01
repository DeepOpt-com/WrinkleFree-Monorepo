"""BitNet 1.58-bit quantization kernels for sgl-kernel.

Provides Python bindings for BitNet CPU kernels:
- bitnet_gemv: GEMV with packed 2-bit weights and INT8 activations
- bitnet_gemm: Batched GEMM with cache-optimized tiling
- quantize_activations: FP32 -> INT8 activation quantization

Uses torch extension for native kernel calls.
"""

import torch
from typing import Tuple

# Block size constant (must match C++ QK_I2_S)
QK_I2_S = 128

# Kernel availability flag
_kernel_available = False


def _check_kernel_available() -> bool:
    """Check if BitNet kernels are available via torch ops."""
    global _kernel_available
    try:
        import sgl_kernel
        # Check if the bitnet ops are registered
        if hasattr(torch.ops.sgl_kernel, 'bitnet_gemv_cpu'):
            _kernel_available = True
            return True
    except (ImportError, AttributeError):
        pass
    return False


# Check on import
_check_kernel_available()


def check_kernel_available() -> bool:
    """Check if BitNet kernels are available.

    Returns:
        True if native kernels are available, False otherwise.
    """
    return _kernel_available


def bitnet_check_kernel_available() -> bool:
    """Check if BitNet kernels are available (alias for check_kernel_available)."""
    return _kernel_available


def get_cpu_capabilities() -> str:
    """Get detected CPU SIMD capabilities.

    Returns:
        String describing available SIMD extensions (e.g., "AVX2 AVX512").
    """
    if _kernel_available:
        try:
            return torch.ops.sgl_kernel.bitnet_get_cpu_capabilities()
        except Exception:
            pass
    return "Unknown (kernel not available)"


def _unpack_ternary_weights(packed_weights: torch.Tensor) -> torch.Tensor:
    """Unpack 2-bit packed weights to ternary {-1, 0, +1} values.

    Weight encoding: 00=-1, 01=0, 10=+1
    """
    out_features = packed_weights.shape[0]
    packed_in_features = packed_weights.shape[1]
    in_features = packed_in_features * 4

    # Convert to int for bit operations
    packed = packed_weights.to(torch.int32)

    # Unpack 4 weights per byte
    weights = torch.zeros(out_features, in_features, dtype=torch.float32)

    for i in range(4):
        shift = i * 2
        bits = (packed >> shift) & 0x03  # Extract 2 bits
        # Map: 00 -> -1, 01 -> 0, 10 -> +1
        unpacked = bits.float() - 1.0  # 0->-1, 1->0, 2->+1
        weights[:, i::4] = unpacked

    return weights


def bitnet_gemv(
    packed_weights: torch.Tensor,
    activations: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """BitNet GEMV: y = scale * (W @ x).

    Args:
        packed_weights: Packed 2-bit weights [out_features, in_features/4], dtype=uint8
        activations: INT8 activations [in_features], dtype=int8
        scale: Weight scale factor

    Returns:
        Output tensor [out_features], dtype=float32
    """
    out_features = packed_weights.shape[0]
    in_features = packed_weights.shape[1] * 4

    if in_features % QK_I2_S != 0:
        raise ValueError(
            f"bitnet_gemv: in_features ({in_features}) must be multiple of {QK_I2_S}"
        )

    if activations.shape[0] != in_features:
        raise ValueError(
            f"bitnet_gemv: activations.shape[0] ({activations.shape[0]}) "
            f"!= in_features ({in_features})"
        )

    # Use native kernel if available
    if _kernel_available:
        return torch.ops.sgl_kernel.bitnet_gemv_cpu(packed_weights, activations, scale)

    # Python fallback: unpack weights and compute
    weights = _unpack_ternary_weights(packed_weights)
    activations_f = activations.to(torch.float32)
    output = torch.matmul(weights, activations_f) * scale

    return output


def bitnet_gemm(
    packed_weights: torch.Tensor,
    activations: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """BitNet GEMM: Y = scale * (W @ X).

    Args:
        packed_weights: Packed 2-bit weights [out_features, in_features/4], dtype=uint8
        activations: Batched INT8 activations [batch, in_features], dtype=int8
        scale: Weight scale factor

    Returns:
        Output tensor [batch, out_features], dtype=float32
    """
    out_features = packed_weights.shape[0]
    in_features = packed_weights.shape[1] * 4

    if in_features % QK_I2_S != 0:
        raise ValueError(
            f"bitnet_gemm: in_features ({in_features}) must be multiple of {QK_I2_S}"
        )

    # Use native kernel if available
    if _kernel_available:
        # NOTE: The C++ kernel writes output indexed as output[mm * batch + n]
        # which is [out_features, batch] layout, but tensor was declared [batch, out].
        # The data is essentially transposed. We reshape to interpret correctly then transpose.
        result = torch.ops.sgl_kernel.bitnet_gemm_cpu(packed_weights, activations, scale)
        batch_size = activations.shape[0]
        # Reinterpret as [out_features, batch] then transpose to [batch, out_features]
        return result.view(out_features, batch_size).t().contiguous()

    # Python fallback: unpack weights and compute batched matmul
    weights = _unpack_ternary_weights(packed_weights)  # [out, in]
    activations_f = activations.to(torch.float32)  # [batch, in]
    output = torch.matmul(activations_f, weights.t()) * scale  # [batch, out]

    return output


def quantize_activations_i8(
    activations: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 activations to INT8.

    Args:
        activations: FP32 activations [*, features]

    Returns:
        Tuple of (quantized INT8 tensor, scale tensor [scalar])
    """
    if _kernel_available:
        quantized, scale_tensor = torch.ops.sgl_kernel.bitnet_quantize_activations_cpu(
            activations.float()
        )
        # Return tensor to avoid graph break in torch.compile
        return quantized, scale_tensor

    # Fallback Python implementation
    max_val = activations.abs().max()
    # Use torch.where to avoid Python conditionals that break graphs
    scale = torch.where(max_val < 1e-6, torch.tensor(1.0 / 127.0, device=activations.device), max_val / 127.0)
    quantized = (activations / scale).round().clamp(-128, 127).to(torch.int8)

    return quantized, scale


def auto_tune_tiles(
    M: int,
    N: int,
    K: int,
) -> Tuple[int, int, int]:
    """Auto-tune tile sizes for BitNet GEMM.

    Args:
        M: Output dimension
        N: Batch dimension
        K: Input dimension

    Returns:
        Tuple of (tile_m, tile_n, tile_k) for optimal cache performance.
    """
    # Default tile sizes optimized for L2 cache
    # These are heuristics based on common CPU cache sizes
    tile_m = min(M, 64)
    tile_n = min(N, 64)
    tile_k = min(K, QK_I2_S * 2)  # 256 elements = 2 blocks

    return tile_m, tile_n, tile_k


# Export public API
__all__ = [
    "check_kernel_available",
    "bitnet_check_kernel_available",
    "get_cpu_capabilities",
    "bitnet_gemv",
    "bitnet_gemm",
    "quantize_activations_i8",
    "auto_tune_tiles",
    "QK_I2_S",
]
