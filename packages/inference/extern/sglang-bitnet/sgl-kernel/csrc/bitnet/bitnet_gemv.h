/**
 * BitNet GEMV/GEMM kernels for CPU inference.
 *
 * Ternary weight format: {-1, 0, +1} packed as 2-bit values
 * - 00 = -1
 * - 01 = 0
 * - 10 = +1
 *
 * Block size: 128 elements (QK_I2_S)
 *
 * Tile sizes (tunable for cache optimization):
 * - BM: Output tile size (default 256)
 * - BK: Input tile size (default 32)
 *
 * References:
 * - https://github.com/microsoft/BitNet
 * - https://arxiv.org/abs/2402.17764
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace sgl_kernel {
namespace bitnet {

// Block size for BitNet quantization
constexpr int QK_I2_S = 128;

// Tunable tile sizes (adjust based on CPU cache hierarchy)
struct TileConfig {
    int BM = 256;  // Output tile size
    int BK = 32;   // Input tile size
};

/**
 * Compute sum of INT8 activations (for bias correction).
 * Call once before processing multiple rows with the same activations.
 *
 * @param n Number of elements
 * @param activations INT8 activations
 * @return Sum of all activation values
 */
int32_t bitnet_sum_activations(int n, const int8_t* activations);

/**
 * BitNet GEMV with pre-computed activation sum: y = W * x
 * More efficient when calling multiple times with the same activations.
 *
 * @param n Input dimension (must be multiple of QK_I2_S)
 * @param result Output scalar (dot product result)
 * @param packed_weights Packed 2-bit ternary weights
 * @param activations INT8 activations
 * @param sum_activations Pre-computed sum from bitnet_sum_activations()
 */
void bitnet_vec_dot_i2_i8_with_sum(
    int n,
    float* result,
    const uint8_t* packed_weights,
    const int8_t* activations,
    int32_t sum_activations
);

/**
 * BitNet GEMV: y = W * x
 * Convenience wrapper that computes sum internally.
 *
 * @param n Input dimension (must be multiple of QK_I2_S)
 * @param result Output scalar (dot product result)
 * @param packed_weights Packed 2-bit ternary weights
 * @param activations INT8 activations
 */
void bitnet_vec_dot_i2_i8(
    int n,
    float* result,
    const uint8_t* packed_weights,
    const int8_t* activations
);

/**
 * BitNet GEMM: Y = W * X (batched)
 *
 * @param M Number of output features
 * @param N Batch size
 * @param K Number of input features (must be multiple of QK_I2_S)
 * @param output Output matrix (M x N)
 * @param packed_weights Packed weights (M x K/4)
 * @param activations Input activations (K x N)
 * @param scale Weight scale factor
 * @param config Tile configuration for cache optimization
 */
void bitnet_gemm_i2_i8(
    int M,
    int N,
    int K,
    float* output,
    const uint8_t* packed_weights,
    const int8_t* activations,
    float scale,
    const TileConfig& config = TileConfig()
);

/**
 * Quantize activations to INT8 for BitNet GEMV.
 *
 * @param n Number of elements
 * @param output INT8 output
 * @param input FP32 input
 * @param scale Output scale factor
 */
void quantize_activations_i8(
    int n,
    int8_t* output,
    const float* input,
    float* scale
);

/**
 * Check CPU SIMD capabilities.
 */
struct CPUCapabilities {
    bool has_avx2;
    bool has_avx512;
    bool has_neon;
    bool has_dotprod;  // ARM dot product extension
};

CPUCapabilities detect_cpu_capabilities();

/**
 * Auto-tune tile sizes for the current CPU.
 *
 * @param M Typical M dimension
 * @param K Typical K dimension
 * @return Optimal tile configuration
 */
TileConfig auto_tune_tiles(int M, int K);

}  // namespace bitnet
}  // namespace sgl_kernel
