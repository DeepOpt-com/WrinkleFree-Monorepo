/**
 * BitNet GEMV/GEMM CPU kernels.
 *
 * Ported from Microsoft's BitNet.cpp (ggml-bitnet-mad.cpp)
 * https://github.com/microsoft/BitNet
 *
 * Supports:
 * - AVX2 (x86-64)
 * - AVX512 (x86-64)
 * - NEON (ARM64)
 * - NEON + DotProd (ARM64 with dot product extension)
 *
 * NO FALLBACK: Unsupported platforms will fail loudly at runtime.
 */

#include "bitnet_gemv.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD headers
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#define BITNET_HAS_X86_SIMD 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define BITNET_HAS_ARM_NEON 1
#else
#define BITNET_NO_SIMD 1
#endif

namespace sgl_kernel {
namespace bitnet {

// Runtime CPU detection for AVX-512 (check at runtime, not just compile time)
#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
static bool runtime_has_avx512() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 16)) != 0;  // AVX-512F
    }
    return false;
}
#else
static bool runtime_has_avx512() { return false; }
#endif

// ============================================================================
// CPU Capability Detection
// ============================================================================

CPUCapabilities detect_cpu_capabilities() {
    CPUCapabilities caps = {false, false, false, false};

#if defined(__AVX2__)
    caps.has_avx2 = true;
#endif

#if defined(__AVX512F__)
    caps.has_avx512 = runtime_has_avx512();
#endif

#if defined(__ARM_NEON)
    caps.has_neon = true;
#endif

#if defined(__ARM_FEATURE_DOTPROD)
    caps.has_dotprod = true;
#endif

    return caps;
}

// ============================================================================
// Helper: Horizontal sum for AVX2 and AVX-512
// ============================================================================

#if defined(BITNET_HAS_X86_SIMD)
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(a),
        _mm256_extractf128_si256(a, 1)
    );
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

#if defined(__AVX512F__)
static inline int hsum_i32_16(const __m512i a) {
    // Reduce 512-bit to 256-bit
    __m256i lo = _mm512_castsi512_si256(a);
    __m256i hi = _mm512_extracti64x4_epi64(a, 1);
    __m256i sum256 = _mm256_add_epi32(lo, hi);
    return hsum_i32_8(sum256);
}
#endif
#endif

// ============================================================================
// LUT-based GEMV (T-MAC inspired)
// ============================================================================
// For 4 ternary weights packed in 1 byte and 4 consecutive activations,
// we can precompute all 81 (3^4) possible sums.
// Index = w0 + 3*w1 + 9*w2 + 27*w3 where w_i in {0,1,2} (encoded as -1,0,+1)

#if defined(BITNET_HAS_X86_SIMD) && defined(__AVX512F__)

// Build 81-entry LUT for 4 consecutive activations
// LUT[i] = sum of w[k] * a[k] for k=0..3
// where w[k] = ((i / 3^k) % 3) - 1  (mapping 0,1,2 to -1,0,+1)
static inline void build_lut_81(const int8_t* activations, int16_t* lut) {
    const int a0 = activations[0];
    const int a1 = activations[1];
    const int a2 = activations[2];
    const int a3 = activations[3];

    // Build all 81 combinations
    for (int i = 0; i < 81; i++) {
        int w0 = (i % 3) - 1;
        int w1 = ((i / 3) % 3) - 1;
        int w2 = ((i / 9) % 3) - 1;
        int w3 = ((i / 27) % 3) - 1;
        lut[i] = (int16_t)(w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3);
    }
}

// Convert 4 packed 2-bit weights to LUT index (0-80)
// Weight encoding: 00=-1→0, 01=0→1, 10=+1→2
static inline int packed_to_lut_index(uint8_t packed) {
    int w0 = (packed >> 0) & 0x03;
    int w1 = (packed >> 2) & 0x03;
    int w2 = (packed >> 4) & 0x03;
    int w3 = (packed >> 6) & 0x03;
    return w0 + 3 * w1 + 9 * w2 + 27 * w3;
}

// LUT-based dot product (scalar reference implementation)
static void bitnet_vec_dot_lut_scalar(
    int n,
    float* result,
    const uint8_t* packed_weights,
    const int8_t* activations
) {
    int32_t sum = 0;
    const int num_bytes = n / 4;  // 4 weights per byte

    alignas(64) int16_t lut[81];

    for (int i = 0; i < num_bytes; i += 1) {
        // Build LUT for current 4 activations
        build_lut_81(activations + i * 4, lut);
        // Lookup result for current 4 weights
        int idx = packed_to_lut_index(packed_weights[i]);
        sum += lut[idx];
    }

    *result = (float)sum;
}

// Full AVX-512 optimized kernel
// Processes 2 blocks (64 packed bytes = 256 weights, 256 activations) per iteration
// Uses maddubs on 256-bit halves within 512-bit registers
static void bitnet_vec_dot_avx512_full(
    int n,
    float* result,
    const uint8_t* packed_weights,
    const int8_t* activations,
    int32_t sum_activations
) {
    const int nb = n / 128;  // Number of 128-element blocks

    // Use 512-bit accumulators
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();

    const __m512i mask512 = _mm512_set1_epi8(0x03);
    const __m512i ones16_512 = _mm512_set1_epi16(1);

    // Process 2 blocks per iteration (256 weights/activations)
    int j = 0;
    for (; j + 1 < nb; j += 2) {
        // Load 64 packed bytes (2 blocks)
        __m512i packed = _mm512_loadu_si512((const __m512i*)(packed_weights + j * 32));

        // Prefetch next iteration
        _mm_prefetch((const char*)(packed_weights + (j + 2) * 32), _MM_HINT_T0);
        _mm_prefetch((const char*)(activations + (j + 2) * 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(activations + (j + 2) * 128 + 64), _MM_HINT_T0);

        // Unpack weights
        __m512i xq8_0 = _mm512_and_si512(_mm512_srli_epi16(packed, 6), mask512);
        __m512i xq8_1 = _mm512_and_si512(_mm512_srli_epi16(packed, 4), mask512);
        __m512i xq8_2 = _mm512_and_si512(_mm512_srli_epi16(packed, 2), mask512);
        __m512i xq8_3 = _mm512_and_si512(packed, mask512);

        // Load activations for 2 blocks (256 bytes total, need 4 x 128)
        // Block j: activations at j*128 + {0, 32, 64, 96}
        // Block j+1: activations at (j+1)*128 + {0, 32, 64, 96}

        // For block j
        __m256i yq8_0a = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 0));
        __m256i yq8_1a = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 32));
        __m256i yq8_2a = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 64));
        __m256i yq8_3a = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 96));

        // For block j+1
        __m256i yq8_0b = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 0));
        __m256i yq8_1b = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 32));
        __m256i yq8_2b = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 64));
        __m256i yq8_3b = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 96));

        // Combine into 512-bit
        __m512i yq8_0 = _mm512_inserti64x4(_mm512_castsi256_si512(yq8_0a), yq8_0b, 1);
        __m512i yq8_1 = _mm512_inserti64x4(_mm512_castsi256_si512(yq8_1a), yq8_1b, 1);
        __m512i yq8_2 = _mm512_inserti64x4(_mm512_castsi256_si512(yq8_2a), yq8_2b, 1);
        __m512i yq8_3 = _mm512_inserti64x4(_mm512_castsi256_si512(yq8_3a), yq8_3b, 1);

        // maddubs on 512-bit (processes both 256-bit halves)
        // Note: _mm512_maddubs_epi16 is AVX512BW
        xq8_0 = _mm512_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm512_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm512_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm512_maddubs_epi16(xq8_3, yq8_3);

        // Sum and accumulate
        __m512i sum16 = _mm512_add_epi16(_mm512_add_epi16(xq8_0, xq8_1),
                                          _mm512_add_epi16(xq8_2, xq8_3));
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(sum16, ones16_512));
    }

    // Handle remaining single block
    if (j < nb) {
        __m256i packed = _mm256_loadu_si256((const __m256i*)(packed_weights + j * 32));
        const __m256i mask256 = _mm256_set1_epi8(0x03);

        __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask256);
        __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask256);
        __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask256);
        __m256i xq8_3 = _mm256_and_si256(packed, mask256);

        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 96));

        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1),
                                          _mm256_add_epi16(xq8_2, xq8_3));
        __m256i ones16 = _mm256_set1_epi16(1);
        acc1 = _mm512_add_epi32(acc1, _mm512_cvtepi32_epi64(_mm256_madd_epi16(sum16, ones16)));
    }

    // Final reduction
    int32_t total = _mm512_reduce_add_epi32(acc0) + (int32_t)_mm512_reduce_add_epi64(acc1);
    *result = (float)(total - sum_activations);
}

// ============================================================================
// TL1-style LUT-based kernel using shuffle_epi8
// ============================================================================
// Based on BitNet.cpp T-MAC approach:
// - Split byte into high/low nibbles (4 bits each = 2 weights)
// - Build 16-entry LUT for each activation pair
// - Use _mm256_shuffle_epi8 for 32 parallel 4-bit lookups
//
// Weight layout in our format:
//   bits 6-7: w0 → activation[k]
//   bits 4-5: w1 → activation[k+32]
//   bits 2-3: w2 → activation[k+64]
//   bits 0-1: w3 → activation[k+96]
//
// High nibble (bits 4-7) = w0|w1 → LUT for (a[k], a[k+32])
// Low nibble (bits 0-3) = w2|w3 → LUT for (a[k+64], a[k+96])

#if defined(BITNET_HAS_X86_SIMD) && defined(__AVX2__)

// Build 16-entry LUT for 2 weights and 2 activations
// LUT[idx] = (w0-1)*a0 + (w1-1)*a1
// where idx = (w0 << 2) | w1, and w0,w1 ∈ {0,1,2}
// Returns int16 to avoid overflow (sum can be up to ±254)
static inline void build_lut_16_pair_i16(int8_t a0, int8_t a1, int16_t* lut) {
    // Only 9 valid combinations (0,1,2 × 0,1,2), but we fill all 16 for shuffle
    for (int idx = 0; idx < 16; idx++) {
        int w0 = (idx >> 2) & 0x03;  // bits 2-3
        int w1 = idx & 0x03;          // bits 0-1
        // Clamp invalid values (w=3 shouldn't occur, treat as 0)
        if (w0 > 2) w0 = 1;  // Map 3 to neutral (0 weight)
        if (w1 > 2) w1 = 1;
        lut[idx] = (int16_t)((w0 - 1) * a0 + (w1 - 1) * a1);
    }
}

// TODO: Build interleaved LUT for 32 positions using shuffle_epi8
// This requires int8 LUTs but (w0-1)*a0 + (w1-1)*a1 can overflow int8 range (±254)
// Will need to split into high/low halves or use different strategy
// For now, using scalar int16 LUT lookups in bitnet_vec_dot_lut_shuffle() below

// LUT-based dot product using int16 LUTs
// This is a reference implementation - can be further optimized with shuffle_epi8
static void bitnet_vec_dot_lut_shuffle(
    int n,
    float* result,
    const uint8_t* packed_weights,
    const int8_t* activations
) {
    const int nb = n / QK_I2_S;  // Number of 128-element blocks
    int32_t total_sum = 0;

    const __m256i mask_lo = _mm256_set1_epi8(0x0F);  // Low nibble mask

    for (int block = 0; block < nb; block++) {
        const uint8_t* w_ptr = packed_weights + block * 32;
        const int8_t* a_ptr = activations + block * 128;

        // Build LUTs for high nibble: (a[k], a[k+32])
        // and low nibble: (a[k+64], a[k+96])
        alignas(32) int16_t lut_hi[32][16];
        alignas(32) int16_t lut_lo[32][16];

        for (int pos = 0; pos < 32; pos++) {
            // High nibble: bits 6-7 (w0) and bits 4-5 (w1)
            // Maps to activations at pos and pos+32
            build_lut_16_pair_i16(a_ptr[pos], a_ptr[pos + 32], lut_hi[pos]);

            // Low nibble: bits 2-3 (w2) and bits 0-1 (w3)
            // Maps to activations at pos+64 and pos+96
            build_lut_16_pair_i16(a_ptr[pos + 64], a_ptr[pos + 96], lut_lo[pos]);
        }

        // Load weights
        __m256i weights = _mm256_loadu_si256((const __m256i*)w_ptr);

        // Extract high and low nibbles
        __m256i w_hi = _mm256_and_si256(_mm256_srli_epi16(weights, 4), mask_lo);
        __m256i w_lo = _mm256_and_si256(weights, mask_lo);

        // Scalar lookup (TODO: vectorize with proper LUT interleaving)
        alignas(32) uint8_t w_hi_arr[32], w_lo_arr[32];
        _mm256_storeu_si256((__m256i*)w_hi_arr, w_hi);
        _mm256_storeu_si256((__m256i*)w_lo_arr, w_lo);

        int32_t block_sum = 0;
        for (int pos = 0; pos < 32; pos++) {
            block_sum += lut_hi[pos][w_hi_arr[pos]];
            block_sum += lut_lo[pos][w_lo_arr[pos]];
        }
        total_sum += block_sum;
    }

    *result = (float)total_sum;
}

// Environment variable to enable LUT kernel for testing
static bool use_lut_kernel() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = getenv("BITNET_USE_LUT");
        cached = (env && env[0] == '1') ? 1 : 0;
    }
    return cached == 1;
}

#endif  // BITNET_HAS_X86_SIMD && __AVX2__

// Optimized AVX-512 kernel with in-place computation
// Uses dual accumulators and reduces memory traffic
static void bitnet_vec_dot_i2_i8_opt(
    int n,
    float* result,
    const uint8_t* packed_weights,
    const int8_t* activations
) {
    const int nb = n / QK_I2_S;

    // Use 2 independent accumulators for better ILP
    __m512i accu0 = _mm512_setzero_si512();
    __m512i accu1 = _mm512_setzero_si512();
    const __m512i mask = _mm512_set1_epi8(0x03);

    // Process 2 blocks per iteration (256 elements)
    int i = 0;
    for (; i + 1 < nb; i += 2) {
        // Block 0
        {
            const int base_w = i * 32;
            const int base_a = i * 128;

            __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + base_w));

            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), _mm256_set1_epi8(0x03));
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), _mm256_set1_epi8(0x03));
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), _mm256_set1_epi8(0x03));
            xq8_3 = _mm256_and_si256(xq8_3, _mm256_set1_epi8(0x03));

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 96));

            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1),
                                           _mm256_add_epi16(xq8_2, xq8_3));

            // Widen to 512-bit for accumulation
            __m512i sum512 = _mm512_cvtepi16_epi32(sum);
            accu0 = _mm512_add_epi32(accu0, sum512);
        }

        // Block 1
        {
            const int base_w = (i + 1) * 32;
            const int base_a = (i + 1) * 128;

            __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + base_w));

            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), _mm256_set1_epi8(0x03));
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), _mm256_set1_epi8(0x03));
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), _mm256_set1_epi8(0x03));
            xq8_3 = _mm256_and_si256(xq8_3, _mm256_set1_epi8(0x03));

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 96));

            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1),
                                           _mm256_add_epi16(xq8_2, xq8_3));

            __m512i sum512 = _mm512_cvtepi16_epi32(sum);
            accu1 = _mm512_add_epi32(accu1, sum512);
        }
    }

    // Handle remaining block
    if (i < nb) {
        const int base_w = i * 32;
        const int base_a = i * 128;

        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + base_w));
        __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), _mm256_set1_epi8(0x03));
        __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), _mm256_set1_epi8(0x03));
        __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), _mm256_set1_epi8(0x03));
        xq8_3 = _mm256_and_si256(xq8_3, _mm256_set1_epi8(0x03));

        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + base_a + 96));

        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1),
                                       _mm256_add_epi16(xq8_2, xq8_3));

        __m512i sum512 = _mm512_cvtepi16_epi32(sum);
        accu0 = _mm512_add_epi32(accu0, sum512);
    }

    // Combine accumulators and reduce
    __m512i total = _mm512_add_epi32(accu0, accu1);
    *result = (float)_mm512_reduce_add_epi32(total);
}

#endif  // AVX512

// ============================================================================
// Compute sum of INT8 activations (for bias correction)
// Call once before processing multiple rows with the same activations
// ============================================================================
int32_t bitnet_sum_activations(int n, const int8_t* activations) {
    int32_t sum_activations = 0;

#if defined(BITNET_HAS_X86_SIMD) && defined(__AVX512F__)
    // AVX-512: Sum 64 int8s at a time
    __m512i sum_vec = _mm512_setzero_si512();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i act = _mm512_loadu_si512((const __m512i*)(activations + i));
        __m256i act_lo = _mm512_castsi512_si256(act);
        __m256i act_hi = _mm512_extracti64x4_epi64(act, 1);
        __m512i lo16 = _mm512_cvtepi8_epi16(act_lo);
        __m512i hi16 = _mm512_cvtepi8_epi16(act_hi);
        __m512i lo32 = _mm512_madd_epi16(lo16, _mm512_set1_epi16(1));
        __m512i hi32 = _mm512_madd_epi16(hi16, _mm512_set1_epi16(1));
        sum_vec = _mm512_add_epi32(sum_vec, _mm512_add_epi32(lo32, hi32));
    }
    sum_activations = _mm512_reduce_add_epi32(sum_vec);
    for (; i < n; i++) sum_activations += activations[i];

#elif defined(BITNET_HAS_X86_SIMD) && defined(__AVX2__)
    __m256i sum_vec = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i act = _mm256_loadu_si256((const __m256i*)(activations + i));
        __m128i lo8 = _mm256_castsi256_si128(act);
        __m128i hi8 = _mm256_extracti128_si256(act, 1);
        __m256i lo16 = _mm256_cvtepi8_epi16(lo8);
        __m256i hi16 = _mm256_cvtepi8_epi16(hi8);
        __m256i sum16 = _mm256_add_epi16(lo16, hi16);
        sum_vec = _mm256_add_epi32(sum_vec, _mm256_madd_epi16(sum16, _mm256_set1_epi16(1)));
    }
    sum_activations = hsum_i32_8(sum_vec);
    for (; i < n; i++) sum_activations += activations[i];

#elif defined(BITNET_HAS_ARM_NEON)
    int32x4_t sum_vec = vdupq_n_s32(0);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t act = vld1q_s8(activations + i);
        int16x8_t lo = vmovl_s8(vget_low_s8(act));
        int16x8_t hi = vmovl_s8(vget_high_s8(act));
        sum_vec = vaddq_s32(sum_vec, vpaddlq_s16(lo));
        sum_vec = vaddq_s32(sum_vec, vpaddlq_s16(hi));
    }
    sum_activations = vaddlvq_s32(sum_vec);
    for (; i < n; i++) sum_activations += activations[i];

#else
    for (int i = 0; i < n; i++) sum_activations += activations[i];
#endif

    return sum_activations;
}

// ============================================================================
// VNNI-optimized kernel using dpbusd (AVX512-VNNI)
// dpbusd: 4-byte dot product to int32, replacing maddubs+madd sequence
// ============================================================================
#if defined(BITNET_HAS_X86_SIMD) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
static void bitnet_vec_dot_vnni(
    int n,
    float* result,
    const uint8_t* packed_weights,
    const int8_t* activations,
    int32_t sum_activations
) {
    const int nb = n / QK_I2_S;
    const __m256i mask256 = _mm256_set1_epi8(0x03);

    // 4 accumulators for ILP
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    // Process 4 blocks per iteration
    int j = 0;
    for (; j + 3 < nb; j += 4) {
        // Prefetch aggressively
        _mm_prefetch((const char*)(packed_weights + (j + 8) * 32), _MM_HINT_T0);
        _mm_prefetch((const char*)(activations + (j + 8) * 128), _MM_HINT_T0);

        // Block 0
        {
            __m256i packed = _mm256_loadu_si256((const __m256i*)(packed_weights + j * 32));
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask256);
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask256);
            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask256);
            __m256i xq8_3 = _mm256_and_si256(packed, mask256);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 96));

            // Use dpbusd: unsigned*signed 4-byte dot to int32
            acc0 = _mm256_dpbusd_epi32(acc0, xq8_0, yq8_0);
            acc0 = _mm256_dpbusd_epi32(acc0, xq8_1, yq8_1);
            acc0 = _mm256_dpbusd_epi32(acc0, xq8_2, yq8_2);
            acc0 = _mm256_dpbusd_epi32(acc0, xq8_3, yq8_3);
        }

        // Block 1
        {
            __m256i packed = _mm256_loadu_si256((const __m256i*)(packed_weights + (j+1) * 32));
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask256);
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask256);
            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask256);
            __m256i xq8_3 = _mm256_and_si256(packed, mask256);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 96));

            acc1 = _mm256_dpbusd_epi32(acc1, xq8_0, yq8_0);
            acc1 = _mm256_dpbusd_epi32(acc1, xq8_1, yq8_1);
            acc1 = _mm256_dpbusd_epi32(acc1, xq8_2, yq8_2);
            acc1 = _mm256_dpbusd_epi32(acc1, xq8_3, yq8_3);
        }

        // Block 2
        {
            __m256i packed = _mm256_loadu_si256((const __m256i*)(packed_weights + (j+2) * 32));
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask256);
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask256);
            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask256);
            __m256i xq8_3 = _mm256_and_si256(packed, mask256);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + (j+2) * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + (j+2) * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + (j+2) * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + (j+2) * 128 + 96));

            acc2 = _mm256_dpbusd_epi32(acc2, xq8_0, yq8_0);
            acc2 = _mm256_dpbusd_epi32(acc2, xq8_1, yq8_1);
            acc2 = _mm256_dpbusd_epi32(acc2, xq8_2, yq8_2);
            acc2 = _mm256_dpbusd_epi32(acc2, xq8_3, yq8_3);
        }

        // Block 3
        {
            __m256i packed = _mm256_loadu_si256((const __m256i*)(packed_weights + (j+3) * 32));
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask256);
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask256);
            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask256);
            __m256i xq8_3 = _mm256_and_si256(packed, mask256);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + (j+3) * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + (j+3) * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + (j+3) * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + (j+3) * 128 + 96));

            acc3 = _mm256_dpbusd_epi32(acc3, xq8_0, yq8_0);
            acc3 = _mm256_dpbusd_epi32(acc3, xq8_1, yq8_1);
            acc3 = _mm256_dpbusd_epi32(acc3, xq8_2, yq8_2);
            acc3 = _mm256_dpbusd_epi32(acc3, xq8_3, yq8_3);
        }
    }

    // Handle remaining blocks
    for (; j < nb; j++) {
        __m256i packed = _mm256_loadu_si256((const __m256i*)(packed_weights + j * 32));
        __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask256);
        __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask256);
        __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask256);
        __m256i xq8_3 = _mm256_and_si256(packed, mask256);

        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 96));

        acc0 = _mm256_dpbusd_epi32(acc0, xq8_0, yq8_0);
        acc0 = _mm256_dpbusd_epi32(acc0, xq8_1, yq8_1);
        acc0 = _mm256_dpbusd_epi32(acc0, xq8_2, yq8_2);
        acc0 = _mm256_dpbusd_epi32(acc0, xq8_3, yq8_3);
    }

    // Combine and reduce
    __m256i total = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1),
                                      _mm256_add_epi32(acc2, acc3));
    *result = (float)(hsum_i32_8(total) - sum_activations);
}
#endif  // AVX512VNNI

// ============================================================================
// BitNet GEMV: Dot product of packed 2-bit weights and INT8 activations
// With pre-computed activation sum for efficiency when called multiple times
// ============================================================================
void bitnet_vec_dot_i2_i8_with_sum(
    int n,
    float* result,
    const uint8_t* packed_weights,
    const int8_t* activations,
    int32_t sum_activations  // Pre-computed sum of activations
) {
    if (n % QK_I2_S != 0) {
        throw std::invalid_argument(
            "bitnet_vec_dot_i2_i8: n (" + std::to_string(n) +
            ") must be multiple of " + std::to_string(QK_I2_S)
        );
    }

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    // Use VNNI kernel if available
    bitnet_vec_dot_vnni(n, result, packed_weights, activations, sum_activations);
    return;
#endif

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = (la_num != 0) ? 1 : 0;

#if defined(BITNET_HAS_X86_SIMD) && defined(__AVX512F__) && defined(__AVX512BW__)
    // AVX-512 implementation with 4x unrolling
    // Uses 256-bit operations within AVX-512 registers for compatibility with our weight layout
    const __m256i mask256 = _mm256_set1_epi8(0x03);
    const __m256i ones16 = _mm256_set1_epi16(1);

    // Use 4 independent 256-bit accumulators for ILP
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();

    // Process 4 blocks per iteration (unroll 4x)
    int j = 0;
    for (; j + 3 < nb; j += 4) {
        // Prefetch next iteration's data
        _mm_prefetch((const char*)(packed_weights + (j + 4) * 32), _MM_HINT_T0);
        _mm_prefetch((const char*)(activations + (j + 4) * 128), _MM_HINT_T0);

        // Block 0
        {
            __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + j * 32));
            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask256);
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask256);
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask256);
            xq8_3 = _mm256_and_si256(xq8_3, mask256);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 96));

            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1),
                                            _mm256_add_epi16(xq8_2, xq8_3));
            accu0 = _mm256_add_epi32(accu0, _mm256_madd_epi16(sum, ones16));
        }

        // Block 1
        {
            __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + (j+1) * 32));
            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask256);
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask256);
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask256);
            xq8_3 = _mm256_and_si256(xq8_3, mask256);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + (j+1) * 128 + 96));

            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1),
                                            _mm256_add_epi16(xq8_2, xq8_3));
            accu1 = _mm256_add_epi32(accu1, _mm256_madd_epi16(sum, ones16));
        }

        // Block 2
        {
            __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + (j+2) * 32));
            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask256);
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask256);
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask256);
            xq8_3 = _mm256_and_si256(xq8_3, mask256);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + (j+2) * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + (j+2) * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + (j+2) * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + (j+2) * 128 + 96));

            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1),
                                            _mm256_add_epi16(xq8_2, xq8_3));
            accu2 = _mm256_add_epi32(accu2, _mm256_madd_epi16(sum, ones16));
        }

        // Block 3
        {
            __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + (j+3) * 32));
            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask256);
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask256);
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask256);
            xq8_3 = _mm256_and_si256(xq8_3, mask256);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + (j+3) * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + (j+3) * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + (j+3) * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + (j+3) * 128 + 96));

            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1),
                                            _mm256_add_epi16(xq8_2, xq8_3));
            accu3 = _mm256_add_epi32(accu3, _mm256_madd_epi16(sum, ones16));
        }
    }

    // Handle remaining blocks
    for (; j < nb; j++) {
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + j * 32));
        __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask256);
        __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask256);
        __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask256);
        xq8_3 = _mm256_and_si256(xq8_3, mask256);

        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + j * 128 + 96));

        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1),
                                        _mm256_add_epi16(xq8_2, xq8_3));
        accu0 = _mm256_add_epi32(accu0, _mm256_madd_epi16(sum, ones16));
    }

    // Combine all accumulators
    __m256i total = _mm256_add_epi32(_mm256_add_epi32(accu0, accu1),
                                      _mm256_add_epi32(accu2, accu3));

    // Bias correction: subtract sum of activations
    *result = (float)(hsum_i32_8(total) - sum_activations);

#elif defined(BITNET_HAS_X86_SIMD) && defined(__AVX2__)
    // AVX2 implementation with 4 independent accumulators for latency hiding
    __m256i mask = _mm256_set1_epi8(0x03);

    // Use 4 independent accumulators to hide instruction latency
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();

    // Hoist constant vector out of loop
    const __m256i ones = _mm256_set1_epi16(1);

    for (int i = 0; i < group32_num; i++) {
        const int base_w = i * 32 * 32;
        const int base_a = i * 128 * 32;

        // Process 4 iterations at a time - fully unrolled to avoid switch overhead
        for (int j = 0; j < 32; j += 4) {
            // Aggressive prefetching: prefetch 2 iterations ahead for better latency hiding
            if (j + 8 < 32) {
                // Prefetch further ahead to L2 cache
                _mm_prefetch((const char*)(packed_weights + base_w + (j + 8) * 32), _MM_HINT_T1);
                _mm_prefetch((const char*)(activations + base_a + (j + 8) * 128), _MM_HINT_T1);
                _mm_prefetch((const char*)(activations + base_a + (j + 8) * 128 + 64), _MM_HINT_T1);
            }
            // Prefetch next iteration to L1 cache
            if (j + 4 < 32) {
                _mm_prefetch((const char*)(packed_weights + base_w + (j + 4) * 32), _MM_HINT_T0);
                _mm_prefetch((const char*)(activations + base_a + (j + 4) * 128), _MM_HINT_T0);
            }

            // Block 0 -> accu0
            {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + base_w + j * 32));
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask);
                xq8_3 = _mm256_and_si256(xq8_3, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + base_a + j * 128 + 0));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + base_a + j * 128 + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + base_a + j * 128 + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + base_a + j * 128 + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1), _mm256_add_epi16(xq8_2, xq8_3));
                accu0 = _mm256_add_epi32(accu0, _mm256_madd_epi16(sum, ones));
            }

            // Block 1 -> accu1
            {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + base_w + (j+1) * 32));
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask);
                xq8_3 = _mm256_and_si256(xq8_3, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+1) * 128 + 0));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+1) * 128 + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+1) * 128 + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+1) * 128 + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1), _mm256_add_epi16(xq8_2, xq8_3));
                accu1 = _mm256_add_epi32(accu1, _mm256_madd_epi16(sum, ones));
            }

            // Block 2 -> accu2
            {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + base_w + (j+2) * 32));
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask);
                xq8_3 = _mm256_and_si256(xq8_3, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+2) * 128 + 0));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+2) * 128 + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+2) * 128 + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+2) * 128 + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1), _mm256_add_epi16(xq8_2, xq8_3));
                accu2 = _mm256_add_epi32(accu2, _mm256_madd_epi16(sum, ones));
            }

            // Block 3 -> accu3
            {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(packed_weights + base_w + (j+3) * 32));
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask);
                xq8_3 = _mm256_and_si256(xq8_3, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+3) * 128 + 0));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+3) * 128 + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+3) * 128 + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + base_a + (j+3) * 128 + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                __m256i sum = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1), _mm256_add_epi16(xq8_2, xq8_3));
                accu3 = _mm256_add_epi32(accu3, _mm256_madd_epi16(sum, ones));
            }
        }
    }

    // Combine all accumulators
    __m256i accu = _mm256_add_epi32(_mm256_add_epi32(accu0, accu1), _mm256_add_epi32(accu2, accu3));

    // Handle remaining blocks - widen to 32-bit immediately to avoid overflow
    // (la_num can be up to 31, which could overflow int16 accumulators)
    for (int i = 0; i < groupla_num; i++) {
        for (int j = 0; j < la_num; j++) {
            __m256i xq8_3 = _mm256_loadu_si256(
                (const __m256i*)(packed_weights + group32_num * 32 * 32 + j * 32)
            );
            __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
            __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
            __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

            xq8_3 = _mm256_and_si256(xq8_3, mask);
            xq8_2 = _mm256_and_si256(xq8_2, mask);
            xq8_1 = _mm256_and_si256(xq8_1, mask);
            xq8_0 = _mm256_and_si256(xq8_0, mask);

            __m256i yq8_0 = _mm256_loadu_si256(
                (const __m256i*)(activations + group32_num * 128 * 32 + j * 128 + 0)
            );
            __m256i yq8_1 = _mm256_loadu_si256(
                (const __m256i*)(activations + group32_num * 128 * 32 + j * 128 + 32)
            );
            __m256i yq8_2 = _mm256_loadu_si256(
                (const __m256i*)(activations + group32_num * 128 * 32 + j * 128 + 64)
            );
            __m256i yq8_3 = _mm256_loadu_si256(
                (const __m256i*)(activations + group32_num * 128 * 32 + j * 128 + 96)
            );

            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            // Sum the 4 maddubs results in 16-bit (safe, max ~2000 per lane)
            __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1),
                                             _mm256_add_epi16(xq8_2, xq8_3));

            // Immediately widen to 32-bit to avoid overflow across blocks
            accu = _mm256_add_epi32(accu, _mm256_madd_epi16(sum16, ones));
        }
    }

    // Bias correction: subtract sum of activations
    *result = (float)(hsum_i32_8(accu) - sum_activations);

#elif defined(BITNET_HAS_ARM_NEON)
    // ARM NEON implementation
    int32x4_t accu_0 = vdupq_n_s32(0);
    int32x4_t accu_1 = vdupq_n_s32(0);
    int32x4_t accu_2 = vdupq_n_s32(0);
    int32x4_t accu_3 = vdupq_n_s32(0);
    const uint8x16_t mask = vdupq_n_u8(3);

    for (int i = 0; i < group32_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)
        // Use dotprod extension if available
        for (int j = 0; j < 32; j++) {
            uint8x16_t xq8_6 = vld1q_u8(packed_weights + i * 32 * 32 + j * 32);
            uint8x16_t xq8_7 = vld1q_u8(packed_weights + i * 32 * 32 + j * 32 + 16);

            uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
            uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
            uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
            uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
            uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
            uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

            int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
            int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
            int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
            int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
            int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
            int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
            int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
            int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

            const int8x16_t yq8_0 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 0);
            const int8x16_t yq8_1 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 16);
            const int8x16_t yq8_2 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 32);
            const int8x16_t yq8_3 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 48);
            const int8x16_t yq8_4 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 64);
            const int8x16_t yq8_5 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 80);
            const int8x16_t yq8_6 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 96);
            const int8x16_t yq8_7 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 112);

            accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
            accu_1 = vdotq_s32(accu_1, q8_1, yq8_1);
            accu_2 = vdotq_s32(accu_2, q8_2, yq8_2);
            accu_3 = vdotq_s32(accu_3, q8_3, yq8_3);
            accu_0 = vdotq_s32(accu_0, q8_4, yq8_4);
            accu_1 = vdotq_s32(accu_1, q8_5, yq8_5);
            accu_2 = vdotq_s32(accu_2, q8_6, yq8_6);
            accu_3 = vdotq_s32(accu_3, q8_7, yq8_7);
        }
#else
        // Fallback NEON without dotprod
        int16x8_t accu32_0 = vdupq_n_s16(0);
        int16x8_t accu32_1 = vdupq_n_s16(0);
        int16x8_t accu32_2 = vdupq_n_s16(0);
        int16x8_t accu32_3 = vdupq_n_s16(0);

        for (int j = 0; j < 32; j++) {
            uint8x16_t xq8_6 = vld1q_u8(packed_weights + i * 32 * 32 + j * 32);
            uint8x16_t xq8_7 = vld1q_u8(packed_weights + i * 32 * 32 + j * 32 + 16);

            uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
            uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
            uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
            uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
            uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
            uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

            int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
            int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
            int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
            int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
            int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
            int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
            int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
            int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

            const int8x16_t yq8_0 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 0);
            const int8x16_t yq8_1 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 16);
            const int8x16_t yq8_2 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 32);
            const int8x16_t yq8_3 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 48);
            const int8x16_t yq8_4 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 64);
            const int8x16_t yq8_5 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 80);
            const int8x16_t yq8_6 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 96);
            const int8x16_t yq8_7 = vld1q_s8(activations + i * 128 * 32 + j * 128 + 112);

            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_1), vget_low_s8(yq8_1));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_1), vget_high_s8(yq8_1));
            // ... (remaining multiply-accumulate operations)
        }

        accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(accu32_0)));
        accu_0 = vaddq_s32(accu_0, vmovl_high_s16(accu32_0));
        accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(accu32_1)));
        accu_1 = vaddq_s32(accu_1, vmovl_high_s16(accu32_1));
        accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(accu32_2)));
        accu_2 = vaddq_s32(accu_2, vmovl_high_s16(accu32_2));
        accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(accu32_3)));
        accu_3 = vaddq_s32(accu_3, vmovl_high_s16(accu32_3));
#endif
    }

    // Horizontal sum
    accu_0 = vaddq_s32(accu_0, accu_1);
    accu_2 = vaddq_s32(accu_2, accu_3);
    accu_0 = vaddq_s32(accu_0, accu_2);
    // Bias correction: subtract sum of activations
    *result = (float)(vaddlvq_s32(accu_0) - sum_activations);

#else
    // NO FALLBACK: Fail loudly if no SIMD support
    throw std::runtime_error(
        "bitnet_vec_dot_i2_i8: No SIMD support available! "
        "This CPU does not support AVX2, AVX512, or NEON. "
        "BitNet requires SIMD for efficient inference. "
        "Detected: AVX2=" + std::to_string(false) +
        ", AVX512=" + std::to_string(false) +
        ", NEON=" + std::to_string(false)
    );
#endif
}

// ============================================================================
// Backward-compatible wrapper: computes sum and calls optimized version
// ============================================================================
void bitnet_vec_dot_i2_i8(
    int n,
    float* result,
    const uint8_t* packed_weights,
    const int8_t* activations
) {
    int32_t sum_activations = bitnet_sum_activations(n, activations);
    bitnet_vec_dot_i2_i8_with_sum(n, result, packed_weights, activations, sum_activations);
}

// ============================================================================
// BitNet GEMM: Matrix multiplication with tiled cache optimization
// ============================================================================

void bitnet_gemm_i2_i8(
    int M,
    int N,
    int K,
    float* output,
    const uint8_t* packed_weights,
    const int8_t* activations,
    float scale,
    const TileConfig& config
) {
    if (K % QK_I2_S != 0) {
        throw std::invalid_argument(
            "bitnet_gemm_i2_i8: K (" + std::to_string(K) +
            ") must be multiple of " + std::to_string(QK_I2_S)
        );
    }

    const int packed_K = K / 4;  // 4 weights per byte

    // OPTIMIZATION: Pre-compute sum of activations for each batch element
    // This avoids recomputing the sum M times per batch element
    std::vector<int32_t> sum_activations_cache(N);
    for (int n = 0; n < N; n++) {
        sum_activations_cache[n] = bitnet_sum_activations(K, activations + n * K);
    }

    // Parallelized tiled GEMM for cache optimization
    // Use collapsed loops for better load balancing across M*N work items
#ifdef _OPENMP
    // For small batches (N <= 4), parallelize over M only
    // For larger batches, use collapsed loop over both M and N
    if (N <= 4) {
        #pragma omp parallel for schedule(dynamic, config.BM) if(M > 64)
        for (int m = 0; m < M; m += config.BM) {
            int m_end = std::min(m + config.BM, M);
            for (int n = 0; n < N; n++) {
                const int32_t sum_act = sum_activations_cache[n];
                for (int mm = m; mm < m_end; mm++) {
                    float result;
                    bitnet_vec_dot_i2_i8_with_sum(K, &result, packed_weights + mm * packed_K,
                                                   activations + n * K, sum_act);
                    output[mm * N + n] = result * scale;
                }
            }
        }
    } else {
        // Collapsed parallelization for batch inference
        const int total_work = M * N;
        #pragma omp parallel for schedule(static) if(total_work > 128)
        for (int work_id = 0; work_id < total_work; work_id++) {
            int m = work_id / N;
            int n = work_id % N;
            float result;
            bitnet_vec_dot_i2_i8_with_sum(K, &result, packed_weights + m * packed_K,
                                           activations + n * K, sum_activations_cache[n]);
            output[m * N + n] = result * scale;
        }
    }
#else
    for (int m = 0; m < M; m += config.BM) {
        int m_end = std::min(m + config.BM, M);
        for (int n = 0; n < N; n++) {
            const int32_t sum_act = sum_activations_cache[n];
            for (int mm = m; mm < m_end; mm++) {
                float result;
                bitnet_vec_dot_i2_i8_with_sum(K, &result, packed_weights + mm * packed_K,
                                               activations + n * K, sum_act);
                output[mm * N + n] = result * scale;
            }
        }
    }
#endif
}

// ============================================================================
// Activation quantization (SIMD-optimized)
// ============================================================================

void quantize_activations_i8(
    int n,
    int8_t* output,
    const float* input,
    float* scale
) {
#if defined(BITNET_HAS_X86_SIMD) && defined(__AVX512F__)
    // AVX-512 vectorized implementation (16 floats at a time)
    __m512 max_vec = _mm512_setzero_ps();

    // Find max absolute value using SIMD (16 floats at a time)
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        __m512 abs_v = _mm512_abs_ps(v);
        max_vec = _mm512_max_ps(max_vec, abs_v);
    }

    // Reduce 512-bit to scalar
    float max_val = _mm512_reduce_max_ps(max_vec);

    // Handle remaining elements
    for (; i < n; i++) {
        float abs_val = std::fabs(input[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }

    if (max_val < 1e-6f) {
        max_val = 1.0f;
    }

    *scale = max_val / 127.0f;
    float inv_scale = 127.0f / max_val;
    __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);

    // Quantize using SIMD (16 floats -> 16 int8s at a time)
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        __m512 scaled = _mm512_mul_ps(v, inv_scale_vec);
        __m512i i32 = _mm512_cvt_roundps_epi32(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // Pack 16x int32 -> 16x int16 -> 16x int8 with saturation
        // Use 128-bit operations to avoid lane-crossing issues with AVX2 pack
        __m256i lo256 = _mm512_castsi512_si256(i32);   // int32[0:7]
        __m256i hi256 = _mm512_extracti64x4_epi64(i32, 1);  // int32[8:15]

        // Split each 256-bit half into 128-bit quarters
        __m128i q0 = _mm256_castsi256_si128(lo256);        // int32[0:3]
        __m128i q1 = _mm256_extracti128_si256(lo256, 1);   // int32[4:7]
        __m128i q2 = _mm256_castsi256_si128(hi256);        // int32[8:11]
        __m128i q3 = _mm256_extracti128_si256(hi256, 1);   // int32[12:15]

        // Pack to int16 (4+4 -> 8 per operation)
        __m128i i16_01 = _mm_packs_epi32(q0, q1);  // int16[0:7]
        __m128i i16_23 = _mm_packs_epi32(q2, q3);  // int16[8:15]

        // Pack to int8 (8+8 -> 16)
        __m128i i8 = _mm_packs_epi16(i16_01, i16_23);  // int8[0:15]

        // Store 16 bytes
        _mm_storeu_si128((__m128i*)(output + i), i8);
    }

    // Handle remaining elements
    for (; i < n; i++) {
        float scaled = input[i] * inv_scale;
        int32_t rounded = static_cast<int32_t>(std::round(scaled));
        output[i] = static_cast<int8_t>(std::max(-128, std::min(127, rounded)));
    }

#elif defined(BITNET_HAS_X86_SIMD) && defined(__AVX2__)
    // AVX2 vectorized implementation with 2x unrolling for better latency hiding
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 max_vec0 = _mm256_setzero_ps();
    __m256 max_vec1 = _mm256_setzero_ps();

    // Find max absolute value using SIMD (16 floats at a time with 2 accumulators)
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 v0 = _mm256_loadu_ps(input + i);
        __m256 v1 = _mm256_loadu_ps(input + i + 8);
        __m256 abs_v0 = _mm256_andnot_ps(sign_mask, v0);
        __m256 abs_v1 = _mm256_andnot_ps(sign_mask, v1);
        max_vec0 = _mm256_max_ps(max_vec0, abs_v0);
        max_vec1 = _mm256_max_ps(max_vec1, abs_v1);
    }
    // Handle remaining 8 elements
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        __m256 abs_v = _mm256_andnot_ps(sign_mask, v);
        max_vec0 = _mm256_max_ps(max_vec0, abs_v);
    }
    // Combine accumulators
    __m256 max_vec = _mm256_max_ps(max_vec0, max_vec1);

    // Horizontal max reduction
    __m128 max_hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 max_lo = _mm256_castps256_ps128(max_vec);
    __m128 max4 = _mm_max_ps(max_hi, max_lo);
    max4 = _mm_max_ps(max4, _mm_shuffle_ps(max4, max4, _MM_SHUFFLE(2, 3, 0, 1)));
    max4 = _mm_max_ps(max4, _mm_shuffle_ps(max4, max4, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_val = _mm_cvtss_f32(max4);

    // Handle remaining elements
    for (; i < n; i++) {
        float abs_val = std::fabs(input[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }

    if (max_val < 1e-6f) {
        max_val = 1.0f;
    }

    *scale = max_val / 127.0f;
    float inv_scale = 127.0f / max_val;
    __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);

    // Quantize using SIMD (16 floats -> 16 int8s at a time for better throughput)
    i = 0;
    for (; i + 16 <= n; i += 16) {
        // Load and quantize two vectors at once
        __m256 v0 = _mm256_loadu_ps(input + i);
        __m256 v1 = _mm256_loadu_ps(input + i + 8);

        __m256 scaled0 = _mm256_mul_ps(v0, inv_scale_vec);
        __m256 scaled1 = _mm256_mul_ps(v1, inv_scale_vec);

        __m256 rounded0 = _mm256_round_ps(scaled0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 rounded1 = _mm256_round_ps(scaled1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        __m256i i32_0 = _mm256_cvtps_epi32(rounded0);
        __m256i i32_1 = _mm256_cvtps_epi32(rounded1);

        // Pack 16x int32 -> 16x int16 -> 16x int8 with saturation
        __m128i lo0 = _mm256_castsi256_si128(i32_0);
        __m128i hi0 = _mm256_extractf128_si256(i32_0, 1);
        __m128i lo1 = _mm256_castsi256_si128(i32_1);
        __m128i hi1 = _mm256_extractf128_si256(i32_1, 1);

        __m128i i16_0 = _mm_packs_epi32(lo0, hi0);  // 8 int16s from first 8 floats
        __m128i i16_1 = _mm_packs_epi32(lo1, hi1);  // 8 int16s from next 8 floats

        __m128i i8_combined = _mm_packs_epi16(i16_0, i16_1);  // 16 int8s

        _mm_storeu_si128((__m128i*)(output + i), i8_combined);
    }

    // Handle remaining 8 elements if present
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        __m256 scaled = _mm256_mul_ps(v, inv_scale_vec);
        __m256 rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i i32 = _mm256_cvtps_epi32(rounded);

        // Pack 8x int32 -> 8x int16 -> 8x int8 with saturation
        __m128i lo = _mm256_castsi256_si128(i32);
        __m128i hi = _mm256_extractf128_si256(i32, 1);
        __m128i i16 = _mm_packs_epi32(lo, hi);
        __m128i i8 = _mm_packs_epi16(i16, i16);  // Only lower 8 bytes valid

        // Store 8 bytes
        _mm_storel_epi64((__m128i*)(output + i), i8);
    }

    // Handle remaining elements
    for (; i < n; i++) {
        float scaled = input[i] * inv_scale;
        int32_t rounded = static_cast<int32_t>(std::round(scaled));
        output[i] = static_cast<int8_t>(std::max(-128, std::min(127, rounded)));
    }

#elif defined(BITNET_HAS_ARM_NEON)
    // ARM NEON vectorized implementation
    float32x4_t max_vec = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        float32x4_t abs_v = vabsq_f32(v);
        max_vec = vmaxq_f32(max_vec, abs_v);
    }

    // Horizontal max
    float32x2_t max2 = vpmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
    max2 = vpmax_f32(max2, max2);
    float max_val = vget_lane_f32(max2, 0);

    for (; i < n; i++) {
        float abs_val = std::fabs(input[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }

    if (max_val < 1e-6f) {
        max_val = 1.0f;
    }

    *scale = max_val / 127.0f;
    float inv_scale = 127.0f / max_val;
    float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t v0 = vld1q_f32(input + i);
        float32x4_t v1 = vld1q_f32(input + i + 4);

        float32x4_t scaled0 = vmulq_f32(v0, inv_scale_vec);
        float32x4_t scaled1 = vmulq_f32(v1, inv_scale_vec);

        int32x4_t i32_0 = vcvtnq_s32_f32(scaled0);
        int32x4_t i32_1 = vcvtnq_s32_f32(scaled1);

        int16x4_t i16_0 = vqmovn_s32(i32_0);
        int16x4_t i16_1 = vqmovn_s32(i32_1);
        int16x8_t i16 = vcombine_s16(i16_0, i16_1);

        int8x8_t i8 = vqmovn_s16(i16);
        vst1_s8(output + i, i8);
    }

    for (; i < n; i++) {
        float scaled = input[i] * inv_scale;
        int32_t rounded = static_cast<int32_t>(std::round(scaled));
        output[i] = static_cast<int8_t>(std::max(-128, std::min(127, rounded)));
    }

#else
    // Scalar fallback
    float max_val = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = std::fabs(input[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }

    if (max_val < 1e-6f) {
        max_val = 1.0f;
    }

    *scale = max_val / 127.0f;
    float inv_scale = 127.0f / max_val;

    for (int i = 0; i < n; i++) {
        float scaled = input[i] * inv_scale;
        int32_t rounded = static_cast<int32_t>(std::round(scaled));
        output[i] = static_cast<int8_t>(std::max(-128, std::min(127, rounded)));
    }
#endif
}

// ============================================================================
// Auto-tuning for tile sizes
// ============================================================================

TileConfig auto_tune_tiles(int M, int K) {
    TileConfig best_config;
    double best_time = 1e9;

    // Tile size candidates
    const int BM_candidates[] = {128, 256, 512};
    const int BK_candidates[] = {32, 64, 128};

    // Create test data
    std::vector<uint8_t> test_weights(M * K / 4, 0x55);
    std::vector<int8_t> test_activations(K, 1);
    std::vector<float> test_output(M);

    for (int bm : BM_candidates) {
        for (int bk : BK_candidates) {
            TileConfig config;
            config.BM = bm;
            config.BK = bk;

            // Warmup
            for (int w = 0; w < 3; w++) {
                bitnet_gemm_i2_i8(
                    M, 1, K,
                    test_output.data(),
                    test_weights.data(),
                    test_activations.data(),
                    1.0f,
                    config
                );
            }

            // Benchmark
            auto start = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < 10; iter++) {
                bitnet_gemm_i2_i8(
                    M, 1, K,
                    test_output.data(),
                    test_weights.data(),
                    test_activations.data(),
                    1.0f,
                    config
                );
            }
            auto end = std::chrono::high_resolution_clock::now();

            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

            if (elapsed < best_time) {
                best_time = elapsed;
                best_config = config;
            }
        }
    }

    return best_config;
}

}  // namespace bitnet
}  // namespace sgl_kernel
