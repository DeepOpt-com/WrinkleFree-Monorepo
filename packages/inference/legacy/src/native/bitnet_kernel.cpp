/**
 * BitNet 1.58-bit Native GEMV Kernel
 *
 * Fused dequant+matmul for ternary weights {-1, 0, +1}
 * Packed as 2-bit values in uint8 (4 weights per byte)
 *
 * Optimizations:
 * - AVX2/AVX512 SIMD for vectorized operations
 * - LUT-based dequantization (no branches)
 * - Fused operation (no intermediate tensor)
 * - OpenMP parallelization across output rows
 */

#include <torch/extension.h>
#include <immintrin.h>
#include <omp.h>
#include <cstdint>

// Check CPU features
#if defined(__AVX512F__) && defined(__AVX512BW__)
    #define USE_AVX512 1
#elif defined(__AVX2__)
    #define USE_AVX2 1
#endif

// Ternary value mapping (2-bit -> float):
// 00 = -1, 01 = 0, 10 = +1, 11 = 0 (unused)
alignas(32) static const float TERNARY_LUT[4] = {-1.0f, 0.0f, 1.0f, 0.0f};

// 256-entry LUT: each byte -> 4 float weights (for vectorized unpacking)
// Layout: BYTE_LUT[byte_value][0..3] = [w0, w1, w2, w3]
alignas(64) static float BYTE_LUT[256][4];
static bool BYTE_LUT_INITIALIZED = false;

static void init_byte_lut() {
    if (BYTE_LUT_INITIALIZED) return;
    for (int b = 0; b < 256; ++b) {
        BYTE_LUT[b][0] = TERNARY_LUT[(b >> 6) & 0x03];  // bits 7-6
        BYTE_LUT[b][1] = TERNARY_LUT[(b >> 4) & 0x03];  // bits 5-4
        BYTE_LUT[b][2] = TERNARY_LUT[(b >> 2) & 0x03];  // bits 3-2
        BYTE_LUT[b][3] = TERNARY_LUT[b & 0x03];         // bits 1-0
    }
    BYTE_LUT_INITIALIZED = true;
}

/**
 * Baseline scalar implementation (for correctness reference)
 */
void bitnet_gemv_scalar(
    const uint8_t* __restrict__ weights,  // [N, K/4] packed
    const float* __restrict__ input,       // [K]
    float* __restrict__ output,            // [N]
    float scale,                           // weight scale
    int N,                                 // output dim
    int K                                  // input dim (must be multiple of 4)
) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        float acc = 0.0f;
        const uint8_t* w_row = weights + i * (K / 4);

        for (int j = 0; j < K / 4; ++j) {
            uint8_t packed = w_row[j];

            // Unpack 4 weights from byte
            int w0 = (packed >> 6) & 0x03;
            int w1 = (packed >> 4) & 0x03;
            int w2 = (packed >> 2) & 0x03;
            int w3 = packed & 0x03;

            // Lookup and accumulate
            acc += TERNARY_LUT[w0] * input[j * 4 + 0];
            acc += TERNARY_LUT[w1] * input[j * 4 + 1];
            acc += TERNARY_LUT[w2] * input[j * 4 + 2];
            acc += TERNARY_LUT[w3] * input[j * 4 + 3];
        }

        output[i] = acc * scale;
    }
}

#ifdef USE_AVX2
/**
 * AVX2 optimized GEMV with LUT-based dequantization
 * Processes 32 weights per iteration
 */
void bitnet_gemv_avx2(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int N,
    int K
) {
    // Broadcast LUT values
    __m256 v_neg1 = _mm256_set1_ps(-1.0f);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_pos1 = _mm256_set1_ps(1.0f);
    __m256 v_scale = _mm256_set1_ps(scale);

    // Mask for 2-bit extraction
    __m256i v_mask = _mm256_set1_epi8(0x03);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        const uint8_t* w_row = weights + i * (K / 4);
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        // Process 32 weights (8 bytes) per iteration
        for (; j + 8 <= K / 4; j += 8) {
            // Load 8 packed bytes (32 weights)
            __m128i packed = _mm_loadl_epi64((const __m128i*)(w_row + j));
            __m256i packed256 = _mm256_cvtepu8_epi32(packed);

            // Load 32 input values
            __m256 x0 = _mm256_loadu_ps(input + j * 4 + 0);
            __m256 x1 = _mm256_loadu_ps(input + j * 4 + 8);
            __m256 x2 = _mm256_loadu_ps(input + j * 4 + 16);
            __m256 x3 = _mm256_loadu_ps(input + j * 4 + 24);

            // Extract and process weights - unroll for 8 bytes
            for (int b = 0; b < 8; ++b) {
                uint8_t byte_val = w_row[j + b];

                // Extract 4 weights from byte
                int w0 = (byte_val >> 6) & 0x03;
                int w1 = (byte_val >> 4) & 0x03;
                int w2 = (byte_val >> 2) & 0x03;
                int w3 = byte_val & 0x03;

                // Get weight values and multiply
                float wf0 = TERNARY_LUT[w0];
                float wf1 = TERNARY_LUT[w1];
                float wf2 = TERNARY_LUT[w2];
                float wf3 = TERNARY_LUT[w3];

                int base = j * 4 + b * 4;
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(
                    _mm256_set_ps(0, 0, 0, 0, wf3, wf2, wf1, wf0),
                    _mm256_loadu_ps(input + base)
                ));
            }
        }

        // Horizontal sum
        __m256 sum = _mm256_add_ps(acc0, _mm256_add_ps(acc1, _mm256_add_ps(acc2, acc3)));
        __m128 sum_high = _mm256_extractf128_ps(sum, 1);
        __m128 sum_low = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(sum_high, sum_low);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        float result = _mm_cvtss_f32(sum128);

        // Handle remaining elements with scalar
        for (; j < K / 4; ++j) {
            uint8_t packed = w_row[j];
            int w0 = (packed >> 6) & 0x03;
            int w1 = (packed >> 4) & 0x03;
            int w2 = (packed >> 2) & 0x03;
            int w3 = packed & 0x03;

            result += TERNARY_LUT[w0] * input[j * 4 + 0];
            result += TERNARY_LUT[w1] * input[j * 4 + 1];
            result += TERNARY_LUT[w2] * input[j * 4 + 2];
            result += TERNARY_LUT[w3] * input[j * 4 + 3];
        }

        output[i] = result * scale;
    }
}
#endif

#ifdef USE_AVX512
/**
 * AVX512 optimized GEMV with LUT lookup and prefetching
 * Best performing version - uses L1-cached LUT with software prefetch
 */
void bitnet_gemv_avx512(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int N,
    int K
) {
    init_byte_lut();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const uint8_t* w_row = weights + i * (K / 4);

        // Use 4 accumulators for better ILP
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        // Process 16 bytes (64 weights) per iteration
        int j = 0;
        for (; j + 16 <= K / 4; j += 16) {
            // Prefetch next weight bytes and input
            _mm_prefetch((const char*)(w_row + j + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(input + (j + 16) * 4), _MM_HINT_T0);
            _mm_prefetch((const char*)(input + (j + 16) * 4 + 16), _MM_HINT_T0);

            // Load 16 bytes = 64 weights, lookup each byte in LUT
            // Each BYTE_LUT[b] is 4 floats = 16 bytes = 1 __m128

            // Bytes 0-3: 16 weights -> acc0
            __m128 w0 = _mm_load_ps(BYTE_LUT[w_row[j + 0]]);
            __m128 w1 = _mm_load_ps(BYTE_LUT[w_row[j + 1]]);
            __m128 w2 = _mm_load_ps(BYTE_LUT[w_row[j + 2]]);
            __m128 w3 = _mm_load_ps(BYTE_LUT[w_row[j + 3]]);
            __m512 wv0 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(w0), w1, 1), w2, 2), w3, 3);
            __m512 xv0 = _mm512_loadu_ps(input + j * 4);
            acc0 = _mm512_fmadd_ps(wv0, xv0, acc0);

            // Bytes 4-7: 16 weights -> acc1
            __m128 w4 = _mm_load_ps(BYTE_LUT[w_row[j + 4]]);
            __m128 w5 = _mm_load_ps(BYTE_LUT[w_row[j + 5]]);
            __m128 w6 = _mm_load_ps(BYTE_LUT[w_row[j + 6]]);
            __m128 w7 = _mm_load_ps(BYTE_LUT[w_row[j + 7]]);
            __m512 wv1 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(w4), w5, 1), w6, 2), w7, 3);
            __m512 xv1 = _mm512_loadu_ps(input + j * 4 + 16);
            acc1 = _mm512_fmadd_ps(wv1, xv1, acc1);

            // Bytes 8-11: 16 weights -> acc2
            __m128 w8 = _mm_load_ps(BYTE_LUT[w_row[j + 8]]);
            __m128 w9 = _mm_load_ps(BYTE_LUT[w_row[j + 9]]);
            __m128 w10 = _mm_load_ps(BYTE_LUT[w_row[j + 10]]);
            __m128 w11 = _mm_load_ps(BYTE_LUT[w_row[j + 11]]);
            __m512 wv2 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(w8), w9, 1), w10, 2), w11, 3);
            __m512 xv2 = _mm512_loadu_ps(input + j * 4 + 32);
            acc2 = _mm512_fmadd_ps(wv2, xv2, acc2);

            // Bytes 12-15: 16 weights -> acc3
            __m128 w12 = _mm_load_ps(BYTE_LUT[w_row[j + 12]]);
            __m128 w13 = _mm_load_ps(BYTE_LUT[w_row[j + 13]]);
            __m128 w14 = _mm_load_ps(BYTE_LUT[w_row[j + 14]]);
            __m128 w15 = _mm_load_ps(BYTE_LUT[w_row[j + 15]]);
            __m512 wv3 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(w12), w13, 1), w14, 2), w15, 3);
            __m512 xv3 = _mm512_loadu_ps(input + j * 4 + 48);
            acc3 = _mm512_fmadd_ps(wv3, xv3, acc3);
        }

        // Combine accumulators
        __m512 sum = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        float result = _mm512_reduce_add_ps(sum);

        // Handle remaining bytes with LUT lookup
        for (; j < K / 4; ++j) {
            const float* w = BYTE_LUT[w_row[j]];
            result += w[0] * input[j * 4 + 0];
            result += w[1] * input[j * 4 + 1];
            result += w[2] * input[j * 4 + 2];
            result += w[3] * input[j * 4 + 3];
        }

        output[i] = result * scale;
    }
}
#endif

/**
 * Optimized GEMV with LUT-based unpacking
 * Uses precomputed BYTE_LUT for fast weight lookup
 */
void bitnet_gemv_opt(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int N,
    int K
) {
    init_byte_lut();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const uint8_t* w_row = weights + i * (K / 4);

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        // Process 4 bytes (16 weights) per iteration with LUT lookup
        int j = 0;
        for (; j + 4 <= K / 4; j += 4) {
            // Lookup 4 bytes in LUT, each gives 4 weights
            const float* w0 = BYTE_LUT[w_row[j + 0]];
            const float* w1 = BYTE_LUT[w_row[j + 1]];
            const float* w2 = BYTE_LUT[w_row[j + 2]];
            const float* w3 = BYTE_LUT[w_row[j + 3]];

            int base = j * 4;
            acc0 += w0[0] * input[base + 0] + w0[1] * input[base + 1] +
                    w0[2] * input[base + 2] + w0[3] * input[base + 3];
            acc1 += w1[0] * input[base + 4] + w1[1] * input[base + 5] +
                    w1[2] * input[base + 6] + w1[3] * input[base + 7];
            acc2 += w2[0] * input[base + 8] + w2[1] * input[base + 9] +
                    w2[2] * input[base + 10] + w2[3] * input[base + 11];
            acc3 += w3[0] * input[base + 12] + w3[1] * input[base + 13] +
                    w3[2] * input[base + 14] + w3[3] * input[base + 15];
        }

        float acc = acc0 + acc1 + acc2 + acc3;

        // Handle remaining
        for (; j < K / 4; ++j) {
            const float* w = BYTE_LUT[w_row[j]];
            acc += w[0] * input[j * 4 + 0];
            acc += w[1] * input[j * 4 + 1];
            acc += w[2] * input[j * 4 + 2];
            acc += w[3] * input[j * 4 + 3];
        }

        output[i] = acc * scale;
    }
}

/**
 * Best available GEMV implementation
 */
void bitnet_gemv_best(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int N,
    int K
) {
#ifdef USE_AVX512
    bitnet_gemv_avx512(weights, input, output, scale, N, K);
#else
    bitnet_gemv_opt(weights, input, output, scale, N, K);
#endif
}

/**
 * Pre-dequantize weights to float32 tensor for caching
 * This matches the Python approach of materializing weights once
 */
torch::Tensor bitnet_dequant(
    torch::Tensor weights,   // [N, K/4] uint8 packed
    float scale
) {
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(weights.dtype() == torch::kUInt8, "weights must be uint8");

    init_byte_lut();

    int N = weights.size(0);
    int K = weights.size(1) * 4;

    auto output = torch::empty({N, K}, torch::kFloat32);

    const uint8_t* w_ptr = weights.data_ptr<uint8_t>();
    float* out_ptr = output.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const uint8_t* w_row = w_ptr + i * (K / 4);
        float* out_row = out_ptr + i * K;

        for (int j = 0; j < K / 4; ++j) {
            const float* w = BYTE_LUT[w_row[j]];
            out_row[j * 4 + 0] = w[0] * scale;
            out_row[j * 4 + 1] = w[1] * scale;
            out_row[j * 4 + 2] = w[2] * scale;
            out_row[j * 4 + 3] = w[3] * scale;
        }
    }

    return output;
}

/**
 * GEMV using pre-dequantized weights (like Python approach)
 */
torch::Tensor bitnet_gemv_cached(
    torch::Tensor weights,   // [N, K] float32 pre-dequantized
    torch::Tensor input      // [K] float32
) {
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    int N = weights.size(0);
    int K = weights.size(1);

    auto output = torch::empty({N}, torch::kFloat32);

    // Use torch's optimized matmul for cached weights
    return torch::mv(weights, input);
}

/**
 * Python-facing GEMV function
 */
torch::Tensor bitnet_gemv(
    torch::Tensor weights,   // [N, K/4] uint8 packed
    torch::Tensor input,     // [batch, K] or [K] float32
    float scale
) {
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weights.dtype() == torch::kUInt8, "weights must be uint8");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    int N = weights.size(0);
    int K = weights.size(1) * 4;  // 4 weights per byte

    bool batched = input.dim() == 2;
    int batch_size = batched ? input.size(0) : 1;

    TORCH_CHECK(!batched || input.size(1) == K, "input dim mismatch");
    TORCH_CHECK(batched || input.size(0) == K, "input dim mismatch");

    auto output = torch::empty({batch_size, N}, input.options());

    const uint8_t* w_ptr = weights.data_ptr<uint8_t>();
    const float* x_ptr = input.data_ptr<float>();
    float* y_ptr = output.data_ptr<float>();

    for (int b = 0; b < batch_size; ++b) {
        bitnet_gemv_best(
            w_ptr,
            x_ptr + b * K,
            y_ptr + b * N,
            scale,
            N,
            K
        );
    }

    if (!batched) {
        output = output.squeeze(0);
    }

    return output;
}

/**
 * Benchmark different implementations
 */
std::vector<double> bitnet_benchmark(
    torch::Tensor weights,
    torch::Tensor input,
    float scale,
    int iterations
) {
    int N = weights.size(0);
    int K = weights.size(1) * 4;

    auto output = torch::empty({N}, input.options());

    const uint8_t* w_ptr = weights.data_ptr<uint8_t>();
    const float* x_ptr = input.data_ptr<float>();
    float* y_ptr = output.data_ptr<float>();

    std::vector<double> times;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        bitnet_gemv_opt(w_ptr, x_ptr, y_ptr, scale, N, K);
    }

    // Benchmark scalar
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        bitnet_gemv_scalar(w_ptr, x_ptr, y_ptr, scale, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(end - start).count() / iterations);

    // Benchmark optimized
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        bitnet_gemv_opt(w_ptr, x_ptr, y_ptr, scale, N, K);
    }
    end = std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(end - start).count() / iterations);

#ifdef USE_AVX512
    // Benchmark AVX512
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        bitnet_gemv_avx512(w_ptr, x_ptr, y_ptr, scale, N, K);
    }
    end = std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(end - start).count() / iterations);
#endif

    return times;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv", &bitnet_gemv, "BitNet 1.58-bit GEMV (fused dequant+matmul)");
    m.def("dequant", &bitnet_dequant, "Pre-dequantize BitNet weights to float32");
    m.def("gemv_cached", &bitnet_gemv_cached, "GEMV with pre-dequantized weights");
    m.def("benchmark", &bitnet_benchmark, "Benchmark GEMV implementations");
}
