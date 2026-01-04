#include <vector>
#include <type_traits>

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include <cmath>
#include <cstring>

#define QK_I2_S 128
#define QK_I2 128

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#include <immintrin.h>
// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
// horizontally add 16 int32_t (AVX-512)
static inline int hsum_i32_16(const __m512i a) {
    // Reduce 512-bit to 256-bit
    __m256i lo = _mm512_castsi512_si256(a);
    __m256i hi = _mm512_extracti64x4_epi64(a, 1);
    __m256i sum256 = _mm256_add_epi32(lo, hi);
    return hsum_i32_8(sum256);
}
#endif
#elif defined(__loongarch_asx)
// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {

    __m256i tmp1 = __lasx_xvpermi_q(a, a, 0x11);
    __m256i tmp2 = __lasx_xvpermi_q(a, a, 0x00);

    __m128i  tmp1_128 = lasx_extracti128_lo(tmp1);
    __m128i  tmp2_128 = lasx_extracti128_lo(tmp2);

    __m128i sum128 = __lsx_vadd_w(tmp1_128, tmp2_128);

    __m128i ev = __lsx_vpickev_w(sum128, sum128);
    __m128i od = __lsx_vpickod_w(sum128, sum128);
    __m128i sum64 = __lsx_vadd_w(ev, od);

    int sum64_1, sum64_2;
    sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
    sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

    return  sum64_1 + sum64_2;
}
#endif

size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    // 2 bits per weight

    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);

    int n = nrow * n_per_row;

    // f32 -> q8
    double max = 0;
    for (int i = 0; i < n; ++i) {
        max = fmax(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int i=0; i<n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    memset(dst, 0, n * sizeof(uint8_t) / 4);

    // q8 -> 0, 1, 2
    //       |  |  |
    //      -1, 0, 1

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int i = 0; i < n / QK_I2; i++) {
        for (int j = 0; j < QK_I2; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (q8[i * QK_I2 + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;            
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = i2_scale;

    free(q8);

    // 32B for alignment
    return nrow * row_size / 4 + 32;
}

void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    (void)bs; (void)bx; (void)by; (void)nrc;  // unused parameters

#if defined(__AVX512F__) && defined(__AVX512BW__)
    // AVX-512 optimized path for AMD EPYC Genoa / Intel Sapphire Rapids
    // Uses 4 independent 256-bit accumulators for ILP, then combines at the end
    // This avoids the complexity of 512-bit operations on 256-bit data chunks

    __m256i mask = _mm256_set1_epi8(0x03);

    // 4 independent accumulators for latency hiding
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();

    // Hoist constant outside loop
    const __m256i ones = _mm256_set1_epi16(1);

    for (int i = 0; i < group32_num; i++) {
        __m256i inner0 = _mm256_setzero_si256();
        __m256i inner1 = _mm256_setzero_si256();
        __m256i inner2 = _mm256_setzero_si256();
        __m256i inner3 = _mm256_setzero_si256();

        const int base_w = i * 32 * 32;
        const int base_a = i * 128 * 32;

        // Process 32 blocks with 4-way unrolling
        for (int j = 0; j < 32; j += 4) {
            // Software prefetch next iteration's data (improves memory latency)
            if (j + 8 < 32) {
                _mm_prefetch((const char*)(x + base_w + (j + 8) * 32), _MM_HINT_T0);
                _mm_prefetch((const char*)(y + base_a + (j + 8) * 128), _MM_HINT_T0);
                _mm_prefetch((const char*)(y + base_a + (j + 8) * 128 + 64), _MM_HINT_T0);
            }

            // Block j -> inner0
            {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + base_w + j * 32));
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask);
                xq8_3 = _mm256_and_si256(xq8_3, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + base_a + j * 128 + 0));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + base_a + j * 128 + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + base_a + j * 128 + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + base_a + j * 128 + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                inner0 = _mm256_add_epi16(inner0, _mm256_add_epi16(xq8_0, xq8_1));
                inner0 = _mm256_add_epi16(inner0, _mm256_add_epi16(xq8_2, xq8_3));
            }

            // Block j+1 -> inner1
            {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + base_w + (j + 1) * 32));
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask);
                xq8_3 = _mm256_and_si256(xq8_3, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 1) * 128 + 0));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 1) * 128 + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 1) * 128 + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 1) * 128 + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                inner1 = _mm256_add_epi16(inner1, _mm256_add_epi16(xq8_0, xq8_1));
                inner1 = _mm256_add_epi16(inner1, _mm256_add_epi16(xq8_2, xq8_3));
            }

            // Block j+2 -> inner2
            {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + base_w + (j + 2) * 32));
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask);
                xq8_3 = _mm256_and_si256(xq8_3, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 2) * 128 + 0));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 2) * 128 + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 2) * 128 + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 2) * 128 + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                inner2 = _mm256_add_epi16(inner2, _mm256_add_epi16(xq8_0, xq8_1));
                inner2 = _mm256_add_epi16(inner2, _mm256_add_epi16(xq8_2, xq8_3));
            }

            // Block j+3 -> inner3
            {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + base_w + (j + 3) * 32));
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask);
                xq8_3 = _mm256_and_si256(xq8_3, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 3) * 128 + 0));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 3) * 128 + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 3) * 128 + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + base_a + (j + 3) * 128 + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                inner3 = _mm256_add_epi16(inner3, _mm256_add_epi16(xq8_0, xq8_1));
                inner3 = _mm256_add_epi16(inner3, _mm256_add_epi16(xq8_2, xq8_3));
            }
        }

        // Widen to int32 and accumulate
        accu0 = _mm256_add_epi32(accu0, _mm256_madd_epi16(inner0, ones));
        accu1 = _mm256_add_epi32(accu1, _mm256_madd_epi16(inner1, ones));
        accu2 = _mm256_add_epi32(accu2, _mm256_madd_epi16(inner2, ones));
        accu3 = _mm256_add_epi32(accu3, _mm256_madd_epi16(inner3, ones));
    }

    // Handle remaining blocks
    for (int i = 0; i < groupla_num; i++) {
        __m256i accula = _mm256_setzero_si256();

        for (int j = 0; j < la_num; j++) {
            __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + group32_num * 32 * 32 + j * 32));
            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask);
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask);
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask);
            xq8_3 = _mm256_and_si256(xq8_3, mask);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 96));

            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
            accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu0 = _mm256_add_epi32(accu0, _mm256_madd_epi16(accula, ones));
    }

    // Combine all 4 accumulators using AVX-512 for fast reduction
    __m256i combined01 = _mm256_add_epi32(accu0, accu1);
    __m256i combined23 = _mm256_add_epi32(accu2, accu3);
    __m256i combined = _mm256_add_epi32(combined01, combined23);
    int sumi = hsum_i32_8(combined);
    *s = (float)sumi;

#elif defined(__AVX2__)

    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i accu = _mm256_setzero_si256();

    for (int i=0; i < group32_num; i++){
        __m256i accu32 = _mm256_setzero_si256();
        for (int j=0; j < 32; j++) {
        // 128 index
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + i * 32 * 32 + j * 32));
        __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
        __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
        __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

        // each 32 index
        xq8_3 = _mm256_and_si256(xq8_3, mask);
        xq8_2 = _mm256_and_si256(xq8_2, mask);
        xq8_1 = _mm256_and_si256(xq8_1, mask);
        xq8_0 = _mm256_and_si256(xq8_0, mask);

        // each 32 index
        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 96));

        // 128 index accumulation add
        // split into 32 accumulation block
        // each block each 128 index accumulated 4index
        // each index maximum 256
        // each block maximum 4 * 256
        // each block accumulation maximum 127 * 256
        // each 32 group index (128 index in one group) needs cast to int32
        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
        accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, _mm256_set1_epi16(1)), accu);
    }

    for (int i = 0; i < groupla_num; i++){
        __m256i accula = _mm256_setzero_si256();
        for (int j = 0; j < la_num; j++) {
        // 128 index
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + group32_num * 32 * 32 + j * 32));
        __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
        __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
        __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

        // each 32 index
        xq8_3 = _mm256_and_si256(xq8_3, mask);
        xq8_2 = _mm256_and_si256(xq8_2, mask);
        xq8_1 = _mm256_and_si256(xq8_1, mask);
        xq8_0 = _mm256_and_si256(xq8_0, mask);

        // each 32 index
        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 96));

        // 128 index accumulation add
        // split into 32 accumulation block
        // each block each 128 index accumulated 4index
        // each index maximum 256
        // each block maximum 4 * 256
        // each block accumulation maximum 127 * 256
        // each 32 group index (128 index in one group) needs cast to int32
        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
        accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, _mm256_set1_epi16(1)));
    }
    int sumi = hsum_i32_8(accu);
    *s = (float)sumi;

#elif defined(__ARM_NEON)

    int32x4_t accu_0 = vdupq_n_s32(0);
    int32x4_t accu_1 = vdupq_n_s32(0);
    int32x4_t accu_2 = vdupq_n_s32(0);
    int32x4_t accu_3 = vdupq_n_s32(0);
    const uint8x16_t mask = vdupq_n_u8(3);

    for (int i=0; i < group32_num; i++) {

#if defined(__ARM_FEATURE_DOTPROD)

#else
        int16x8_t accu32_0 = vdupq_n_s16(0);
        int16x8_t accu32_1 = vdupq_n_s16(0);
        int16x8_t accu32_2 = vdupq_n_s16(0);
        int16x8_t accu32_3 = vdupq_n_s16(0);
#endif

        for (int j=0; j < 32; j++) {
            uint8x16_t xq8_6 = vld1q_u8(x + i * 32 * 32 + j * 32);
            uint8x16_t xq8_7 = vld1q_u8(x + i * 32 * 32 + j * 32 + 16);
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

            const int8x16_t yq8_0 = vld1q_s8(y + i * 128 * 32 + j * 128 + 0);
            const int8x16_t yq8_1 = vld1q_s8(y + i * 128 * 32 + j * 128 + 16);
            const int8x16_t yq8_2 = vld1q_s8(y + i * 128 * 32 + j * 128 + 32);
            const int8x16_t yq8_3 = vld1q_s8(y + i * 128 * 32 + j * 128 + 48);
            const int8x16_t yq8_4 = vld1q_s8(y + i * 128 * 32 + j * 128 + 64);
            const int8x16_t yq8_5 = vld1q_s8(y + i * 128 * 32 + j * 128 + 80);
            const int8x16_t yq8_6 = vld1q_s8(y + i * 128 * 32 + j * 128 + 96);
            const int8x16_t yq8_7 = vld1q_s8(y + i * 128 * 32 + j * 128 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
            accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
            accu_1 = vdotq_s32(accu_1, q8_1, yq8_1);
            accu_2 = vdotq_s32(accu_2, q8_2, yq8_2);
            accu_3 = vdotq_s32(accu_3, q8_3, yq8_3);
            accu_0 = vdotq_s32(accu_0, q8_4, yq8_4);
            accu_1 = vdotq_s32(accu_1, q8_5, yq8_5);
            accu_2 = vdotq_s32(accu_2, q8_6, yq8_6);
            accu_3 = vdotq_s32(accu_3, q8_7, yq8_7);
#else
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_1), vget_low_s8(yq8_1));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_1), vget_high_s8(yq8_1));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_3), vget_low_s8(yq8_3));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_3), vget_high_s8(yq8_3));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_5), vget_low_s8(yq8_5));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_5), vget_high_s8(yq8_5));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_7), vget_low_s8(yq8_7));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_7), vget_high_s8(yq8_7));
#endif
        }

#if defined(__ARM_FEATURE_DOTPROD)

#else
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

    for (int i = 0; i < groupla_num; i++){
#if defined(__ARM_FEATURE_DOTPROD)

#else
        int16x8_t accula_0 = vdupq_n_s16(0);
        int16x8_t accula_1 = vdupq_n_s16(0);
        int16x8_t accula_2 = vdupq_n_s16(0);
        int16x8_t accula_3 = vdupq_n_s16(0);
#endif
        for (int j = 0; j < la_num; j++) {
            uint8x16_t xq8_6 = vld1q_u8(x + group32_num * 32 * 32 + j * 32);
            uint8x16_t xq8_7 = vld1q_u8(x + group32_num * 32 * 32 + j * 32 + 16);
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

            const int8x16_t yq8_0 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 0);
            const int8x16_t yq8_1 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 16);
            const int8x16_t yq8_2 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 32);
            const int8x16_t yq8_3 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 48);
            const int8x16_t yq8_4 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 64);
            const int8x16_t yq8_5 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 80);
            const int8x16_t yq8_6 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 96);
            const int8x16_t yq8_7 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
            accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
            accu_1 = vdotq_s32(accu_1, q8_1, yq8_1);
            accu_2 = vdotq_s32(accu_2, q8_2, yq8_2);
            accu_3 = vdotq_s32(accu_3, q8_3, yq8_3);
            accu_0 = vdotq_s32(accu_0, q8_4, yq8_4);
            accu_1 = vdotq_s32(accu_1, q8_5, yq8_5);
            accu_2 = vdotq_s32(accu_2, q8_6, yq8_6);
            accu_3 = vdotq_s32(accu_3, q8_7, yq8_7);
#else
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_1), vget_low_s8(yq8_1));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_1), vget_high_s8(yq8_1));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_3), vget_low_s8(yq8_3));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_3), vget_high_s8(yq8_3));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_5), vget_low_s8(yq8_5));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_5), vget_high_s8(yq8_5));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_7), vget_low_s8(yq8_7));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_7), vget_high_s8(yq8_7));
#endif
        }
#if defined(__ARM_FEATURE_DOTPROD)

#else
        accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(accula_0)));
        accu_0 = vaddq_s32(accu_0, vmovl_high_s16(accula_0));
        accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(accula_1)));
        accu_1 = vaddq_s32(accu_1, vmovl_high_s16(accula_1));
        accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(accula_2)));
        accu_2 = vaddq_s32(accu_2, vmovl_high_s16(accula_2));
        accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(accula_3)));
        accu_3 = vaddq_s32(accu_3, vmovl_high_s16(accula_3));
#endif
    }
    accu_0 = vaddq_s32(accu_0, accu_1);
    accu_2 = vaddq_s32(accu_2, accu_3);
    accu_0 = vaddq_s32(accu_0, accu_2);
    int sumi = vaddlvq_s32(accu_0);
    *s = (float)sumi;

#endif
}