//! ARM NEON optimized BitNet GEMV.
//!
//! This module provides SIMD-optimized dot product for ARM64 processors.
//! Falls back to scalar implementation on non-ARM platforms.

use super::types::{QK_BLOCK, BLOCK_BYTES};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON-optimized dot product.
///
/// Uses ARM NEON SIMD instructions for fast ternary-weight × int8-activation
/// multiply-accumulate.
///
/// # Algorithm
/// 1. Load 32 packed weight bytes (128 weights)
/// 2. Unpack to 4 groups of 32 2-bit values
/// 3. Load 128 int8 activations
/// 4. Multiply-accumulate using vmlal_s8 (widening multiply-add)
/// 5. Horizontal sum to get final result
///
/// # Arguments
/// * `packed_weights` - Packed I2_S weights
/// * `activations` - INT8 activations
///
/// # Returns
/// Dot product result as i32.
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_neon(packed_weights: &[u8], activations: &[i8]) -> i32 {
    let n = activations.len();
    assert!(n % QK_BLOCK == 0);
    assert!(packed_weights.len() >= n / 4);

    let num_blocks = n / QK_BLOCK;

    unsafe {
        // Accumulators for 4 groups of 32 elements each
        let mut acc0: int32x4_t = vdupq_n_s32(0);
        let mut acc1: int32x4_t = vdupq_n_s32(0);
        let mut acc2: int32x4_t = vdupq_n_s32(0);
        let mut acc3: int32x4_t = vdupq_n_s32(0);

        // Mask for extracting 2-bit values
        let mask = vdupq_n_u8(0x03);
        // Bias for converting 0,1,2 -> -1,0,+1
        let bias = vdupq_n_s8(1);

        for block in 0..num_blocks {
            let w_ptr = packed_weights.as_ptr().add(block * BLOCK_BYTES);
            let a_ptr = activations.as_ptr().add(block * QK_BLOCK);

            // Load 32 packed bytes (128 weights)
            let w_lo: uint8x16_t = vld1q_u8(w_ptr);
            let w_hi: uint8x16_t = vld1q_u8(w_ptr.add(16));

            // Unpack weights to 4 groups of 32 bytes each
            // Group 0: bits 6-7 of each byte
            let w0_lo = vandq_u8(vshrq_n_u8(w_lo, 6), mask);
            let w0_hi = vandq_u8(vshrq_n_u8(w_hi, 6), mask);

            // Group 1: bits 4-5 of each byte
            let w1_lo = vandq_u8(vshrq_n_u8(w_lo, 4), mask);
            let w1_hi = vandq_u8(vshrq_n_u8(w_hi, 4), mask);

            // Group 2: bits 2-3 of each byte
            let w2_lo = vandq_u8(vshrq_n_u8(w_lo, 2), mask);
            let w2_hi = vandq_u8(vshrq_n_u8(w_hi, 2), mask);

            // Group 3: bits 0-1 of each byte
            let w3_lo = vandq_u8(w_lo, mask);
            let w3_hi = vandq_u8(w_hi, mask);

            // Convert from {0,1,2} to {-1,0,+1} by subtracting 1
            let w0_lo_s = vsubq_s8(vreinterpretq_s8_u8(w0_lo), bias);
            let w0_hi_s = vsubq_s8(vreinterpretq_s8_u8(w0_hi), bias);
            let w1_lo_s = vsubq_s8(vreinterpretq_s8_u8(w1_lo), bias);
            let w1_hi_s = vsubq_s8(vreinterpretq_s8_u8(w1_hi), bias);
            let w2_lo_s = vsubq_s8(vreinterpretq_s8_u8(w2_lo), bias);
            let w2_hi_s = vsubq_s8(vreinterpretq_s8_u8(w2_hi), bias);
            let w3_lo_s = vsubq_s8(vreinterpretq_s8_u8(w3_lo), bias);
            let w3_hi_s = vsubq_s8(vreinterpretq_s8_u8(w3_hi), bias);

            // Load activations (128 int8s in 8 vectors)
            let a0_lo: int8x16_t = vld1q_s8(a_ptr.add(0) as *const i8);
            let a0_hi: int8x16_t = vld1q_s8(a_ptr.add(16) as *const i8);
            let a1_lo: int8x16_t = vld1q_s8(a_ptr.add(32) as *const i8);
            let a1_hi: int8x16_t = vld1q_s8(a_ptr.add(48) as *const i8);
            let a2_lo: int8x16_t = vld1q_s8(a_ptr.add(64) as *const i8);
            let a2_hi: int8x16_t = vld1q_s8(a_ptr.add(80) as *const i8);
            let a3_lo: int8x16_t = vld1q_s8(a_ptr.add(96) as *const i8);
            let a3_hi: int8x16_t = vld1q_s8(a_ptr.add(112) as *const i8);

            // Multiply-accumulate with widening
            // vmlal_s8: int16 += int8 * int8 (lower half)
            // vmlal_high_s8: int16 += int8 * int8 (upper half)

            // Group 0: activations[0:32] * weights[0:32]
            let mut sum0: int16x8_t = vdupq_n_s16(0);
            sum0 = vmlal_s8(sum0, vget_low_s8(w0_lo_s), vget_low_s8(a0_lo));
            sum0 = vmlal_high_s8(sum0, w0_lo_s, a0_lo);
            sum0 = vmlal_s8(sum0, vget_low_s8(w0_hi_s), vget_low_s8(a0_hi));
            sum0 = vmlal_high_s8(sum0, w0_hi_s, a0_hi);
            acc0 = vpadalq_s16(acc0, sum0);

            // Group 1: activations[32:64] * weights[32:64]
            let mut sum1: int16x8_t = vdupq_n_s16(0);
            sum1 = vmlal_s8(sum1, vget_low_s8(w1_lo_s), vget_low_s8(a1_lo));
            sum1 = vmlal_high_s8(sum1, w1_lo_s, a1_lo);
            sum1 = vmlal_s8(sum1, vget_low_s8(w1_hi_s), vget_low_s8(a1_hi));
            sum1 = vmlal_high_s8(sum1, w1_hi_s, a1_hi);
            acc1 = vpadalq_s16(acc1, sum1);

            // Group 2: activations[64:96] * weights[64:96]
            let mut sum2: int16x8_t = vdupq_n_s16(0);
            sum2 = vmlal_s8(sum2, vget_low_s8(w2_lo_s), vget_low_s8(a2_lo));
            sum2 = vmlal_high_s8(sum2, w2_lo_s, a2_lo);
            sum2 = vmlal_s8(sum2, vget_low_s8(w2_hi_s), vget_low_s8(a2_hi));
            sum2 = vmlal_high_s8(sum2, w2_hi_s, a2_hi);
            acc2 = vpadalq_s16(acc2, sum2);

            // Group 3: activations[96:128] * weights[96:128]
            let mut sum3: int16x8_t = vdupq_n_s16(0);
            sum3 = vmlal_s8(sum3, vget_low_s8(w3_lo_s), vget_low_s8(a3_lo));
            sum3 = vmlal_high_s8(sum3, w3_lo_s, a3_lo);
            sum3 = vmlal_s8(sum3, vget_low_s8(w3_hi_s), vget_low_s8(a3_hi));
            sum3 = vmlal_high_s8(sum3, w3_hi_s, a3_hi);
            acc3 = vpadalq_s16(acc3, sum3);
        }

        // Horizontal sum of all accumulators
        let total = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        vaddvq_s32(total)
    }
}

// NOTE: The dotprod extension (ARMv8.2+) with vdotq_s32 would be 4x faster,
// but it requires the unstable `stdarch_neon_dotprod` feature in Rust.
// For now, we use the standard NEON path which is still much faster than scalar.
// TODO: Enable dotprod when the feature stabilizes (tracking issue #117224)

/// Fallback for non-ARM platforms - delegates to scalar implementation.
#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_neon(packed_weights: &[u8], activations: &[i8]) -> i32 {
    super::gemv_scalar::vec_dot_scalar(packed_weights, activations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::gemv_scalar::{pack_weights, vec_dot_scalar};

    #[test]
    fn test_neon_matches_scalar_all_ones() {
        let weights = vec![1i8; 128];
        let packed = pack_weights(&weights);
        let activations = vec![1i8; 128];

        let scalar_result = vec_dot_scalar(&packed, &activations);
        let neon_result = vec_dot_neon(&packed, &activations);

        assert_eq!(scalar_result, neon_result);
        assert_eq!(neon_result, 128);
    }

    #[test]
    fn test_neon_matches_scalar_all_minus_ones() {
        let weights = vec![-1i8; 128];
        let packed = pack_weights(&weights);
        let activations = vec![1i8; 128];

        let scalar_result = vec_dot_scalar(&packed, &activations);
        let neon_result = vec_dot_neon(&packed, &activations);

        assert_eq!(scalar_result, neon_result);
        assert_eq!(neon_result, -128);
    }

    #[test]
    fn test_neon_matches_scalar_all_zeros() {
        let weights = vec![0i8; 128];
        let packed = pack_weights(&weights);
        let activations: Vec<i8> = (0..128).map(|i| (i % 127) as i8).collect();

        let scalar_result = vec_dot_scalar(&packed, &activations);
        let neon_result = vec_dot_neon(&packed, &activations);

        assert_eq!(scalar_result, neon_result);
        assert_eq!(neon_result, 0);
    }

    #[test]
    fn test_neon_matches_scalar_random() {
        // Pseudo-random weights and activations
        let weights: Vec<i8> = (0..128).map(|i| ((i * 7 + 3) % 3) as i8 - 1).collect();
        let packed = pack_weights(&weights);
        // Use wrapping arithmetic to avoid overflow
        let activations: Vec<i8> = (0..128).map(|i| (((i * 11 + 5) % 255) as i32 - 127) as i8).collect();

        let scalar_result = vec_dot_scalar(&packed, &activations);
        let neon_result = vec_dot_neon(&packed, &activations);

        assert_eq!(scalar_result, neon_result);
    }

    #[test]
    fn test_neon_matches_scalar_multiple_blocks() {
        // 256 elements = 2 blocks
        let weights: Vec<i8> = (0..256).map(|i| ((i * 7) % 3) as i8 - 1).collect();
        let packed = pack_weights(&weights);
        // Use wrapping arithmetic to avoid overflow
        let activations: Vec<i8> = (0..256).map(|i| (((i * 13) % 255) as i32 - 127) as i8).collect();

        let scalar_result = vec_dot_scalar(&packed, &activations);
        let neon_result = vec_dot_neon(&packed, &activations);

        assert_eq!(scalar_result, neon_result);
    }

    #[test]
    fn test_neon_matches_scalar_large() {
        // 1024 elements = 8 blocks
        let weights: Vec<i8> = (0..1024).map(|i| ((i * 17) % 3) as i8 - 1).collect();
        let packed = pack_weights(&weights);
        // Use wrapping arithmetic to avoid overflow
        let activations: Vec<i8> = (0..1024).map(|i| (((i * 23) % 255) as i32 - 127) as i8).collect();

        let scalar_result = vec_dot_scalar(&packed, &activations);
        let neon_result = vec_dot_neon(&packed, &activations);

        assert_eq!(scalar_result, neon_result);
    }

    #[test]
    fn test_neon_extreme_values() {
        // Test with maximum i8 values
        let weights = vec![1i8; 128];
        let packed = pack_weights(&weights);
        let activations = vec![127i8; 128];

        let scalar_result = vec_dot_scalar(&packed, &activations);
        let neon_result = vec_dot_neon(&packed, &activations);

        assert_eq!(scalar_result, neon_result);
        assert_eq!(neon_result, 127 * 128);
    }

    #[test]
    fn test_neon_negative_activations() {
        let weights = vec![1i8; 128];
        let packed = pack_weights(&weights);
        let activations = vec![-100i8; 128];

        let scalar_result = vec_dot_scalar(&packed, &activations);
        let neon_result = vec_dot_neon(&packed, &activations);

        assert_eq!(scalar_result, neon_result);
        assert_eq!(neon_result, -100 * 128);
    }

    #[test]
    fn test_neon_mixed_weights_and_activations() {
        // Alternating positive and negative
        let weights: Vec<i8> = (0..128).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let packed = pack_weights(&weights);
        let activations: Vec<i8> = (0..128).map(|i| if i % 3 == 0 { 50 } else { -25 }).collect();

        let scalar_result = vec_dot_scalar(&packed, &activations);
        let neon_result = vec_dot_neon(&packed, &activations);

        assert_eq!(scalar_result, neon_result);
    }

    // NOTE: test_dotprod_matches_neon removed because vdotq_s32 requires nightly Rust
    // (unstable feature stdarch_neon_dotprod). Will re-add when it stabilizes.
}
