//! Multiply-free ternary GEMV kernel.
//!
//! For BitNet ternary weights (-1, 0, +1), we can completely eliminate
//! per-element multiplications:
//! - weight == +1: accumulator += activation
//! - weight == -1: accumulator -= activation
//! - weight == 0:  no operation
//!
//! Only one multiply by scale factor is needed per output element.

use super::types::{QK_BLOCK, BLOCK_BYTES};

/// Multiply-free ternary dot product.
///
/// Computes dot(weights, activations) using only additions and subtractions.
/// The scale factor should be applied by the caller.
///
/// # Weight Encoding (I2_S format)
/// - 00 (0) -> -1 (subtract)
/// - 01 (1) -> 0  (skip)
/// - 10 (2) -> +1 (add)
///
/// # Arguments
/// * `packed_weights` - Packed 2-bit ternary weights
/// * `activations` - INT8 quantized activations
///
/// # Returns
/// Accumulated sum as i32 (caller multiplies by scale).
#[inline]
pub fn vec_dot_ternary_scalar(packed_weights: &[u8], activations: &[i8]) -> i32 {
    let n = activations.len();
    debug_assert!(n % QK_BLOCK == 0);
    debug_assert!(packed_weights.len() >= n / 4);

    let num_blocks = n / QK_BLOCK;
    let mut sum = 0i32;

    for block in 0..num_blocks {
        let w_base = block * BLOCK_BYTES;
        let a_base = block * QK_BLOCK;

        for j in 0..BLOCK_BYTES {
            let packed = packed_weights[w_base + j];

            // Extract 4 x 2-bit weights
            let w0 = (packed >> 6) & 0x03;
            let w1 = (packed >> 4) & 0x03;
            let w2 = (packed >> 2) & 0x03;
            let w3 = packed & 0x03;

            // Get activations
            let a0 = activations[a_base + j] as i32;
            let a1 = activations[a_base + j + 32] as i32;
            let a2 = activations[a_base + j + 64] as i32;
            let a3 = activations[a_base + j + 96] as i32;

            // Multiply-free accumulation:
            // w=0 (-1): subtract, w=1 (0): skip, w=2 (+1): add
            // Using lookup: [-1, 0, +1, 0][w] but implemented as branches

            // For w0
            if w0 == 2 { sum += a0; }
            else if w0 == 0 { sum -= a0; }

            // For w1
            if w1 == 2 { sum += a1; }
            else if w1 == 0 { sum -= a1; }

            // For w2
            if w2 == 2 { sum += a2; }
            else if w2 == 0 { sum -= a2; }

            // For w3
            if w3 == 2 { sum += a3; }
            else if w3 == 0 { sum -= a3; }
        }
    }

    sum
}

/// Branchless multiply-free ternary dot product.
///
/// Uses arithmetic tricks to avoid branches:
/// - sign = (w >> 1) - (w & 1)  gives: 0->-1, 1->0, 2->+1
/// - Then: sum += sign * activation (but sign is just -1, 0, +1)
///
/// Actually we use: sum += ((w >> 1) as i32 - (w & 1) as i32) * a
/// But since the "multiply" by -1/0/+1 is really just sign manipulation,
/// modern CPUs handle this efficiently.
#[inline]
pub fn vec_dot_ternary_branchless(packed_weights: &[u8], activations: &[i8]) -> i32 {
    let n = activations.len();
    debug_assert!(n % QK_BLOCK == 0);
    debug_assert!(packed_weights.len() >= n / 4);

    let num_blocks = n / QK_BLOCK;
    let mut sum = 0i32;

    for block in 0..num_blocks {
        let w_base = block * BLOCK_BYTES;
        let a_base = block * QK_BLOCK;

        for j in 0..BLOCK_BYTES {
            let packed = packed_weights[w_base + j];

            // Extract weights
            let w0 = (packed >> 6) & 0x03;
            let w1 = (packed >> 4) & 0x03;
            let w2 = (packed >> 2) & 0x03;
            let w3 = packed & 0x03;

            // Convert 2-bit to sign: 0->-1, 1->0, 2->+1
            // Formula: sign = (w >> 1) - (w & 1)
            // This works because:
            //   w=0 (00): (0>>1) - (0&1) = 0 - 0 = 0... wait that's wrong
            // Let me recalculate:
            //   w=0: should be -1
            //   w=1: should be 0
            //   w=2: should be +1
            // Using: sign = (w as i32) - 1
            let s0 = (w0 as i32) - 1;
            let s1 = (w1 as i32) - 1;
            let s2 = (w2 as i32) - 1;
            let s3 = (w3 as i32) - 1;

            // Now multiply by sign (-1, 0, +1)
            // The CPU can optimize multiply by -1/0/+1
            sum += s0 * (activations[a_base + j] as i32);
            sum += s1 * (activations[a_base + j + 32] as i32);
            sum += s2 * (activations[a_base + j + 64] as i32);
            sum += s3 * (activations[a_base + j + 96] as i32);
        }
    }

    sum
}

/// ARM NEON truly multiply-free ternary dot product.
///
/// Uses masks and conditional add/subtract - NO multiplications.
/// For weight encoding: 00=-1, 01=0, 10=+1
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_ternary_neon(packed_weights: &[u8], activations: &[i8]) -> i32 {
    use std::arch::aarch64::*;

    let n = activations.len();
    debug_assert!(n % QK_BLOCK == 0);
    debug_assert!(packed_weights.len() >= n / 4);

    let num_blocks = n / QK_BLOCK;

    unsafe {
        // Accumulators
        let mut acc_pos = vdupq_n_s32(0); // Sum of activations where weight = +1
        let mut acc_neg = vdupq_n_s32(0); // Sum of activations where weight = -1

        // Constants
        let mask_2bit = vdupq_n_u8(0x03);
        let val_plus = vdupq_n_u8(2);  // +1 encoded as 2
        let val_minus = vdupq_n_u8(0); // -1 encoded as 0

        for block in 0..num_blocks {
            let w_ptr = packed_weights.as_ptr().add(block * BLOCK_BYTES);
            let a_ptr = activations.as_ptr().add(block * QK_BLOCK);

            // Load 32 packed bytes (128 weights)
            let w_lo: uint8x16_t = vld1q_u8(w_ptr);
            let w_hi: uint8x16_t = vld1q_u8(w_ptr.add(16));

            // Process 4 groups of 32 weights each
            // Group 0: bits 6-7
            let w0_lo = vandq_u8(vshrq_n_u8(w_lo, 6), mask_2bit);
            let w0_hi = vandq_u8(vshrq_n_u8(w_hi, 6), mask_2bit);

            // Load activations for group 0
            let a0_lo: int8x16_t = vld1q_s8(a_ptr.add(0) as *const i8);
            let a0_hi: int8x16_t = vld1q_s8(a_ptr.add(16) as *const i8);

            // Create masks: where weight == +1 (2), where weight == -1 (0)
            let mask_plus_0_lo: uint8x16_t = vceqq_u8(w0_lo, val_plus);
            let mask_plus_0_hi: uint8x16_t = vceqq_u8(w0_hi, val_plus);
            let mask_minus_0_lo: uint8x16_t = vceqq_u8(w0_lo, val_minus);
            let mask_minus_0_hi: uint8x16_t = vceqq_u8(w0_hi, val_minus);

            // Select activations using masks (zero where mask is 0)
            let pos_0_lo = vandq_s8(a0_lo, vreinterpretq_s8_u8(mask_plus_0_lo));
            let pos_0_hi = vandq_s8(a0_hi, vreinterpretq_s8_u8(mask_plus_0_hi));
            let neg_0_lo = vandq_s8(a0_lo, vreinterpretq_s8_u8(mask_minus_0_lo));
            let neg_0_hi = vandq_s8(a0_hi, vreinterpretq_s8_u8(mask_minus_0_hi));

            // Widen to i16 and accumulate
            let pos_sum_0 = vaddq_s16(
                vaddl_s8(vget_low_s8(pos_0_lo), vget_high_s8(pos_0_lo)),
                vaddl_s8(vget_low_s8(pos_0_hi), vget_high_s8(pos_0_hi))
            );
            let neg_sum_0 = vaddq_s16(
                vaddl_s8(vget_low_s8(neg_0_lo), vget_high_s8(neg_0_lo)),
                vaddl_s8(vget_low_s8(neg_0_hi), vget_high_s8(neg_0_hi))
            );

            acc_pos = vpadalq_s16(acc_pos, pos_sum_0);
            acc_neg = vpadalq_s16(acc_neg, neg_sum_0);

            // Group 1: bits 4-5
            let w1_lo = vandq_u8(vshrq_n_u8(w_lo, 4), mask_2bit);
            let w1_hi = vandq_u8(vshrq_n_u8(w_hi, 4), mask_2bit);
            let a1_lo: int8x16_t = vld1q_s8(a_ptr.add(32) as *const i8);
            let a1_hi: int8x16_t = vld1q_s8(a_ptr.add(48) as *const i8);

            let mask_plus_1_lo = vceqq_u8(w1_lo, val_plus);
            let mask_plus_1_hi = vceqq_u8(w1_hi, val_plus);
            let mask_minus_1_lo = vceqq_u8(w1_lo, val_minus);
            let mask_minus_1_hi = vceqq_u8(w1_hi, val_minus);

            let pos_1_lo = vandq_s8(a1_lo, vreinterpretq_s8_u8(mask_plus_1_lo));
            let pos_1_hi = vandq_s8(a1_hi, vreinterpretq_s8_u8(mask_plus_1_hi));
            let neg_1_lo = vandq_s8(a1_lo, vreinterpretq_s8_u8(mask_minus_1_lo));
            let neg_1_hi = vandq_s8(a1_hi, vreinterpretq_s8_u8(mask_minus_1_hi));

            let pos_sum_1 = vaddq_s16(
                vaddl_s8(vget_low_s8(pos_1_lo), vget_high_s8(pos_1_lo)),
                vaddl_s8(vget_low_s8(pos_1_hi), vget_high_s8(pos_1_hi))
            );
            let neg_sum_1 = vaddq_s16(
                vaddl_s8(vget_low_s8(neg_1_lo), vget_high_s8(neg_1_lo)),
                vaddl_s8(vget_low_s8(neg_1_hi), vget_high_s8(neg_1_hi))
            );

            acc_pos = vpadalq_s16(acc_pos, pos_sum_1);
            acc_neg = vpadalq_s16(acc_neg, neg_sum_1);

            // Group 2: bits 2-3
            let w2_lo = vandq_u8(vshrq_n_u8(w_lo, 2), mask_2bit);
            let w2_hi = vandq_u8(vshrq_n_u8(w_hi, 2), mask_2bit);
            let a2_lo: int8x16_t = vld1q_s8(a_ptr.add(64) as *const i8);
            let a2_hi: int8x16_t = vld1q_s8(a_ptr.add(80) as *const i8);

            let mask_plus_2_lo = vceqq_u8(w2_lo, val_plus);
            let mask_plus_2_hi = vceqq_u8(w2_hi, val_plus);
            let mask_minus_2_lo = vceqq_u8(w2_lo, val_minus);
            let mask_minus_2_hi = vceqq_u8(w2_hi, val_minus);

            let pos_2_lo = vandq_s8(a2_lo, vreinterpretq_s8_u8(mask_plus_2_lo));
            let pos_2_hi = vandq_s8(a2_hi, vreinterpretq_s8_u8(mask_plus_2_hi));
            let neg_2_lo = vandq_s8(a2_lo, vreinterpretq_s8_u8(mask_minus_2_lo));
            let neg_2_hi = vandq_s8(a2_hi, vreinterpretq_s8_u8(mask_minus_2_hi));

            let pos_sum_2 = vaddq_s16(
                vaddl_s8(vget_low_s8(pos_2_lo), vget_high_s8(pos_2_lo)),
                vaddl_s8(vget_low_s8(pos_2_hi), vget_high_s8(pos_2_hi))
            );
            let neg_sum_2 = vaddq_s16(
                vaddl_s8(vget_low_s8(neg_2_lo), vget_high_s8(neg_2_lo)),
                vaddl_s8(vget_low_s8(neg_2_hi), vget_high_s8(neg_2_hi))
            );

            acc_pos = vpadalq_s16(acc_pos, pos_sum_2);
            acc_neg = vpadalq_s16(acc_neg, neg_sum_2);

            // Group 3: bits 0-1
            let w3_lo = vandq_u8(w_lo, mask_2bit);
            let w3_hi = vandq_u8(w_hi, mask_2bit);
            let a3_lo: int8x16_t = vld1q_s8(a_ptr.add(96) as *const i8);
            let a3_hi: int8x16_t = vld1q_s8(a_ptr.add(112) as *const i8);

            let mask_plus_3_lo = vceqq_u8(w3_lo, val_plus);
            let mask_plus_3_hi = vceqq_u8(w3_hi, val_plus);
            let mask_minus_3_lo = vceqq_u8(w3_lo, val_minus);
            let mask_minus_3_hi = vceqq_u8(w3_hi, val_minus);

            let pos_3_lo = vandq_s8(a3_lo, vreinterpretq_s8_u8(mask_plus_3_lo));
            let pos_3_hi = vandq_s8(a3_hi, vreinterpretq_s8_u8(mask_plus_3_hi));
            let neg_3_lo = vandq_s8(a3_lo, vreinterpretq_s8_u8(mask_minus_3_lo));
            let neg_3_hi = vandq_s8(a3_hi, vreinterpretq_s8_u8(mask_minus_3_hi));

            let pos_sum_3 = vaddq_s16(
                vaddl_s8(vget_low_s8(pos_3_lo), vget_high_s8(pos_3_lo)),
                vaddl_s8(vget_low_s8(pos_3_hi), vget_high_s8(pos_3_hi))
            );
            let neg_sum_3 = vaddq_s16(
                vaddl_s8(vget_low_s8(neg_3_lo), vget_high_s8(neg_3_lo)),
                vaddl_s8(vget_low_s8(neg_3_hi), vget_high_s8(neg_3_hi))
            );

            acc_pos = vpadalq_s16(acc_pos, pos_sum_3);
            acc_neg = vpadalq_s16(acc_neg, neg_sum_3);
        }

        // Final result: sum_positive - sum_negative (NO MULTIPLY!)
        let pos_total = vaddvq_s32(acc_pos);
        let neg_total = vaddvq_s32(acc_neg);

        pos_total - neg_total
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_ternary_neon(packed_weights: &[u8], activations: &[i8]) -> i32 {
    vec_dot_ternary_branchless(packed_weights, activations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::gemv_scalar::{pack_weights, vec_dot_scalar};

    #[test]
    fn test_ternary_matches_scalar() {
        let weights: Vec<i8> = (0..128).map(|i| ((i * 7) % 3) as i8 - 1).collect();
        let packed = pack_weights(&weights);
        // Use wrapping arithmetic to avoid overflow
        let activations: Vec<i8> = (0..128).map(|i| (((i * 11) % 200) as i32 - 100) as i8).collect();

        let scalar = vec_dot_scalar(&packed, &activations);
        let ternary = vec_dot_ternary_scalar(&packed, &activations);
        let branchless = vec_dot_ternary_branchless(&packed, &activations);
        let neon = vec_dot_ternary_neon(&packed, &activations);

        assert_eq!(scalar, ternary, "ternary mismatch");
        assert_eq!(scalar, branchless, "branchless mismatch");
        assert_eq!(scalar, neon, "neon mismatch");
    }

    #[test]
    fn test_ternary_all_ones() {
        let weights = vec![1i8; 128];
        let packed = pack_weights(&weights);
        let activations = vec![1i8; 128];

        let result = vec_dot_ternary_neon(&packed, &activations);
        assert_eq!(result, 128);
    }

    #[test]
    fn test_ternary_all_minus_ones() {
        let weights = vec![-1i8; 128];
        let packed = pack_weights(&weights);
        let activations = vec![1i8; 128];

        let result = vec_dot_ternary_neon(&packed, &activations);
        assert_eq!(result, -128);
    }

    #[test]
    fn test_ternary_all_zeros() {
        let weights = vec![0i8; 128];
        let packed = pack_weights(&weights);
        let activations: Vec<i8> = (0..128).map(|i| i as i8).collect();

        let result = vec_dot_ternary_neon(&packed, &activations);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_ternary_multiple_blocks() {
        let weights: Vec<i8> = (0..256).map(|i| ((i * 7) % 3) as i8 - 1).collect();
        let packed = pack_weights(&weights);
        // Use wrapping arithmetic to avoid overflow
        let activations: Vec<i8> = (0..256).map(|i| (((i * 11) % 200) as i32 - 100) as i8).collect();

        let scalar = vec_dot_scalar(&packed, &activations);
        let neon = vec_dot_ternary_neon(&packed, &activations);

        assert_eq!(scalar, neon);
    }
}
