//! TL1 Multiply-Free kernel for BitNet ternary inference.
//!
//! Truly multiply-free implementation:
//! - Weight -1: subtract activation
//! - Weight  0: skip (add zero)
//! - Weight +1: add activation
//!
//! Uses comparison masks and conditional add/subtract - zero multiplications!

use super::types::{QK_BLOCK, BLOCK_BYTES};

/// Truly multiply-free dot product for ternary weights.
///
/// For weight encoding: 0=-1, 1=0, 2=+1
/// - When w=0 (meaning -1): acc -= activation
/// - When w=1 (meaning  0): acc += 0
/// - When w=2 (meaning +1): acc += activation
///
/// Uses comparison masks and conditional operations - NO multiplications!
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_tl1(packed_weights: &[u8], activations: &[i8]) -> i32 {
    use std::arch::aarch64::*;

    let n = activations.len();
    debug_assert!(n % QK_BLOCK == 0);
    debug_assert!(packed_weights.len() >= n / 4);

    let num_blocks = n / QK_BLOCK;

    unsafe {
        // Accumulator (int32 to avoid overflow)
        let mut acc = vdupq_n_s32(0);

        for block in 0..num_blocks {
            let w_ptr = packed_weights.as_ptr().add(block * BLOCK_BYTES);
            let a_ptr = activations.as_ptr().add(block * QK_BLOCK);

            // Load all 32 packed weight bytes (128 weights, 2 bits each)
            let w_bytes_lo = vld1q_u8(w_ptr);           // bytes 0-15
            let w_bytes_hi = vld1q_u8(w_ptr.add(16));   // bytes 16-31

            // Process in 4 groups of 32 weights each
            // Group 0: bits 6-7 of each byte (weights 0-31)
            acc = process_group_multiply_free(acc, w_bytes_lo, w_bytes_hi, a_ptr, 6);

            // Group 1: bits 4-5 of each byte (weights 32-63)
            acc = process_group_multiply_free(acc, w_bytes_lo, w_bytes_hi, a_ptr.add(32), 4);

            // Group 2: bits 2-3 of each byte (weights 64-95)
            acc = process_group_multiply_free(acc, w_bytes_lo, w_bytes_hi, a_ptr.add(64), 2);

            // Group 3: bits 0-1 of each byte (weights 96-127)
            acc = process_group_multiply_free(acc, w_bytes_lo, w_bytes_hi, a_ptr.add(96), 0);
        }

        // Horizontal sum
        vaddvq_s32(acc)
    }
}

/// Process 32 weights multiply-free using comparison masks.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn process_group_multiply_free(
    acc: std::arch::aarch64::int32x4_t,
    w_bytes_lo: std::arch::aarch64::uint8x16_t,
    w_bytes_hi: std::arch::aarch64::uint8x16_t,
    a_ptr: *const i8,
    shift: i32,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;

    let mask_2bit = vdupq_n_u8(0x03);

    // Extract 2-bit weights for this group (0, 1, or 2)
    let w_lo: uint8x16_t;
    let w_hi: uint8x16_t;

    match shift {
        6 => {
            w_lo = vandq_u8(vshrq_n_u8::<6>(w_bytes_lo), mask_2bit);
            w_hi = vandq_u8(vshrq_n_u8::<6>(w_bytes_hi), mask_2bit);
        }
        4 => {
            w_lo = vandq_u8(vshrq_n_u8::<4>(w_bytes_lo), mask_2bit);
            w_hi = vandq_u8(vshrq_n_u8::<4>(w_bytes_hi), mask_2bit);
        }
        2 => {
            w_lo = vandq_u8(vshrq_n_u8::<2>(w_bytes_lo), mask_2bit);
            w_hi = vandq_u8(vshrq_n_u8::<2>(w_bytes_hi), mask_2bit);
        }
        _ => {
            w_lo = vandq_u8(w_bytes_lo, mask_2bit);
            w_hi = vandq_u8(w_bytes_hi, mask_2bit);
        }
    }

    // Load 32 activations
    let a_0_15 = vld1q_s8(a_ptr);
    let a_16_31 = vld1q_s8(a_ptr.add(16));

    // Constants for comparison
    let zero_u8 = vdupq_n_u8(0);
    let two_u8 = vdupq_n_u8(2);

    // Create masks: neg_mask where w==0 (meaning -1), pos_mask where w==2 (meaning +1)
    let neg_mask_lo = vceqq_u8(w_lo, zero_u8);  // 0xFF where weight = -1
    let neg_mask_hi = vceqq_u8(w_hi, zero_u8);
    let pos_mask_lo = vceqq_u8(w_lo, two_u8);   // 0xFF where weight = +1
    let pos_mask_hi = vceqq_u8(w_hi, two_u8);

    // Convert activations to unsigned for masking
    let a_lo_u = vreinterpretq_u8_s8(a_0_15);
    let a_hi_u = vreinterpretq_u8_s8(a_16_31);

    // Select activations: a where mask is set, 0 otherwise
    let a_pos_lo = vandq_u8(a_lo_u, pos_mask_lo);
    let a_pos_hi = vandq_u8(a_hi_u, pos_mask_hi);
    let a_neg_lo = vandq_u8(a_lo_u, neg_mask_lo);
    let a_neg_hi = vandq_u8(a_hi_u, neg_mask_hi);

    // Reinterpret back to signed
    let a_pos_lo_s = vreinterpretq_s8_u8(a_pos_lo);
    let a_pos_hi_s = vreinterpretq_s8_u8(a_pos_hi);
    let a_neg_lo_s = vreinterpretq_s8_u8(a_neg_lo);
    let a_neg_hi_s = vreinterpretq_s8_u8(a_neg_hi);

    // Widening accumulation into int16
    // Add positive contributions, subtract negative contributions
    let mut sum = vdupq_n_s16(0);

    // Add where +1
    sum = vaddw_s8(sum, vget_low_s8(a_pos_lo_s));
    sum = vaddw_high_s8(sum, a_pos_lo_s);
    sum = vaddw_s8(sum, vget_low_s8(a_pos_hi_s));
    sum = vaddw_high_s8(sum, a_pos_hi_s);

    // Subtract where -1
    sum = vsubw_s8(sum, vget_low_s8(a_neg_lo_s));
    sum = vsubw_high_s8(sum, a_neg_lo_s);
    sum = vsubw_s8(sum, vget_low_s8(a_neg_hi_s));
    sum = vsubw_high_s8(sum, a_neg_hi_s);

    // Widen to int32 and accumulate
    vpadalq_s16(acc, sum)
}

/// Alternative: Branchless multiply-free using arithmetic instead of masks.
///
/// Uses the insight that for w in {-1, 0, +1}:
/// w * a = (w > 0 ? a : 0) - (w < 0 ? a : 0)
///       = a * sign(w) when w != 0
///
/// With ternary encoding (0=-1, 1=0, 2=+1):
/// contribution = a * (w - 1) without actual multiplication
/// by using: (w - 1) is -1, 0, or +1
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub fn vec_dot_tl1_v2(packed_weights: &[u8], activations: &[i8]) -> i32 {
    use std::arch::aarch64::*;

    let n = activations.len();
    debug_assert!(n % QK_BLOCK == 0);
    debug_assert!(packed_weights.len() >= n / 4);

    let num_blocks = n / QK_BLOCK;

    unsafe {
        let mut acc = vdupq_n_s32(0);

        for block in 0..num_blocks {
            let w_ptr = packed_weights.as_ptr().add(block * BLOCK_BYTES);
            let a_ptr = activations.as_ptr().add(block * QK_BLOCK);

            let w_bytes_lo = vld1q_u8(w_ptr);
            let w_bytes_hi = vld1q_u8(w_ptr.add(16));

            acc = process_group_v2(acc, w_bytes_lo, w_bytes_hi, a_ptr, 6);
            acc = process_group_v2(acc, w_bytes_lo, w_bytes_hi, a_ptr.add(32), 4);
            acc = process_group_v2(acc, w_bytes_lo, w_bytes_hi, a_ptr.add(64), 2);
            acc = process_group_v2(acc, w_bytes_lo, w_bytes_hi, a_ptr.add(96), 0);
        }

        vaddvq_s32(acc)
    }
}

/// Process using sign extraction (branchless, no multiplication).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn process_group_v2(
    acc: std::arch::aarch64::int32x4_t,
    w_bytes_lo: std::arch::aarch64::uint8x16_t,
    w_bytes_hi: std::arch::aarch64::uint8x16_t,
    a_ptr: *const i8,
    shift: i32,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;

    let mask_2bit = vdupq_n_u8(0x03);
    let bias = vdupq_n_s8(1);

    // Extract 2-bit weights
    let w_lo: uint8x16_t;
    let w_hi: uint8x16_t;

    match shift {
        6 => {
            w_lo = vandq_u8(vshrq_n_u8::<6>(w_bytes_lo), mask_2bit);
            w_hi = vandq_u8(vshrq_n_u8::<6>(w_bytes_hi), mask_2bit);
        }
        4 => {
            w_lo = vandq_u8(vshrq_n_u8::<4>(w_bytes_lo), mask_2bit);
            w_hi = vandq_u8(vshrq_n_u8::<4>(w_bytes_hi), mask_2bit);
        }
        2 => {
            w_lo = vandq_u8(vshrq_n_u8::<2>(w_bytes_lo), mask_2bit);
            w_hi = vandq_u8(vshrq_n_u8::<2>(w_bytes_hi), mask_2bit);
        }
        _ => {
            w_lo = vandq_u8(w_bytes_lo, mask_2bit);
            w_hi = vandq_u8(w_bytes_hi, mask_2bit);
        }
    }

    // Center weights: 0,1,2 -> -1,0,+1
    let w_lo_s = vsubq_s8(vreinterpretq_s8_u8(w_lo), bias);
    let w_hi_s = vsubq_s8(vreinterpretq_s8_u8(w_hi), bias);

    // Load activations
    let a_0_15 = vld1q_s8(a_ptr);
    let a_16_31 = vld1q_s8(a_ptr.add(16));

    // Extract sign of weights: -1 for negative, 0 for zero, +1 for positive
    // We can use comparison to create masks
    let zero = vdupq_n_s8(0);

    // For w_lo_s (which is -1, 0, or +1):
    // neg_mask = w < 0 gives 0xFF for -1
    // pos_mask = w > 0 gives 0xFF for +1
    let neg_lo = vcltq_s8(w_lo_s, zero);
    let pos_lo = vcgtq_s8(w_lo_s, zero);
    let neg_hi = vcltq_s8(w_hi_s, zero);
    let pos_hi = vcgtq_s8(w_hi_s, zero);

    // Reinterpret masks as int8 (0xFF = -1, 0x00 = 0)
    // sign = pos_mask - neg_mask gives +1, 0, or -1
    // But we need this as signed bytes
    let sign_lo = vsubq_s8(
        vreinterpretq_s8_u8(pos_lo),
        vreinterpretq_s8_u8(neg_lo)
    );
    let sign_hi = vsubq_s8(
        vreinterpretq_s8_u8(pos_hi),
        vreinterpretq_s8_u8(neg_hi)
    );

    // Now sign_lo/hi contains -1, 0, or +1 for each position
    // We need: acc += activation * sign (but multiply-free!)
    //
    // Alternative: separate into positive and negative contributions
    // positive = a & pos_mask
    // negative = a & neg_mask
    // result = positive - negative

    let a_lo_u = vreinterpretq_u8_s8(a_0_15);
    let a_hi_u = vreinterpretq_u8_s8(a_16_31);

    let a_pos_lo = vreinterpretq_s8_u8(vandq_u8(a_lo_u, pos_lo));
    let a_neg_lo = vreinterpretq_s8_u8(vandq_u8(a_lo_u, neg_lo));
    let a_pos_hi = vreinterpretq_s8_u8(vandq_u8(a_hi_u, pos_hi));
    let a_neg_hi = vreinterpretq_s8_u8(vandq_u8(a_hi_u, neg_hi));

    // Widening accumulation
    let mut sum = vdupq_n_s16(0);

    sum = vaddw_s8(sum, vget_low_s8(a_pos_lo));
    sum = vaddw_high_s8(sum, a_pos_lo);
    sum = vaddw_s8(sum, vget_low_s8(a_pos_hi));
    sum = vaddw_high_s8(sum, a_pos_hi);

    sum = vsubw_s8(sum, vget_low_s8(a_neg_lo));
    sum = vsubw_high_s8(sum, a_neg_lo);
    sum = vsubw_s8(sum, vget_low_s8(a_neg_hi));
    sum = vsubw_high_s8(sum, a_neg_hi);

    // Suppress warning for unused variable
    let _ = sign_lo;
    let _ = sign_hi;

    vpadalq_s16(acc, sum)
}

/// Fallback for non-ARM platforms.
#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_tl1(packed_weights: &[u8], activations: &[i8]) -> i32 {
    super::gemv_scalar::vec_dot_scalar(packed_weights, activations)
}

#[cfg(not(target_arch = "aarch64"))]
#[allow(dead_code)]
pub fn vec_dot_tl1_v2(packed_weights: &[u8], activations: &[i8]) -> i32 {
    super::gemv_scalar::vec_dot_scalar(packed_weights, activations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::gemv_scalar::{pack_weights, vec_dot_scalar};

    #[test]
    fn test_tl1_matches_scalar() {
        let weights: Vec<i8> = (0..128).map(|i| ((i * 7) % 3) as i8 - 1).collect();
        let packed = pack_weights(&weights);
        let activations: Vec<i8> = (0..128).map(|i| (((i * 11) % 200) as i32 - 100) as i8).collect();

        let scalar = vec_dot_scalar(&packed, &activations);
        let tl1 = vec_dot_tl1(&packed, &activations);

        assert_eq!(scalar, tl1, "TL1 kernel mismatch");
    }

    #[test]
    fn test_tl1_v2_matches_scalar() {
        let weights: Vec<i8> = (0..128).map(|i| ((i * 7) % 3) as i8 - 1).collect();
        let packed = pack_weights(&weights);
        let activations: Vec<i8> = (0..128).map(|i| (((i * 11) % 200) as i32 - 100) as i8).collect();

        let scalar = vec_dot_scalar(&packed, &activations);
        let tl1_v2 = vec_dot_tl1_v2(&packed, &activations);

        assert_eq!(scalar, tl1_v2, "TL1 v2 kernel mismatch");
    }

    #[test]
    fn test_tl1_multiple_blocks() {
        let weights: Vec<i8> = (0..256).map(|i| ((i * 17) % 3) as i8 - 1).collect();
        let packed = pack_weights(&weights);
        let activations: Vec<i8> = (0..256).map(|i| (((i * 23) % 200) as i32 - 100) as i8).collect();

        let scalar = vec_dot_scalar(&packed, &activations);
        let tl1 = vec_dot_tl1(&packed, &activations);

        assert_eq!(scalar, tl1, "TL1 kernel mismatch for multiple blocks");
    }

    #[test]
    fn test_tl1_all_ones() {
        let weights = vec![1i8; 128];  // All +1
        let packed = pack_weights(&weights);
        let activations: Vec<i8> = (0..128).map(|i| i as i8).collect();

        let expected: i32 = activations.iter().map(|&a| a as i32).sum();
        let tl1 = vec_dot_tl1(&packed, &activations);

        assert_eq!(expected, tl1, "All +1 weights should sum activations");
    }

    #[test]
    fn test_tl1_all_neg_ones() {
        let weights = vec![-1i8; 128];  // All -1
        let packed = pack_weights(&weights);
        let activations: Vec<i8> = (0..128).map(|i| i as i8).collect();

        let expected: i32 = -activations.iter().map(|&a| a as i32).sum::<i32>();
        let tl1 = vec_dot_tl1(&packed, &activations);

        assert_eq!(expected, tl1, "All -1 weights should negate sum");
    }

    #[test]
    fn test_tl1_all_zeros() {
        let weights = vec![0i8; 128];  // All 0
        let packed = pack_weights(&weights);
        let activations: Vec<i8> = (0..128).map(|i| i as i8).collect();

        let tl1 = vec_dot_tl1(&packed, &activations);

        assert_eq!(0, tl1, "All 0 weights should give 0");
    }
}
