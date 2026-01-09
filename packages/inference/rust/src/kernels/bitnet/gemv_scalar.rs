//! Scalar fallback implementation for BitNet GEMV.
//!
//! This is a reference implementation used for testing correctness.
//! It's not optimized for performance - use gemv_neon.rs for production.

use super::types::{QK_BLOCK, BLOCK_BYTES};

/// Scalar dot product: packed_weights (I2_S) @ quantized_activations (INT8).
///
/// # Weight Format (I2_S)
/// - 128-element blocks, 4 weights per byte (2 bits each)
/// - Byte layout within a block (32 bytes for 128 weights):
///   - byte[j].bits[6:7] -> weight[j+0]    (first 32 weights)
///   - byte[j].bits[4:5] -> weight[j+32]   (second 32 weights)
///   - byte[j].bits[2:3] -> weight[j+64]   (third 32 weights)
///   - byte[j].bits[0:1] -> weight[j+96]   (fourth 32 weights)
///
/// # Encoding
/// - 00 (0) -> -1
/// - 01 (1) -> 0
/// - 10 (2) -> +1
/// - 11 (3) -> reserved (should not appear)
///
/// # Arguments
/// * `packed_weights` - Packed 2-bit weights (n/4 bytes)
/// * `activations` - INT8 activations (n elements)
///
/// # Returns
/// The dot product result as i32.
///
/// # Panics
/// Panics if activations length is not a multiple of 128.
pub fn vec_dot_scalar(packed_weights: &[u8], activations: &[i8]) -> i32 {
    let n = activations.len();
    assert!(n % QK_BLOCK == 0, "activations length must be multiple of {}", QK_BLOCK);
    assert!(
        packed_weights.len() >= n / 4,
        "packed_weights too short: {} < {}",
        packed_weights.len(),
        n / 4
    );

    let num_blocks = n / QK_BLOCK;
    let mut sum = 0i32;

    for block in 0..num_blocks {
        let w_base = block * BLOCK_BYTES;
        let a_base = block * QK_BLOCK;

        for j in 0..BLOCK_BYTES {
            let packed = packed_weights[w_base + j];

            // Extract 4 weights from packed byte
            // Each 2-bit value: 0=-1, 1=0, 2=+1
            let w0 = ((packed >> 6) & 0x03) as i32 - 1; // bits 6-7 -> weight[j]
            let w1 = ((packed >> 4) & 0x03) as i32 - 1; // bits 4-5 -> weight[j+32]
            let w2 = ((packed >> 2) & 0x03) as i32 - 1; // bits 2-3 -> weight[j+64]
            let w3 = (packed & 0x03) as i32 - 1;        // bits 0-1 -> weight[j+96]

            // Multiply-accumulate
            sum += w0 * (activations[a_base + j] as i32);
            sum += w1 * (activations[a_base + j + 32] as i32);
            sum += w2 * (activations[a_base + j + 64] as i32);
            sum += w3 * (activations[a_base + j + 96] as i32);
        }
    }

    sum
}

/// Pack ternary weights into I2_S format.
///
/// # Arguments
/// * `weights` - Ternary weights as i8 (-1, 0, +1 only)
///
/// # Returns
/// Packed weights in I2_S format.
///
/// # Panics
/// Panics if weights length is not a multiple of 128 or contains invalid values.
#[allow(dead_code)]
pub fn pack_weights(weights: &[i8]) -> Vec<u8> {
    let n = weights.len();
    assert!(n % QK_BLOCK == 0, "weights length must be multiple of {}", QK_BLOCK);

    let num_blocks = n / QK_BLOCK;
    let mut packed = vec![0u8; n / 4];

    for block in 0..num_blocks {
        let w_base = block * BLOCK_BYTES;
        let src_base = block * QK_BLOCK;

        for j in 0..BLOCK_BYTES {
            // Get the 4 weights that go into this byte
            let w0 = weights[src_base + j];           // -> bits 6-7
            let w1 = weights[src_base + j + 32];      // -> bits 4-5
            let w2 = weights[src_base + j + 64];      // -> bits 2-3
            let w3 = weights[src_base + j + 96];      // -> bits 0-1

            // Convert -1,0,+1 to 0,1,2
            let e0 = (w0 + 1) as u8;
            let e1 = (w1 + 1) as u8;
            let e2 = (w2 + 1) as u8;
            let e3 = (w3 + 1) as u8;

            assert!(e0 <= 2 && e1 <= 2 && e2 <= 2 && e3 <= 2,
                "Invalid weight value at block {}, position {}", block, j);

            // Pack into byte
            packed[w_base + j] = (e0 << 6) | (e1 << 4) | (e2 << 2) | e3;
        }
    }

    packed
}

/// Unpack I2_S weights to ternary values.
///
/// Inverse of pack_weights(). Useful for debugging.
#[allow(dead_code)]
pub fn unpack_weights(packed: &[u8], n: usize) -> Vec<i8> {
    assert!(n % QK_BLOCK == 0);
    assert!(packed.len() >= n / 4);

    let mut weights = vec![0i8; n];
    let num_blocks = n / QK_BLOCK;

    for block in 0..num_blocks {
        let w_base = block * BLOCK_BYTES;
        let dst_base = block * QK_BLOCK;

        for j in 0..BLOCK_BYTES {
            let packed_byte = packed[w_base + j];

            // Extract and convert 2-bit values to ternary
            weights[dst_base + j]      = (((packed_byte >> 6) & 0x03) as i8) - 1;
            weights[dst_base + j + 32] = (((packed_byte >> 4) & 0x03) as i8) - 1;
            weights[dst_base + j + 64] = (((packed_byte >> 2) & 0x03) as i8) - 1;
            weights[dst_base + j + 96] = ((packed_byte & 0x03) as i8) - 1;
        }
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        // Create random ternary weights
        let weights: Vec<i8> = (0..128)
            .map(|i| ((i % 3) as i8) - 1) // -1, 0, 1, -1, 0, 1, ...
            .collect();

        let packed = pack_weights(&weights);
        let unpacked = unpack_weights(&packed, 128);

        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_pack_unpack_all_ones() {
        let weights = vec![1i8; 128];
        let packed = pack_weights(&weights);
        let unpacked = unpack_weights(&packed, 128);

        assert_eq!(weights, unpacked);
        // All +1 encodes as 10, so each byte should be 10_10_10_10 = 0xAA
        assert!(packed.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn test_pack_unpack_all_zeros() {
        let weights = vec![0i8; 128];
        let packed = pack_weights(&weights);
        let unpacked = unpack_weights(&packed, 128);

        assert_eq!(weights, unpacked);
        // All 0 encodes as 01, so each byte should be 01_01_01_01 = 0x55
        assert!(packed.iter().all(|&b| b == 0x55));
    }

    #[test]
    fn test_pack_unpack_all_minus_ones() {
        let weights = vec![-1i8; 128];
        let packed = pack_weights(&weights);
        let unpacked = unpack_weights(&packed, 128);

        assert_eq!(weights, unpacked);
        // All -1 encodes as 00, so each byte should be 00_00_00_00 = 0x00
        assert!(packed.iter().all(|&b| b == 0x00));
    }

    #[test]
    fn test_vec_dot_all_ones() {
        // 128 weights = +1, 128 activations = +1
        // Result should be 128
        let weights = vec![1i8; 128];
        let packed = pack_weights(&weights);
        let activations = vec![1i8; 128];

        let result = vec_dot_scalar(&packed, &activations);
        assert_eq!(result, 128);
    }

    #[test]
    fn test_vec_dot_all_minus_ones() {
        // 128 weights = -1, 128 activations = +1
        // Result should be -128
        let weights = vec![-1i8; 128];
        let packed = pack_weights(&weights);
        let activations = vec![1i8; 128];

        let result = vec_dot_scalar(&packed, &activations);
        assert_eq!(result, -128);
    }

    #[test]
    fn test_vec_dot_all_zeros() {
        // 128 weights = 0, any activations
        // Result should be 0
        let weights = vec![0i8; 128];
        let packed = pack_weights(&weights);
        let activations: Vec<i8> = (0..128).map(|i| (i % 256) as i8).collect();

        let result = vec_dot_scalar(&packed, &activations);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_vec_dot_alternating() {
        // Alternating +1, -1 weights
        let weights: Vec<i8> = (0..128).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let packed = pack_weights(&weights);
        // Activations = 1 for all
        let activations = vec![1i8; 128];

        // Sum should be 0 (64 * 1 + 64 * -1)
        let result = vec_dot_scalar(&packed, &activations);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_vec_dot_mixed() {
        // Known values for manual verification
        let mut weights = vec![0i8; 128];
        weights[0] = 1;   // +1
        weights[1] = -1;  // -1
        weights[32] = 1;  // +1 (second group)
        weights[64] = -1; // -1 (third group)
        weights[96] = 1;  // +1 (fourth group)

        let packed = pack_weights(&weights);

        let mut activations = vec![0i8; 128];
        activations[0] = 10;
        activations[1] = 20;
        activations[32] = 30;
        activations[64] = 40;
        activations[96] = 50;

        // Expected: 1*10 + (-1)*20 + 1*30 + (-1)*40 + 1*50 = 10 - 20 + 30 - 40 + 50 = 30
        let result = vec_dot_scalar(&packed, &activations);
        assert_eq!(result, 30);
    }

    #[test]
    fn test_vec_dot_multiple_blocks() {
        // Test with 256 elements (2 blocks)
        let weights = vec![1i8; 256];
        let packed = pack_weights(&weights);
        let activations = vec![1i8; 256];

        let result = vec_dot_scalar(&packed, &activations);
        assert_eq!(result, 256);
    }

    #[test]
    fn test_vec_dot_large_activations() {
        // Test with larger activation values to check overflow handling
        let weights = vec![1i8; 128];
        let packed = pack_weights(&weights);
        let activations = vec![127i8; 128]; // Max i8 value

        let result = vec_dot_scalar(&packed, &activations);
        assert_eq!(result, 127 * 128);
    }

    #[test]
    fn test_vec_dot_negative_activations() {
        // Test with negative activations
        let weights = vec![1i8; 128];
        let packed = pack_weights(&weights);
        let activations = vec![-100i8; 128];

        let result = vec_dot_scalar(&packed, &activations);
        assert_eq!(result, -100 * 128);
    }

    #[test]
    #[should_panic(expected = "activations length must be multiple of 128")]
    fn test_vec_dot_invalid_length() {
        let packed = vec![0u8; 32];
        let activations = vec![0i8; 100]; // Not multiple of 128
        vec_dot_scalar(&packed, &activations);
    }

    #[test]
    fn test_byte_layout() {
        // Verify the byte layout matches the spec
        // Create weights where we know exactly which values go where
        let mut weights = vec![0i8; 128];
        weights[0] = 1;   // Should be in byte[0] bits 6-7
        weights[32] = -1; // Should be in byte[0] bits 4-5
        weights[64] = 1;  // Should be in byte[0] bits 2-3
        weights[96] = 0;  // Should be in byte[0] bits 0-1

        let packed = pack_weights(&weights);

        // byte[0] should be: 10_00_10_01 = 0x89
        // w0=1 (+1) -> 10, w1=-1 -> 00, w2=1 (+1) -> 10, w3=0 -> 01
        assert_eq!(packed[0], 0b10_00_10_01);
    }
}
