//! FP32 to INT8 activation quantization.
//!
//! Implements symmetric quantization: q = round(x / scale) where scale = max(|x|) / 127.

/// Quantize FP32 activations to INT8 using symmetric quantization.
///
/// # Algorithm
/// 1. Find max absolute value across all elements
/// 2. Compute scale = max_abs / 127
/// 3. Quantize: q[i] = clamp(round(x[i] / scale), -127, 127)
///
/// # Returns
/// (quantized_values, scale_factor)
///
/// # Example
/// ```ignore
/// let input = vec![1.0, -0.5, 0.25, -1.0, 0.0];
/// let (quantized, scale) = quantize_activations(&input);
/// assert_eq!(quantized[0], 127);   // 1.0 maps to 127
/// assert_eq!(quantized[3], -127);  // -1.0 maps to -127
/// ```
pub fn quantize_activations(input: &[f32]) -> (Vec<i8>, f32) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), 1.0);
    }

    // Find max absolute value
    let max_abs = input
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);

    // Handle zero input
    if max_abs == 0.0 {
        return (vec![0i8; n], 1.0);
    }

    // Compute scale: map max_abs to 127
    let scale = max_abs / 127.0;
    let inv_scale = 127.0 / max_abs;

    // Quantize
    let output: Vec<i8> = input
        .iter()
        .map(|&x| {
            let val = (x * inv_scale).round() as i32;
            val.clamp(-127, 127) as i8
        })
        .collect();

    (output, scale)
}

/// Dequantize INT8 values back to FP32.
///
/// Useful for testing quantization accuracy.
#[allow(dead_code)]
pub fn dequantize_activations(quantized: &[i8], scale: f32) -> Vec<f32> {
    quantized
        .iter()
        .map(|&q| (q as f32) * scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_empty() {
        let (quantized, scale) = quantize_activations(&[]);
        assert!(quantized.is_empty());
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_quantize_zeros() {
        let input = vec![0.0; 10];
        let (quantized, scale) = quantize_activations(&input);
        assert!(quantized.iter().all(|&x| x == 0));
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_quantize_unit_range() {
        // Values in [-1, 1] should map to [-127, 127]
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0];
        let (quantized, scale) = quantize_activations(&input);

        assert_eq!(quantized[0], 127);   // 1.0
        assert_eq!(quantized[1], -127);  // -1.0
        assert_eq!(quantized[2], 64);    // 0.5 -> round(63.5) = 64
        assert_eq!(quantized[3], -64);   // -0.5 -> round(-63.5) = -64
        assert_eq!(quantized[4], 0);     // 0.0

        // Scale should be 1/127
        assert!((scale - 1.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_larger_range() {
        // Values in [-2, 2] should scale appropriately
        let input = vec![2.0, -2.0, 1.0, -1.0, 0.0];
        let (quantized, scale) = quantize_activations(&input);

        assert_eq!(quantized[0], 127);   // 2.0 -> max -> 127
        assert_eq!(quantized[1], -127);  // -2.0 -> min -> -127
        assert_eq!(quantized[2], 64);    // 1.0 -> half of max -> 64
        assert_eq!(quantized[3], -64);   // -1.0 -> -64
        assert_eq!(quantized[4], 0);     // 0.0

        // Scale should be 2/127
        assert!((scale - 2.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_asymmetric() {
        // Max is 3.0, min is -1.0, so scale is based on 3.0
        let input = vec![3.0, -1.0, 1.5, 0.0];
        let (quantized, scale) = quantize_activations(&input);

        assert_eq!(quantized[0], 127);   // 3.0 -> max -> 127
        assert_eq!(quantized[1], -42);   // -1.0 -> round(-127/3) = -42
        assert_eq!(quantized[2], 64);    // 1.5 -> round(127*1.5/3) = 64
        assert_eq!(quantized[3], 0);     // 0.0

        // Scale should be 3/127
        assert!((scale - 3.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_small_values() {
        // Very small values should still quantize properly
        let input = vec![0.001, -0.001, 0.0005];
        let (quantized, scale) = quantize_activations(&input);

        assert_eq!(quantized[0], 127);   // max value
        assert_eq!(quantized[1], -127);  // min value
        assert_eq!(quantized[2], 64);    // half of max
    }

    #[test]
    fn test_roundtrip_accuracy() {
        // Test that quantize -> dequantize is reasonably accurate
        let input: Vec<f32> = (-127..=127).map(|i| i as f32 / 127.0).collect();
        let (quantized, scale) = quantize_activations(&input);
        let recovered = dequantize_activations(&quantized, scale);

        // Maximum error should be small
        let max_error = input
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Error should be at most 1 quantization step
        assert!(max_error <= scale + 1e-6, "Max error {} > {}", max_error, scale);
    }

    #[test]
    fn test_quantize_block_size() {
        // Test with typical block size (128 elements)
        let input: Vec<f32> = (0..128)
            .map(|i| (i as f32 - 64.0) / 64.0)
            .collect();
        let (quantized, scale) = quantize_activations(&input);

        assert_eq!(quantized.len(), 128);
        // Check that extreme values are correct
        assert!(quantized.iter().any(|&x| x == 127 || x == -127));
        // Verify scale
        assert!(scale > 0.0);
    }

    #[test]
    fn test_quantize_preserves_sign() {
        let input = vec![-1.0, -0.5, -0.25, 0.25, 0.5, 1.0];
        let (quantized, _scale) = quantize_activations(&input);

        // Negative inputs -> negative outputs
        assert!(quantized[0] < 0);
        assert!(quantized[1] < 0);
        assert!(quantized[2] < 0);
        // Positive inputs -> positive outputs
        assert!(quantized[3] > 0);
        assert!(quantized[4] > 0);
        assert!(quantized[5] > 0);
    }

    #[test]
    fn test_quantize_saturation() {
        // Values beyond range should saturate to [-127, 127]
        // (This shouldn't happen with proper scaling, but test the clamp)
        let input = vec![1000.0, -1000.0];
        let (quantized, _scale) = quantize_activations(&input);

        // Should saturate to extremes
        assert_eq!(quantized[0], 127);
        assert_eq!(quantized[1], -127);
    }
}
