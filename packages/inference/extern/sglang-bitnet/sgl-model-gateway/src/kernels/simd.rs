//! Pure Rust SIMD helpers for operations not in the C++ kernels.
//!
//! These are used for auxiliary operations like RMSNorm, softmax, etc.

/// RMS normalization (pure Rust, SIMD-friendly)
///
/// output[i] = input[i] / sqrt(mean(input^2) + eps)
pub fn rms_norm(input: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute mean of squared values
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / n as f32;
    let rms = (mean_sq + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Normalize
    input.iter().map(|x| x * inv_rms).collect()
}

/// RMS normalization in place
pub fn rms_norm_inplace(data: &mut [f32], eps: f32) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let sum_sq: f32 = data.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    for x in data.iter_mut() {
        *x *= inv_rms;
    }
}

/// Apply RMS norm with learned scale (gamma)
pub fn rms_norm_with_scale(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(input.len(), gamma.len());

    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    input
        .iter()
        .zip(gamma.iter())
        .map(|(x, g)| x * inv_rms * g)
        .collect()
}

/// SiLU (Swish) activation: x * sigmoid(x)
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SiLU activation for slice
pub fn silu_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = *x / (1.0 + (-*x).exp());
    }
}

/// Softmax over a slice
pub fn softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let exp_vals: Vec<f32> = input.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    // Normalize
    exp_vals.iter().map(|x| x / sum).collect()
}

/// Softmax in place
pub fn softmax_inplace(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }

    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for x in data.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }

    let inv_sum = 1.0 / sum;
    for x in data.iter_mut() {
        *x *= inv_sum;
    }
}

/// Argmax of a slice
pub fn argmax(data: &[f32]) -> usize {
    if data.is_empty() {
        return 0;
    }

    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Element-wise multiply
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x *= y;
    }
}

/// Element-wise add
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

/// Dot product
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Scale a vector in place
pub fn scale_inplace(data: &mut [f32], factor: f32) {
    for x in data.iter_mut() {
        *x *= factor;
    }
}

/// Rope positional encoding
/// Applies rotary position embedding to Q or K vectors
pub fn apply_rope(data: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    let half_dim = head_dim / 2;

    for i in 0..half_dim {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let cos = angle.cos();
        let sin = angle.sin();

        let x0 = data[i];
        let x1 = data[i + half_dim];

        data[i] = x0 * cos - x1 * sin;
        data[i + half_dim] = x0 * sin + x1 * cos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = rms_norm(&input, 1e-6);

        // Check that output has same length
        assert_eq!(output.len(), input.len());

        // Check that RMS of output is approximately 1
        let rms: f32 = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let output = softmax(&input);

        // Check sum is 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check ordering preserved
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_argmax() {
        let data = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(argmax(&data), 1);
    }

    #[test]
    fn test_silu() {
        // silu(0) = 0
        assert!((silu(0.0)).abs() < 1e-6);

        // silu(x) > 0 for x > 0
        assert!(silu(1.0) > 0.0);

        // silu(x) < 0 for x < 0 (slightly)
        assert!(silu(-1.0) < 0.0);
    }
}
