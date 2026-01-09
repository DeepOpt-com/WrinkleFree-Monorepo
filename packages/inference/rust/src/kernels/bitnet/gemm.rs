//! Batched GEMM with Rayon parallelization.
//!
//! Implements: output = weights @ activations * scale
//! Where weights are packed I2_S ternary and activations are INT8.

use rayon::prelude::*;
use super::types::{QK_BLOCK, TileConfig};

// Use multiply-free ternary kernel for all platforms
use super::gemv_ternary::vec_dot_ternary_neon as vec_dot;

/// BitNet GEMM: output = packed_weights @ activations * scale
///
/// # Arguments
/// * `m` - Number of output features (rows of weights)
/// * `n` - Batch size (columns of activations)
/// * `k` - Number of input features (must be multiple of 128)
/// * `packed_weights` - Packed I2_S weights [M, K/4]
/// * `activations` - INT8 activations [K, N] in column-major layout
/// * `scale` - Weight scale factor
/// * `config` - Tile configuration for cache optimization
///
/// # Returns
/// Output matrix [M, N] in row-major layout.
///
/// # Layout
/// - packed_weights: Row-major [M, K/4] - each row is K/4 bytes
/// - activations: Column-major [K, N] - contiguous columns for better cache
/// - output: Row-major [M, N]
///
/// # Parallelization
/// Uses Rayon to parallelize over output rows for large matrices.
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    packed_weights: &[u8],
    activations: &[i8],
    scale: f32,
    _config: &TileConfig,
) -> Vec<f32> {
    assert!(k % QK_BLOCK == 0, "k must be multiple of {}", QK_BLOCK);
    let k_packed = k / 4;
    assert!(
        packed_weights.len() >= m * k_packed,
        "packed_weights too short: {} < {}",
        packed_weights.len(),
        m * k_packed
    );
    assert!(
        activations.len() >= k * n,
        "activations too short: {} < {}",
        activations.len(),
        k * n
    );

    let mut output = vec![0.0f32; m * n];

    if n == 1 {
        // Single column - simple GEMV, parallelize over rows
        gemv_parallel(m, k, packed_weights, activations, scale, &mut output);
    } else {
        // Multiple columns - parallelize over rows
        gemm_parallel(m, n, k, packed_weights, activations, scale, &mut output);
    }

    output
}

/// Single-column GEMV with row parallelization.
fn gemv_parallel(
    _m: usize,
    k: usize,
    packed_weights: &[u8],
    activations: &[i8],
    scale: f32,
    output: &mut [f32],
) {
    let k_packed = k / 4;

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let w_start = row * k_packed;
        let w_row = &packed_weights[w_start..w_start + k_packed];

        let dot = vec_dot(w_row, activations);
        *out = (dot as f32) * scale;
    });
}

/// Multi-column GEMM with row parallelization.
fn gemm_parallel(
    _m: usize,
    n: usize,
    k: usize,
    packed_weights: &[u8],
    activations: &[i8],
    scale: f32,
    output: &mut [f32],
) {
    let k_packed = k / 4;

    // Parallelize over rows
    output
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(row, out_row)| {
            let w_start = row * k_packed;
            let w_row = &packed_weights[w_start..w_start + k_packed];

            for col in 0..n {
                // Extract column from activations (column-major layout)
                // activations[i, col] = activations[col * k + i]
                let a_col_start = col * k;
                let a_col = &activations[a_col_start..a_col_start + k];

                let dot = vec_dot(w_row, a_col);
                out_row[col] = (dot as f32) * scale;
            }
        });
}

/// GEMM with row-major activations (alternative layout).
///
/// Use this when activations are stored in row-major [N, K] format.
/// This requires copying each column, so column-major is preferred.
#[allow(dead_code)]
pub fn gemm_row_major_activations(
    m: usize,
    n: usize,
    k: usize,
    packed_weights: &[u8],
    activations: &[i8], // [N, K] row-major
    scale: f32,
    config: &TileConfig,
) -> Vec<f32> {
    // Convert row-major to column-major
    let mut col_major = vec![0i8; k * n];
    for col in 0..n {
        for i in 0..k {
            // activations[col, i] in row-major = activations[col * k + i]
            // col_major[i, col] in col-major = col_major[col * k + i]
            col_major[col * k + i] = activations[col * k + i];
        }
    }

    gemm(m, n, k, packed_weights, &col_major, scale, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::gemv_scalar::{pack_weights, vec_dot_scalar};

    fn compute_gemm_reference(
        m: usize,
        n: usize,
        k: usize,
        packed_weights: &[u8],
        activations: &[i8],
        scale: f32,
    ) -> Vec<f32> {
        let k_packed = k / 4;
        let mut output = vec![0.0f32; m * n];

        for row in 0..m {
            let w_start = row * k_packed;
            let w_row = &packed_weights[w_start..w_start + k_packed];

            for col in 0..n {
                let a_start = col * k;
                let a_col = &activations[a_start..a_start + k];

                let dot = vec_dot_scalar(w_row, a_col);
                output[row * n + col] = (dot as f32) * scale;
            }
        }

        output
    }

    #[test]
    fn test_gemv_single_column() {
        let m = 4;
        let k = 128;
        let n = 1;

        // Create simple weights and activations
        let weights: Vec<i8> = (0..m * k).map(|i| ((i % 3) as i8) - 1).collect();
        let mut packed_weights = Vec::new();
        for row in 0..m {
            let row_weights = &weights[row * k..(row + 1) * k];
            packed_weights.extend(pack_weights(row_weights));
        }

        let activations = vec![1i8; k];
        let scale = 1.0;
        let config = TileConfig::default();

        let result = gemm(m, n, k, &packed_weights, &activations, scale, &config);
        let reference = compute_gemm_reference(m, n, k, &packed_weights, &activations, scale);

        assert_eq!(result.len(), m * n);
        for i in 0..m {
            assert!(
                (result[i] - reference[i]).abs() < 1e-6,
                "Mismatch at {}: {} vs {}",
                i,
                result[i],
                reference[i]
            );
        }
    }

    #[test]
    fn test_gemm_multiple_columns() {
        let m = 4;
        let k = 128;
        let n = 3;

        let weights: Vec<i8> = (0..m * k).map(|i| ((i % 3) as i8) - 1).collect();
        let mut packed_weights = Vec::new();
        for row in 0..m {
            let row_weights = &weights[row * k..(row + 1) * k];
            packed_weights.extend(pack_weights(row_weights));
        }

        // Column-major activations (use i32 intermediate to avoid overflow)
        let activations: Vec<i8> = (0..k * n)
            .map(|i| (((i * 7) % 255) as i32 - 127) as i8)
            .collect();
        let scale = 0.5;
        let config = TileConfig::default();

        let result = gemm(m, n, k, &packed_weights, &activations, scale, &config);
        let reference = compute_gemm_reference(m, n, k, &packed_weights, &activations, scale);

        assert_eq!(result.len(), m * n);
        for i in 0..m * n {
            assert!(
                (result[i] - reference[i]).abs() < 1e-6,
                "Mismatch at {}: {} vs {}",
                i,
                result[i],
                reference[i]
            );
        }
    }

    #[test]
    fn test_gemm_larger_matrix() {
        let m = 16;
        let k = 256;
        let n = 8;

        let weights: Vec<i8> = (0..m * k).map(|i| ((i * 11) % 3) as i8 - 1).collect();
        let mut packed_weights = Vec::new();
        for row in 0..m {
            let row_weights = &weights[row * k..(row + 1) * k];
            packed_weights.extend(pack_weights(row_weights));
        }

        let activations: Vec<i8> = (0..k * n)
            .map(|i| (((i * 13) % 255) as i32 - 127) as i8)
            .collect();
        let scale = 0.1;
        let config = TileConfig::default();

        let result = gemm(m, n, k, &packed_weights, &activations, scale, &config);
        let reference = compute_gemm_reference(m, n, k, &packed_weights, &activations, scale);

        assert_eq!(result.len(), m * n);
        for i in 0..m * n {
            assert!(
                (result[i] - reference[i]).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                result[i],
                reference[i]
            );
        }
    }

    #[test]
    fn test_gemm_all_ones() {
        let m = 2;
        let k = 128;
        let n = 2;

        // All +1 weights
        let weights = vec![1i8; m * k];
        let mut packed_weights = Vec::new();
        for row in 0..m {
            let row_weights = &weights[row * k..(row + 1) * k];
            packed_weights.extend(pack_weights(row_weights));
        }

        // All +1 activations
        let activations = vec![1i8; k * n];
        let scale = 1.0;
        let config = TileConfig::default();

        let result = gemm(m, n, k, &packed_weights, &activations, scale, &config);

        // Each output should be k (128 * 1 * 1)
        for i in 0..m * n {
            assert_eq!(result[i], k as f32, "Output[{}] = {}", i, result[i]);
        }
    }

    #[test]
    fn test_gemm_scale_factor() {
        let m = 2;
        let k = 128;
        let n = 1;

        let weights = vec![1i8; m * k];
        let mut packed_weights = Vec::new();
        for row in 0..m {
            packed_weights.extend(pack_weights(&weights[row * k..(row + 1) * k]));
        }

        let activations = vec![1i8; k];
        let config = TileConfig::default();

        // Test different scales
        for scale in [0.5, 1.0, 2.0, 0.001] {
            let result = gemm(m, n, k, &packed_weights, &activations, scale, &config);
            let expected = (k as f32) * scale;

            for i in 0..m {
                assert!(
                    (result[i] - expected).abs() < 1e-4,
                    "Scale {}: expected {}, got {}",
                    scale,
                    expected,
                    result[i]
                );
            }
        }
    }

    #[test]
    fn test_gemm_identity_like() {
        // Test with a pattern that resembles identity behavior
        let m = 4;
        let k = 128;
        let n = 4;

        // Each row has weights that only activate one column
        let mut weights = vec![0i8; m * k];
        for row in 0..m {
            let col_offset = row * (k / m);
            for i in 0..(k / m) {
                weights[row * k + col_offset + i] = 1;
            }
        }

        let mut packed_weights = Vec::new();
        for row in 0..m {
            packed_weights.extend(pack_weights(&weights[row * k..(row + 1) * k]));
        }

        let activations: Vec<i8> = (0..k * n).map(|i| ((i % k) as i8).min(100)).collect();
        let scale = 1.0;
        let config = TileConfig::default();

        let result = gemm(m, n, k, &packed_weights, &activations, scale, &config);

        assert_eq!(result.len(), m * n);
        // Just verify it runs without panicking for now
    }

    #[test]
    #[should_panic(expected = "k must be multiple of 128")]
    fn test_gemm_invalid_k() {
        let packed_weights = vec![0u8; 32];
        let activations = vec![0i8; 100]; // Not multiple of 128
        let config = TileConfig::default();
        gemm(1, 1, 100, &packed_weights, &activations, 1.0, &config);
    }

    #[test]
    fn test_gemm_parallel_consistency() {
        // Run the same computation multiple times to check for race conditions
        let m = 64;
        let k = 256;
        let n = 8;

        let weights: Vec<i8> = (0..m * k).map(|i| ((i * 17) % 3) as i8 - 1).collect();
        let mut packed_weights = Vec::new();
        for row in 0..m {
            packed_weights.extend(pack_weights(&weights[row * k..(row + 1) * k]));
        }

        let activations: Vec<i8> = (0..k * n).map(|i| (((i * 23) % 255) as i32 - 127) as i8).collect();
        let scale = 0.1;
        let config = TileConfig::default();

        // Run multiple times and check consistency
        let reference = gemm(m, n, k, &packed_weights, &activations, scale, &config);

        for _ in 0..5 {
            let result = gemm(m, n, k, &packed_weights, &activations, scale, &config);
            for i in 0..m * n {
                assert!(
                    (result[i] - reference[i]).abs() < 1e-6,
                    "Inconsistent result at {}",
                    i
                );
            }
        }
    }
}
