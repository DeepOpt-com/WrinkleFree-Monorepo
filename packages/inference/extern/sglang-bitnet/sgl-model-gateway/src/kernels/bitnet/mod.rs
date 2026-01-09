//! Pure Rust BitNet kernels for ARM NEON.
//!
//! This module provides SIMD-optimized ternary weight × int8 activation
//! kernels for BitNet inference on ARM64 processors.
//!
//! # Target Architecture
//! - Primary: ARM64 with NEON (aarch64)
//! - Fallback: Scalar implementation for other architectures
//!
//! # Features
//! - Packed I2_S weight format (2-bit ternary encoding)
//! - INT8 activation quantization
//! - Batched GEMM with Rayon parallelization
//! - Optional dotprod extension support (ARMv8.2+)
//!
//! # Example
//! ```ignore
//! use sgl_model_gateway::kernels::bitnet::BitNetKernel;
//!
//! let kernel = BitNetKernel::new();
//! println!("CPU: {}", kernel.capabilities.description());
//!
//! let (quantized, scale) = kernel.quantize_activations(&input);
//! let result = kernel.vec_dot(&packed_weights, &quantized);
//! ```

mod types;
mod quantize;
mod gemv_scalar;
mod gemv_neon;
mod gemm;

pub use types::{QK_BLOCK, BLOCK_BYTES, TileConfig, CpuCapabilities};
pub use quantize::{quantize_activations, dequantize_activations};
pub use gemv_scalar::{vec_dot_scalar, pack_weights, unpack_weights};
pub use gemv_neon::vec_dot_neon;
pub use gemm::gemm;

/// Pure Rust BitNet kernel wrapper.
///
/// Provides a clean interface to the underlying SIMD kernels.
/// Automatically dispatches to the best available implementation
/// based on CPU capabilities.
pub struct BitNetKernel {
    /// Detected CPU capabilities.
    pub capabilities: CpuCapabilities,
    /// Tile configuration for cache optimization.
    pub tile_config: TileConfig,
}

impl BitNetKernel {
    /// Create a new kernel with auto-detected capabilities.
    pub fn new() -> Self {
        Self {
            capabilities: CpuCapabilities::detect(),
            tile_config: TileConfig::default(),
        }
    }

    /// Create with custom tile configuration.
    pub fn with_tile_config(tile_config: TileConfig) -> Self {
        Self {
            capabilities: CpuCapabilities::detect(),
            tile_config,
        }
    }

    /// Create with auto-tuned tile configuration for typical dimensions.
    pub fn with_auto_tune(typical_m: usize, typical_k: usize) -> Self {
        Self {
            capabilities: CpuCapabilities::detect(),
            tile_config: TileConfig::for_dimensions(typical_m, typical_k),
        }
    }

    /// Quantize FP32 activations to INT8.
    ///
    /// # Returns
    /// (quantized_values, scale_factor)
    pub fn quantize_activations(&self, input: &[f32]) -> (Vec<i8>, f32) {
        quantize_activations(input)
    }

    /// Compute dot product of packed weights and activations.
    ///
    /// Uses ARM NEON when available, scalar fallback otherwise.
    ///
    /// # Arguments
    /// * `packed_weights` - Packed I2_S weights (k/4 bytes)
    /// * `activations` - INT8 activations (k elements, multiple of 128)
    ///
    /// # Returns
    /// Dot product result as f32.
    pub fn vec_dot(&self, packed_weights: &[u8], activations: &[i8]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            // Use NEON SIMD on ARM64
            // NOTE: dotprod extension (vdotq_s32) would be faster but requires nightly Rust
            vec_dot_neon(packed_weights, activations) as f32
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            vec_dot_scalar(packed_weights, activations) as f32
        }
    }

    /// Compute batched GEMM: output = packed_weights @ activations * scale.
    ///
    /// # Arguments
    /// * `m` - Number of output features (rows)
    /// * `n` - Batch size (columns)
    /// * `k` - Number of input features (must be multiple of 128)
    /// * `packed_weights` - Packed weights [M, K/4]
    /// * `activations` - INT8 activations [K, N] column-major
    /// * `scale` - Weight scale factor
    ///
    /// # Returns
    /// Output matrix [M, N] row-major.
    pub fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        packed_weights: &[u8],
        activations: &[i8],
        scale: f32,
    ) -> Vec<f32> {
        gemm(m, n, k, packed_weights, activations, scale, &self.tile_config)
    }
}

impl Default for BitNetKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if the current platform has SIMD support.
pub fn is_simd_available() -> bool {
    let caps = CpuCapabilities::detect();
    caps.has_neon || caps.has_avx2 || caps.has_avx512
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        let kernel = BitNetKernel::new();
        let desc = kernel.capabilities.description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn test_kernel_quantize() {
        let kernel = BitNetKernel::new();
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0];
        let (quantized, scale) = kernel.quantize_activations(&input);

        assert_eq!(quantized.len(), 5);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_kernel_vec_dot() {
        let kernel = BitNetKernel::new();

        let weights = vec![1i8; 128];
        let packed = pack_weights(&weights);
        let activations = vec![1i8; 128];

        let result = kernel.vec_dot(&packed, &activations);
        assert_eq!(result, 128.0);
    }

    #[test]
    fn test_kernel_gemm() {
        let kernel = BitNetKernel::new();

        let m = 4;
        let n = 2;
        let k = 128;

        let weights: Vec<i8> = (0..m * k).map(|_| 1).collect();
        let mut packed_weights = Vec::new();
        for row in 0..m {
            packed_weights.extend(pack_weights(&weights[row * k..(row + 1) * k]));
        }

        let activations = vec![1i8; k * n];
        let scale = 1.0;

        let result = kernel.gemm(m, n, k, &packed_weights, &activations, scale);

        assert_eq!(result.len(), m * n);
        for val in result {
            assert_eq!(val, k as f32);
        }
    }

    #[test]
    fn test_is_simd_available() {
        // This should work on any platform
        let _ = is_simd_available();
    }

    #[test]
    fn test_kernel_auto_tune() {
        let kernel = BitNetKernel::with_auto_tune(4096, 2048);
        assert!(kernel.tile_config.bm > 0);
        assert!(kernel.tile_config.bk > 0);
    }
}
