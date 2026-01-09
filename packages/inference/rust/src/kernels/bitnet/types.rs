//! BitNet kernel type definitions.
//!
//! This module defines constants and types for the pure Rust BitNet kernels.
//! Target: ARM NEON (aarch64) with scalar fallback for testing.

/// Block size for I2_S quantization.
/// Must match training block size (128 elements per block).
pub const QK_BLOCK: usize = 128;

/// Bytes per block (4 weights per byte, 2 bits each).
pub const BLOCK_BYTES: usize = QK_BLOCK / 4; // 32 bytes

/// Tile configuration for cache-optimized GEMM.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Output tile size (rows per tile).
    pub bm: usize,
    /// Input tile size (blocks per tile).
    pub bk: usize,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self { bm: 256, bk: 32 }
    }
}

impl TileConfig {
    /// Create a tile config optimized for the given dimensions.
    pub fn for_dimensions(m: usize, k: usize) -> Self {
        // Simple heuristic: smaller tiles for smaller matrices
        let bm = if m < 256 { m.max(16) } else { 256 };
        let k_blocks = k / QK_BLOCK;
        let bk = if k_blocks < 32 { k_blocks.max(1) } else { 32 };
        Self { bm, bk }
    }
}

/// CPU capability flags for runtime dispatch.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuCapabilities {
    /// ARM NEON (always true on aarch64).
    pub has_neon: bool,
    /// ARM v8.2 dot product extension.
    pub has_dotprod: bool,
    /// x86 AVX2 (for scalar fallback detection).
    pub has_avx2: bool,
    /// x86 AVX-512 (not used, for compatibility).
    pub has_avx512: bool,
}

impl CpuCapabilities {
    /// Detect CPU capabilities at runtime.
    #[cfg(target_arch = "aarch64")]
    pub fn detect() -> Self {
        Self {
            has_neon: true, // Always available on aarch64
            has_dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
            has_avx2: false,
            has_avx512: false,
        }
    }

    /// Detect CPU capabilities at runtime (x86 fallback).
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        Self {
            has_neon: false,
            has_dotprod: false,
            has_avx2: is_x86_feature_detected!("avx2"),
            has_avx512: is_x86_feature_detected!("avx512f"),
        }
    }

    /// Fallback for other architectures.
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    pub fn detect() -> Self {
        Self::default()
    }

    /// Get a human-readable description of capabilities.
    pub fn description(&self) -> String {
        let mut caps = Vec::new();
        if self.has_neon {
            caps.push("NEON");
        }
        if self.has_dotprod {
            caps.push("DotProd");
        }
        if self.has_avx512 {
            caps.push("AVX-512");
        }
        if self.has_avx2 {
            caps.push("AVX2");
        }
        if caps.is_empty() {
            "Scalar (no SIMD)".to_string()
        } else {
            caps.join(", ")
        }
    }

    /// Check if we have ARM SIMD support.
    pub fn has_arm_simd(&self) -> bool {
        self.has_neon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(QK_BLOCK, 128);
        assert_eq!(BLOCK_BYTES, 32);
        assert_eq!(QK_BLOCK / 4, BLOCK_BYTES);
    }

    #[test]
    fn test_tile_config_default() {
        let config = TileConfig::default();
        assert_eq!(config.bm, 256);
        assert_eq!(config.bk, 32);
    }

    #[test]
    fn test_tile_config_for_small_matrix() {
        let config = TileConfig::for_dimensions(64, 256);
        assert_eq!(config.bm, 64);
        assert_eq!(config.bk, 2); // 256 / 128 = 2 blocks
    }

    #[test]
    fn test_tile_config_for_large_matrix() {
        let config = TileConfig::for_dimensions(8192, 4096);
        assert_eq!(config.bm, 256);
        assert_eq!(config.bk, 32);
    }

    #[test]
    fn test_cpu_capabilities_detect() {
        let caps = CpuCapabilities::detect();
        // Just verify it doesn't panic
        let desc = caps.description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn test_cpu_capabilities_description() {
        let caps = CpuCapabilities {
            has_neon: true,
            has_dotprod: true,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(caps.description().contains("NEON"));
        assert!(caps.description().contains("DotProd"));
    }

    #[test]
    fn test_cpu_capabilities_scalar_fallback() {
        let caps = CpuCapabilities::default();
        assert_eq!(caps.description(), "Scalar (no SIMD)");
    }
}
