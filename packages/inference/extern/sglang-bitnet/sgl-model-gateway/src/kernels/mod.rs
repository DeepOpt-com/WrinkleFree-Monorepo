//! BitNet inference kernels.
//!
//! This module provides pure Rust SIMD-optimized kernels for BitNet inference.
//! Primary target: ARM NEON (aarch64) with scalar fallback for other platforms.
//!
//! ## Features
//! - ARM NEON SIMD acceleration (with optional dotprod extension)
//! - Ternary weight computation (packed I2_S format)
//! - INT8 activation quantization
//! - Rayon parallelization for batched operations
//!
//! ## Usage
//! ```ignore
//! use sgl_model_gateway::kernels::{BitNetKernel, CpuCapabilities};
//!
//! let caps = CpuCapabilities::detect();
//! println!("NEON: {}", caps.has_neon);
//!
//! let kernel = BitNetKernel::new();
//! let (quantized, scale) = kernel.quantize_activations(&input);
//! let result = kernel.vec_dot(&weights, &quantized);
//! ```

pub mod bitnet;
pub mod simd;

// Re-export main types from bitnet module
pub use bitnet::{
    BitNetKernel,
    CpuCapabilities,
    TileConfig,
    QK_BLOCK,
    quantize_activations,
    pack_weights,
    is_simd_available,
};

// Keep simd module exports for backward compatibility
pub use simd::*;

// Compatibility alias for old code using CPUCapabilities (capital U)
pub type CPUCapabilities = CpuCapabilities;
