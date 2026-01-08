//! Native BitNet kernel bindings.
//!
//! This module provides Rust FFI bindings to the optimized C++ BitNet kernels
//! in sgl-kernel/csrc/bitnet/.
//!
//! ## Features
//! - AVX-512/AVX2/NEON SIMD acceleration
//! - Ternary weight computation (no multiply, just add/subtract)
//! - INT8 activation quantization
//!
//! ## Usage
//! ```ignore
//! use sgl_model_gateway::kernels::{BitNetKernel, CPUCapabilities};
//!
//! let caps = CPUCapabilities::detect();
//! println!("AVX-512: {}", caps.has_avx512);
//!
//! let kernel = BitNetKernel::new();
//! let result = kernel.vec_dot(&weights, &activations);
//! ```

pub mod ffi;
pub mod simd;

pub use ffi::*;
pub use simd::*;
