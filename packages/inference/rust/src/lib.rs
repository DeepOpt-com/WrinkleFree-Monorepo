//! WrinkleFree Inference Engine
//!
//! Rust inference engine for BitNet 1.58-bit quantized LLMs with DLM block diffusion support.
//!
//! # Features
//!
//! - **native-inference**: Enable pure Rust BitNet inference with SIMD kernels
//! - **llama-inference**: Enable llama.cpp-based inference for DLM block diffusion

// Enable ARMv8.2+ dotprod intrinsics (requires nightly Rust)
#![cfg_attr(
    all(target_arch = "aarch64", feature = "native-inference"),
    feature(stdarch_neon_dotprod)
)]

// Pure Rust GGUF reader (no C++ dependency)
pub mod gguf;

// Native BitNet kernels (feature-gated)
#[cfg(feature = "native-inference")]
pub mod kernels;

// Native BitNet inference engine (feature-gated)
#[cfg(feature = "native-inference")]
pub mod engine;

// C++ llama.cpp-based inference for DLM (feature-gated)
#[cfg(feature = "llama-inference")]
pub mod inference;
