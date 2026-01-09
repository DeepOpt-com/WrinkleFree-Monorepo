//! WrinkleFree Inference Engine
//!
//! Rust inference engine for BitNet 1.58-bit quantized LLMs with DLM block diffusion support.
//!
//! # Features
//!
//! - **native-inference**: Enable pure Rust BitNet inference with SIMD kernels
//! - **llama-inference**: Enable llama.cpp-based inference for DLM block diffusion

// Core modules (always available)
pub mod config;
pub mod observability;
pub mod protocols;
pub mod tokenizer;
pub mod version;

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
