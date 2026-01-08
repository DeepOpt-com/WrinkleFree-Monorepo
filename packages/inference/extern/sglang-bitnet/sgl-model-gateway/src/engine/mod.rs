//! Native BitNet inference engine.
//!
//! This module implements a pure Rust transformer inference engine
//! that uses the native sgl-kernel BitNet SIMD kernels for linear layers.
//!
//! ## Features
//! - GGUF model loading (no llama.cpp dependency)
//! - Native ternary SIMD kernels (AVX-512/AVX2/NEON)
//! - KV cache management
//! - Continuous batching support
//!
//! ## Usage
//! ```ignore
//! use sgl_model_gateway::engine::BitNetEngine;
//!
//! let engine = BitNetEngine::load("model.gguf")?;
//! let output_ids = engine.generate(&input_ids, 64)?;
//! ```

mod kv_cache;
mod model;
mod sampling;

pub use kv_cache::{KVCache, KVCacheConfig};
pub use model::{BitNetEngine, BitNetConfig, LayerWeights};
pub use sampling::{SamplingConfig, sample_token, top_p_sampling, top_k_sampling};
