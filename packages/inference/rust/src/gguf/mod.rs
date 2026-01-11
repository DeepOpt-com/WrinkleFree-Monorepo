//! Pure Rust GGUF file reader.
//!
//! This module implements a memory-mapped GGUF parser that can load models
//! without depending on llama.cpp for model loading.
//!
//! ## Features
//! - Memory-mapped file access for efficient large model loading
//! - Support for all GGUF value types and metadata
//! - Support for quantized tensor formats (I2_S, TQ1_0, F16, etc.)
//! - Weight repacking for native BitNet kernels
//!
//! ## Usage
//! ```ignore
//! use sgl_model_gateway::gguf::GgufReader;
//!
//! let reader = GgufReader::open("model.gguf")?;
//! println!("Architecture: {:?}", reader.architecture());
//! println!("Vocab size: {}", reader.get_u32("llama.vocab_size")?);
//!
//! for tensor in reader.tensors() {
//!     println!("Tensor: {} {:?} {:?}", tensor.name, tensor.shape, tensor.dtype);
//! }
//! ```

pub mod types;
pub mod reader;
pub mod repack;
pub mod tokenizer;

// Re-export commonly used types
pub use types::{
    GgufValueType, GgmlQuantType, GgufValue, GgufTensorInfo, ModelConfig, GgufError,
    GGUF_MAGIC, GGUF_VERSION, QK_I2_S_NATIVE,
};
pub use reader::GgufReader;
pub use repack::{repack_ternary_weights, NativeWeightFormat};
pub use tokenizer::Tokenizer;
