//! FFI bindings to the C++ BitNet inference engine.
//!
//! These bindings match the C API defined in bitnet_engine.h

use libc::{c_char, c_float, c_int, c_void};
use std::ffi::CStr;

/// Opaque engine handle
#[repr(C)]
pub struct BitNetEngine {
    _private: [u8; 0],
}

/// Engine configuration
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct BitNetConfig {
    pub max_seq_len: i32,
    pub num_threads: i32,
    pub kv_cache_size: i32,
}

/// Sampling parameters
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CSamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
    pub max_tokens: i32,
}

impl Default for CSamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            max_tokens: 256,
        }
    }
}

/// Generation result from C++
#[repr(C)]
pub struct CGenerationResult {
    pub output_ids: *mut i32,
    pub num_tokens: i32,
    pub logits: *mut f32,
    pub logits_size: i32,
}

// Link to the C++ library
#[link(name = "sgl_kernel_inference")]
extern "C" {
    // Engine lifecycle
    pub fn bitnet_engine_create(
        model_path: *const c_char,
        config: *const BitNetConfig,
    ) -> *mut BitNetEngine;

    pub fn bitnet_engine_destroy(engine: *mut BitNetEngine);

    pub fn bitnet_get_error() -> *const c_char;

    // Inference
    pub fn bitnet_generate(
        engine: *mut BitNetEngine,
        input_ids: *const i32,
        num_input_tokens: i32,
        params: *const CSamplingParams,
        result: *mut CGenerationResult,
    ) -> c_int;

    pub fn bitnet_prefill(
        engine: *mut BitNetEngine,
        input_ids: *const i32,
        num_tokens: i32,
    ) -> c_int;

    pub fn bitnet_decode_step(
        engine: *mut BitNetEngine,
        position: i32,
        params: *const CSamplingParams,
        output_id: *mut i32,
    ) -> c_int;

    pub fn bitnet_reset_cache(engine: *mut BitNetEngine);

    // Model info
    pub fn bitnet_vocab_size(engine: *mut BitNetEngine) -> i32;
    pub fn bitnet_hidden_size(engine: *mut BitNetEngine) -> i32;
    pub fn bitnet_num_layers(engine: *mut BitNetEngine) -> i32;
    pub fn bitnet_max_seq_len(engine: *mut BitNetEngine) -> i32;

    // Memory management
    pub fn bitnet_free_result(result: *mut CGenerationResult);
}

/// Get the last error message from the C++ engine
pub fn get_last_error() -> Option<String> {
    unsafe {
        let ptr = bitnet_get_error();
        if ptr.is_null() {
            return None;
        }
        let c_str = CStr::from_ptr(ptr);
        Some(c_str.to_string_lossy().into_owned())
    }
}
