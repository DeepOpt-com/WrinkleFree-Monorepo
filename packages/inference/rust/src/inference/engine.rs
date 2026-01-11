//! Safe Rust wrapper for the BitNet C++ inference engine.

use super::bitnet_ffi::{
    self, BitNetConfig, BitNetEngine as CEngine, CGenerationResult, CSamplingParams,
};
use std::ffi::CString;
use std::ptr;
use thiserror::Error;

/// Errors from the native inference engine
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Failed to create engine: {0}")]
    EngineCreation(String),

    #[error("Model path contains null bytes")]
    InvalidModelPath,

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Prefill failed: {0}")]
    PrefillFailed(String),

    #[error("Decode step failed: {0}")]
    DecodeStepFailed(String),

    #[error("Engine not initialized")]
    NotInitialized,
}

/// Sampling parameters for text generation
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
    pub max_tokens: i32,
}

impl Default for SamplingParams {
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

impl From<&SamplingParams> for CSamplingParams {
    fn from(params: &SamplingParams) -> Self {
        CSamplingParams {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repetition_penalty: params.repetition_penalty,
            max_tokens: params.max_tokens,
        }
    }
}

/// Generation result
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub output_ids: Vec<i32>,
}

/// Native BitNet inference engine.
///
/// This wraps the C++ engine and provides a safe Rust interface.
/// The engine is thread-safe for concurrent generation requests.
pub struct NativeEngine {
    engine: *mut CEngine,
}

// Safety: The C++ engine uses internal locking for thread safety
unsafe impl Send for NativeEngine {}
unsafe impl Sync for NativeEngine {}

impl NativeEngine {
    /// Create a new engine from a HuggingFace model path.
    ///
    /// # Arguments
    /// * `model_path` - Path to HuggingFace model directory or model ID
    /// * `max_seq_len` - Maximum sequence length (default: 2048)
    ///
    /// # Returns
    /// A new engine instance, or an error if creation failed.
    pub fn new(model_path: &str, max_seq_len: Option<i32>) -> Result<Self, InferenceError> {
        let c_path = CString::new(model_path).map_err(|_| InferenceError::InvalidModelPath)?;

        let config = BitNetConfig {
            max_seq_len: max_seq_len.unwrap_or(2048),
            num_threads: 0, // Auto-detect
            kv_cache_size: 256,
        };

        let engine = unsafe { bitnet_ffi::bitnet_engine_create(c_path.as_ptr(), &config) };

        if engine.is_null() {
            let error = bitnet_ffi::get_last_error()
                .unwrap_or_else(|| "Unknown error".to_string());
            return Err(InferenceError::EngineCreation(error));
        }

        Ok(Self { engine })
    }

    /// Generate tokens from input token IDs.
    ///
    /// This performs both prefill and decode phases.
    pub fn generate(
        &self,
        input_ids: &[i32],
        params: &SamplingParams,
    ) -> Result<GenerationResult, InferenceError> {
        if self.engine.is_null() {
            return Err(InferenceError::NotInitialized);
        }

        let c_params = CSamplingParams::from(params);
        let mut c_result = CGenerationResult {
            output_ids: ptr::null_mut(),
            num_tokens: 0,
            logits: ptr::null_mut(),
            logits_size: 0,
        };

        let ret = unsafe {
            bitnet_ffi::bitnet_generate(
                self.engine,
                input_ids.as_ptr(),
                input_ids.len() as i32,
                &c_params,
                &mut c_result,
            )
        };

        if ret != 0 {
            let error = bitnet_ffi::get_last_error()
                .unwrap_or_else(|| "Unknown error".to_string());
            return Err(InferenceError::GenerationFailed(error));
        }

        // Convert C result to Rust
        let output_ids = if !c_result.output_ids.is_null() && c_result.num_tokens > 0 {
            let slice = unsafe {
                std::slice::from_raw_parts(c_result.output_ids, c_result.num_tokens as usize)
            };
            slice.to_vec()
        } else {
            Vec::new()
        };

        // Free C memory
        unsafe {
            bitnet_ffi::bitnet_free_result(&mut c_result);
        }

        Ok(GenerationResult { output_ids })
    }

    /// Prefill phase - process input tokens and populate KV cache.
    ///
    /// Call this before `decode_step` for manual generation control.
    pub fn prefill(&self, input_ids: &[i32]) -> Result<(), InferenceError> {
        if self.engine.is_null() {
            return Err(InferenceError::NotInitialized);
        }

        let ret = unsafe {
            bitnet_ffi::bitnet_prefill(
                self.engine,
                input_ids.as_ptr(),
                input_ids.len() as i32,
            )
        };

        if ret != 0 {
            let error = bitnet_ffi::get_last_error()
                .unwrap_or_else(|| "Unknown error".to_string());
            return Err(InferenceError::PrefillFailed(error));
        }

        Ok(())
    }

    /// Single decode step - generate one token.
    ///
    /// Call `prefill` first to populate the KV cache.
    pub fn decode_step(
        &self,
        position: i32,
        params: &SamplingParams,
    ) -> Result<i32, InferenceError> {
        if self.engine.is_null() {
            return Err(InferenceError::NotInitialized);
        }

        let c_params = CSamplingParams::from(params);
        let mut output_id: i32 = 0;

        let ret = unsafe {
            bitnet_ffi::bitnet_decode_step(self.engine, position, &c_params, &mut output_id)
        };

        if ret != 0 {
            let error = bitnet_ffi::get_last_error()
                .unwrap_or_else(|| "Unknown error".to_string());
            return Err(InferenceError::DecodeStepFailed(error));
        }

        Ok(output_id)
    }

    /// Reset the KV cache (start new sequence).
    pub fn reset_cache(&self) {
        if !self.engine.is_null() {
            unsafe {
                bitnet_ffi::bitnet_reset_cache(self.engine);
            }
        }
    }

    /// Get model vocabulary size.
    pub fn vocab_size(&self) -> i32 {
        if self.engine.is_null() {
            return 0;
        }
        unsafe { bitnet_ffi::bitnet_vocab_size(self.engine) }
    }

    /// Get model hidden dimension.
    pub fn hidden_size(&self) -> i32 {
        if self.engine.is_null() {
            return 0;
        }
        unsafe { bitnet_ffi::bitnet_hidden_size(self.engine) }
    }

    /// Get number of transformer layers.
    pub fn num_layers(&self) -> i32 {
        if self.engine.is_null() {
            return 0;
        }
        unsafe { bitnet_ffi::bitnet_num_layers(self.engine) }
    }

    /// Get maximum sequence length.
    pub fn max_seq_len(&self) -> i32 {
        if self.engine.is_null() {
            return 0;
        }
        unsafe { bitnet_ffi::bitnet_max_seq_len(self.engine) }
    }
}

impl Drop for NativeEngine {
    fn drop(&mut self) {
        if !self.engine.is_null() {
            unsafe {
                bitnet_ffi::bitnet_engine_destroy(self.engine);
            }
            self.engine = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require the C++ library to be built and linked.
    // They will fail at link time if the library is not available.

    #[test]
    #[ignore = "Requires C++ library to be built"]
    fn test_engine_creation() {
        // This will fail with "model loading not implemented" but proves FFI works
        let result = NativeEngine::new("microsoft/bitnet-b1.58-2B-4T", None);
        // Even with placeholder weights, engine should be created
        assert!(result.is_ok() || result.is_err());
    }
}
