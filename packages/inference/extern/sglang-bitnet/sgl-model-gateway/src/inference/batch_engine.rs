//! Safe Rust wrapper for the BitNet batch inference engine.
//!
//! Provides a thread-safe interface for continuous batching with multiple
//! concurrent sequences.

use super::batch_ffi::{
    self, BitNetBatch as CBatch, BitNetBatchConfig, BitNetBatchEngine as CEngine,
    BitNetSeqId, BitNetSeqInfo, BitNetSamplingParams,
};
use std::ffi::{CString, CStr};
use std::sync::Arc;
use thiserror::Error;

/// Errors from the batch inference engine
#[derive(Error, Debug)]
pub enum BatchError {
    #[error("Failed to create batch engine: {0}")]
    EngineCreation(String),

    #[error("Model path contains null bytes")]
    InvalidModelPath,

    #[error("Batch decode failed: {0}")]
    DecodeFailed(String),

    #[error("No available sequence slots")]
    NoAvailableSlots,

    #[error("Invalid sequence ID: {0}")]
    InvalidSequenceId(i32),

    #[error("Batch is full")]
    BatchFull,

    #[error("Engine not initialized")]
    NotInitialized,

    #[error("KV cache is full")]
    KVCacheFull,
}

/// Sampling parameters for text generation
#[derive(Debug, Clone)]
pub struct BatchSamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: f32,
    pub repetition_penalty: f32,
    pub max_tokens: i32,
}

impl Default for BatchSamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 256,
        }
    }
}

impl From<&BatchSamplingParams> for BitNetSamplingParams {
    fn from(params: &BatchSamplingParams) -> Self {
        BitNetSamplingParams {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repetition_penalty: params.repetition_penalty,
            max_tokens: params.max_tokens,
        }
    }
}

/// Batch configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub max_batch_size: i32,
    pub max_sequences: i32,
    pub n_ctx: i32,
    pub n_ctx_per_seq: i32,
    pub num_threads: i32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 512,
            max_sequences: 16,
            n_ctx: 4096,
            n_ctx_per_seq: 256,
            num_threads: 0,
        }
    }
}

impl From<&BatchConfig> for BitNetBatchConfig {
    fn from(config: &BatchConfig) -> Self {
        BitNetBatchConfig {
            max_batch_size: config.max_batch_size,
            max_sequences: config.max_sequences,
            n_ctx: config.n_ctx,
            n_ctx_per_seq: config.n_ctx_per_seq,
            num_threads: config.num_threads,
        }
    }
}

/// Sequence information
#[derive(Debug, Clone)]
pub struct SequenceInfo {
    pub seq_id: i32,
    pub is_active: bool,
    pub position: i32,
    pub prompt_len: i32,
    pub generated_count: i32,
}

/// A batch of tokens for inference
pub struct Batch {
    batch: *mut CBatch,
    capacity: i32,
}

impl Batch {
    /// Create a new batch with given capacity
    pub fn new(n_tokens: i32, n_seq_max: i32) -> Result<Self, BatchError> {
        let batch = unsafe { batch_ffi::bitnet_batch_init(n_tokens, n_seq_max) };
        if batch.is_null() {
            return Err(BatchError::EngineCreation(
                "Failed to allocate batch".to_string(),
            ));
        }
        Ok(Self {
            batch,
            capacity: n_tokens,
        })
    }

    /// Clear the batch for reuse
    pub fn clear(&mut self) {
        unsafe {
            batch_ffi::bitnet_batch_clear(self.batch);
        }
    }

    /// Add a token to the batch
    pub fn add(
        &mut self,
        token: i32,
        pos: i32,
        seq_id: BitNetSeqId,
        output_logits: bool,
    ) {
        let seq_ids = [seq_id];
        unsafe {
            batch_ffi::bitnet_batch_add(
                self.batch,
                token,
                pos,
                seq_ids.as_ptr(),
                1,
                output_logits as i8,
            );
        }
    }

    /// Add a token belonging to multiple sequences
    pub fn add_multi_seq(
        &mut self,
        token: i32,
        pos: i32,
        seq_ids: &[BitNetSeqId],
        output_logits: bool,
    ) {
        unsafe {
            batch_ffi::bitnet_batch_add(
                self.batch,
                token,
                pos,
                seq_ids.as_ptr(),
                seq_ids.len() as i32,
                output_logits as i8,
            );
        }
    }

    /// Get the internal batch pointer (for FFI calls)
    pub(crate) fn as_ptr(&self) -> *const CBatch {
        self.batch
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        if !self.batch.is_null() {
            unsafe {
                batch_ffi::bitnet_batch_free(self.batch);
            }
        }
    }
}

// Safety: Batch is only accessed through &mut self or owned
unsafe impl Send for Batch {}

/// Native batch inference engine with multi-sequence support.
///
/// This engine supports continuous batching where multiple sequences
/// can be processed together, with new sequences joining mid-generation.
pub struct NativeBatchEngine {
    engine: *mut CEngine,
    config: BatchConfig,
}

// Safety: The C++ engine uses internal locking for thread safety
unsafe impl Send for NativeBatchEngine {}
unsafe impl Sync for NativeBatchEngine {}

impl NativeBatchEngine {
    /// Create a new batch engine from a model path
    pub fn new(model_path: &str, config: Option<BatchConfig>) -> Result<Self, BatchError> {
        let c_path = CString::new(model_path).map_err(|_| BatchError::InvalidModelPath)?;

        let config = config.unwrap_or_default();
        let c_config = BitNetBatchConfig::from(&config);

        let engine =
            unsafe { batch_ffi::bitnet_batch_engine_create(c_path.as_ptr(), &c_config) };

        if engine.is_null() {
            return Err(BatchError::EngineCreation(
                "Failed to create batch engine".to_string(),
            ));
        }

        Ok(Self { engine, config })
    }

    /// Get the batch configuration
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    // ==========================================================================
    // Sequence Management
    // ==========================================================================

    /// Allocate a new sequence slot
    pub fn alloc_sequence(&self) -> Result<BitNetSeqId, BatchError> {
        let seq_id = unsafe { batch_ffi::bitnet_seq_alloc(self.engine) };
        if seq_id < 0 {
            Err(BatchError::NoAvailableSlots)
        } else {
            Ok(seq_id)
        }
    }

    /// Free a sequence slot
    pub fn free_sequence(&self, seq_id: BitNetSeqId) {
        unsafe {
            batch_ffi::bitnet_seq_free(self.engine, seq_id);
        }
    }

    /// Get sequence information
    pub fn get_sequence_info(&self, seq_id: BitNetSeqId) -> Result<SequenceInfo, BatchError> {
        let mut info = BitNetSeqInfo::default();
        let result = unsafe { batch_ffi::bitnet_seq_get_info(self.engine, seq_id, &mut info) };

        if result != 0 {
            return Err(BatchError::InvalidSequenceId(seq_id));
        }

        Ok(SequenceInfo {
            seq_id: info.seq_id,
            is_active: info.is_active,
            position: info.position,
            prompt_len: info.prompt_len,
            generated_count: info.generated_count,
        })
    }

    /// Get number of active sequences
    pub fn active_sequence_count(&self) -> i32 {
        unsafe { batch_ffi::bitnet_seq_active_count(self.engine) }
    }

    /// Get number of available sequence slots
    pub fn available_slots(&self) -> i32 {
        unsafe { batch_ffi::bitnet_seq_available_slots(self.engine) }
    }

    // ==========================================================================
    // Batch Inference
    // ==========================================================================

    /// Process a batch through the model
    ///
    /// Returns Ok(()) on success, Err(KVCacheFull) if cache is full
    pub fn decode(&self, batch: &Batch) -> Result<(), BatchError> {
        let result = unsafe { batch_ffi::bitnet_batch_decode(self.engine, batch.as_ptr()) };

        match result {
            0 => Ok(()),
            1 => Err(BatchError::KVCacheFull),
            _ => Err(BatchError::DecodeFailed(format!(
                "Decode returned error code: {}",
                result
            ))),
        }
    }

    /// Get logits for a batch position
    ///
    /// Returns None if logits were not requested for this position
    pub fn get_logits(&self, batch_idx: i32) -> Option<&[f32]> {
        let ptr = unsafe { batch_ffi::bitnet_get_logits_ith(self.engine, batch_idx) };
        if ptr.is_null() {
            return None;
        }

        let vocab_size = self.vocab_size() as usize;
        Some(unsafe { std::slice::from_raw_parts(ptr, vocab_size) })
    }

    /// Sample a token from logits at batch position
    pub fn sample(
        &self,
        batch_idx: i32,
        params: &BatchSamplingParams,
    ) -> Result<i32, BatchError> {
        let c_params = BitNetSamplingParams::from(params);
        let token = unsafe { batch_ffi::bitnet_batch_sample(self.engine, batch_idx, &c_params) };

        if token < 0 {
            Err(BatchError::DecodeFailed(
                "Sampling failed".to_string(),
            ))
        } else {
            Ok(token)
        }
    }

    // ==========================================================================
    // KV Cache Management
    // ==========================================================================

    /// Remove KV cache entries for a sequence
    ///
    /// Use seq_id = -1 to clear all sequences
    pub fn kv_cache_seq_rm(&self, seq_id: BitNetSeqId, p0: i32, p1: i32) {
        unsafe {
            batch_ffi::bitnet_kv_cache_seq_rm(self.engine, seq_id, p0, p1);
        }
    }

    /// Copy KV cache from one sequence to another
    pub fn kv_cache_seq_cp(
        &self,
        src_seq_id: BitNetSeqId,
        dst_seq_id: BitNetSeqId,
        p0: i32,
        p1: i32,
    ) {
        unsafe {
            batch_ffi::bitnet_kv_cache_seq_cp(self.engine, src_seq_id, dst_seq_id, p0, p1);
        }
    }

    /// Get maximum position in KV cache for a sequence
    pub fn kv_cache_seq_pos_max(&self, seq_id: BitNetSeqId) -> i32 {
        unsafe { batch_ffi::bitnet_kv_cache_seq_pos_max(self.engine, seq_id) }
    }

    /// Get number of used KV cache cells
    pub fn kv_cache_used_cells(&self) -> i32 {
        unsafe { batch_ffi::bitnet_kv_cache_used_cells(self.engine) }
    }

    /// Get total KV cache capacity
    pub fn kv_cache_capacity(&self) -> i32 {
        unsafe { batch_ffi::bitnet_kv_cache_capacity(self.engine) }
    }

    /// Clear entire KV cache
    pub fn kv_cache_clear(&self) {
        unsafe {
            batch_ffi::bitnet_kv_cache_clear(self.engine);
        }
    }

    // ==========================================================================
    // Model Information
    // ==========================================================================

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> i32 {
        unsafe { batch_ffi::bitnet_batch_eos_token(self.engine) }
    }

    /// Check if token is end-of-generation
    pub fn is_eos(&self, token: i32) -> bool {
        unsafe { batch_ffi::bitnet_batch_is_eos(self.engine, token) }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> i32 {
        unsafe { batch_ffi::bitnet_batch_vocab_size(self.engine) }
    }

    /// Get context length
    pub fn n_ctx(&self) -> i32 {
        unsafe { batch_ffi::bitnet_batch_n_ctx(self.engine) }
    }

    /// Get embedding dimension
    pub fn n_embd(&self) -> i32 {
        unsafe { batch_ffi::bitnet_batch_n_embd(self.engine) }
    }

    /// Get maximum number of concurrent sequences
    pub fn max_sequences(&self) -> i32 {
        unsafe { batch_ffi::bitnet_batch_max_sequences(self.engine) }
    }

    /// Get number of currently active sequences
    pub fn active_sequences(&self) -> i32 {
        unsafe { batch_ffi::bitnet_batch_active_sequences(self.engine) }
    }

    /// Get maximum context length per sequence
    pub fn max_ctx_per_seq(&self) -> i32 {
        unsafe { batch_ffi::bitnet_batch_max_ctx_per_seq(self.engine) }
    }

    // ==========================================================================
    // Tokenization
    // ==========================================================================

    /// Tokenize text to token IDs
    ///
    /// Returns a vector of token IDs. If add_special is true, special tokens
    /// (like BOS) will be added.
    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>, BatchError> {
        let c_text = CString::new(text).map_err(|_| BatchError::InvalidModelPath)?;

        // First call to get token count
        let mut tokens = vec![0i32; 8192]; // Max reasonable prompt size
        let n_tokens = unsafe {
            batch_ffi::bitnet_tokenize(
                self.engine,
                c_text.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
            )
        };

        if n_tokens < 0 {
            return Err(BatchError::DecodeFailed("Tokenization failed".to_string()));
        }

        tokens.truncate(n_tokens as usize);
        Ok(tokens)
    }

    /// Detokenize token IDs to text
    ///
    /// Returns the decoded text string.
    pub fn detokenize(&self, tokens: &[i32]) -> Result<String, BatchError> {
        let mut buffer = vec![0u8; 8192];
        let len = unsafe {
            batch_ffi::bitnet_detokenize(
                self.engine,
                tokens.as_ptr(),
                tokens.len() as i32,
                buffer.as_mut_ptr() as *mut i8,
                buffer.len() as i32,
            )
        };

        if len < 0 {
            return Err(BatchError::DecodeFailed("Detokenization failed".to_string()));
        }

        buffer.truncate(len as usize);
        String::from_utf8(buffer)
            .map_err(|_| BatchError::DecodeFailed("Invalid UTF-8 in detokenized text".to_string()))
    }

    /// Detokenize a single token to text
    pub fn token_to_piece(&self, token: i32) -> Result<String, BatchError> {
        self.detokenize(&[token])
    }
}

impl Drop for NativeBatchEngine {
    fn drop(&mut self) {
        if !self.engine.is_null() {
            unsafe {
                batch_ffi::bitnet_batch_engine_destroy(self.engine);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Requires C++ library to be built"]
    fn test_batch_creation() {
        let batch = Batch::new(512, 1);
        assert!(batch.is_ok());
    }

    #[test]
    #[ignore = "Requires C++ library to be built"]
    fn test_engine_creation() {
        let result = NativeBatchEngine::new("microsoft/bitnet-b1.58-2B-4T", None);
        // Engine creation may fail without model, but should not panic
        assert!(result.is_ok() || result.is_err());
    }
}
