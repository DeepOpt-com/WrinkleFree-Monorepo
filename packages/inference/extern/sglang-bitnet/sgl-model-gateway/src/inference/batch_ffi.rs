//! FFI bindings to the C++ BitNet batch inference engine.
//!
//! These bindings match the C API defined in bitnet_batch.h

use libc::{c_char, c_int, c_void};
use std::ffi::CStr;

/// Sequence ID type
pub type BitNetSeqId = i32;

/// Opaque batch engine handle
#[repr(C)]
pub struct BitNetBatchEngine {
    _private: [u8; 0],
}

/// Opaque batch handle
#[repr(C)]
pub struct BitNetBatch {
    _private: [u8; 0],
}

/// Batch engine configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BitNetBatchConfig {
    pub max_batch_size: i32,
    pub max_sequences: i32,
    pub n_ctx: i32,
    pub n_ctx_per_seq: i32,
    pub num_threads: i32,
}

impl Default for BitNetBatchConfig {
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

/// Sequence state enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitNetSeqState {
    Idle = 0,
    Prefilling = 1,
    Decoding = 2,
    Finished = 3,
}

/// Sequence info structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BitNetSeqInfo {
    pub seq_id: BitNetSeqId,
    pub state: BitNetSeqState,
    pub position: i32,
    pub prompt_len: i32,
    pub generated_count: i32,
}

impl Default for BitNetSeqInfo {
    fn default() -> Self {
        Self {
            seq_id: -1,
            state: BitNetSeqState::Idle,
            position: 0,
            prompt_len: 0,
            generated_count: 0,
        }
    }
}

/// Sampling parameters (reused from single-sequence API)
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

// Link to the C++ library
#[link(name = "sgl_kernel_inference")]
extern "C" {
    // ==========================================================================
    // Batch Engine Lifecycle
    // ==========================================================================

    /// Get default batch configuration
    pub fn bitnet_batch_config_default() -> BitNetBatchConfig;

    /// Create batch engine with multi-sequence support
    pub fn bitnet_batch_engine_create(
        model_path: *const c_char,
        config: *const BitNetBatchConfig,
    ) -> *mut BitNetBatchEngine;

    /// Destroy batch engine
    pub fn bitnet_batch_engine_destroy(engine: *mut BitNetBatchEngine);

    // ==========================================================================
    // Batch Management
    // ==========================================================================

    /// Initialize a batch structure
    pub fn bitnet_batch_init(n_tokens: i32, n_seq_max: i32) -> *mut BitNetBatch;

    /// Free batch structure
    pub fn bitnet_batch_free(batch: *mut BitNetBatch);

    /// Clear batch for reuse
    pub fn bitnet_batch_clear(batch: *mut BitNetBatch);

    /// Add a token to the batch
    pub fn bitnet_batch_add(
        batch: *mut BitNetBatch,
        token: i32,
        pos: i32,
        seq_ids: *const BitNetSeqId,
        n_seq_ids: i32,
        output_logits: i8,
    );

    // ==========================================================================
    // Sequence Management
    // ==========================================================================

    /// Allocate a sequence slot
    pub fn bitnet_seq_alloc(engine: *mut BitNetBatchEngine) -> BitNetSeqId;

    /// Free a sequence slot
    pub fn bitnet_seq_free(engine: *mut BitNetBatchEngine, seq_id: BitNetSeqId);

    /// Get sequence information
    pub fn bitnet_seq_get_info(
        engine: *mut BitNetBatchEngine,
        seq_id: BitNetSeqId,
        info: *mut BitNetSeqInfo,
    ) -> c_int;

    /// Get number of active sequences
    pub fn bitnet_seq_active_count(engine: *mut BitNetBatchEngine) -> i32;

    /// Get number of available slots
    pub fn bitnet_seq_available_slots(engine: *mut BitNetBatchEngine) -> i32;

    // ==========================================================================
    // Batch Inference
    // ==========================================================================

    /// Process batch through model
    pub fn bitnet_batch_decode(
        engine: *mut BitNetBatchEngine,
        batch: *const BitNetBatch,
    ) -> c_int;

    /// Get logits for a batch position
    pub fn bitnet_get_logits_ith(
        engine: *mut BitNetBatchEngine,
        batch_idx: i32,
    ) -> *const f32;

    /// Sample token from logits at batch position
    pub fn bitnet_batch_sample(
        engine: *mut BitNetBatchEngine,
        batch_idx: i32,
        params: *const CSamplingParams,
    ) -> i32;

    // ==========================================================================
    // KV Cache Management
    // ==========================================================================

    /// Remove KV cache entries for a sequence
    pub fn bitnet_kv_cache_seq_rm(
        engine: *mut BitNetBatchEngine,
        seq_id: BitNetSeqId,
        p0: i32,
        p1: i32,
    ) -> c_int;

    /// Copy KV cache from one sequence to another
    pub fn bitnet_kv_cache_seq_cp(
        engine: *mut BitNetBatchEngine,
        seq_id_src: BitNetSeqId,
        seq_id_dst: BitNetSeqId,
        p0: i32,
        p1: i32,
    );

    /// Get maximum position in KV cache for a sequence
    pub fn bitnet_kv_cache_seq_pos_max(
        engine: *mut BitNetBatchEngine,
        seq_id: BitNetSeqId,
    ) -> i32;

    /// Get number of used KV cache cells
    pub fn bitnet_kv_cache_used_cells(engine: *mut BitNetBatchEngine) -> i32;

    /// Get total KV cache capacity
    pub fn bitnet_kv_cache_capacity(engine: *mut BitNetBatchEngine) -> i32;

    /// Clear entire KV cache
    pub fn bitnet_kv_cache_clear(engine: *mut BitNetBatchEngine);

    // ==========================================================================
    // Model Information
    // ==========================================================================

    /// Get EOS token ID
    pub fn bitnet_batch_eos_token(engine: *mut BitNetBatchEngine) -> i32;

    /// Get vocabulary size
    pub fn bitnet_batch_vocab_size(engine: *mut BitNetBatchEngine) -> i32;

    /// Get maximum context length per sequence
    pub fn bitnet_batch_max_ctx_per_seq(engine: *mut BitNetBatchEngine) -> i32;
}
