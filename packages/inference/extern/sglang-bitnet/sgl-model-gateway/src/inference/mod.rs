//! Native inference module for BitNet models.
//!
//! This module provides Rust bindings to the C++ inference engine,
//! enabling direct model inference without going through Python.
//!
//! ## Single-Sequence API
//! - `NativeEngine` - Simple single-sequence inference
//! - `SamplingParams` - Sampling configuration
//!
//! ## Batch API (Continuous Batching)
//! - `NativeBatchEngine` - Multi-sequence batched inference
//! - `Batch` - Token batch for inference
//! - `BatchConfig` - Batch engine configuration
//! - `BatchSamplingParams` - Sampling configuration for batched inference
//!
//! ## Scheduler
//! - `BatchScheduler` - Request queuing and batch formation
//! - `SchedulerHandle` - Handle for submitting requests
//! - `SchedulerConfig` - Scheduler configuration

mod bitnet_ffi;
mod engine;
mod batch_ffi;
mod batch_engine;
mod sequence;
mod scheduler;
pub mod radix_tree;
pub mod radix_cache;

// Single-sequence API
pub use engine::NativeEngine;
pub use engine::SamplingParams;

// Batch API
pub use batch_engine::{
    Batch,
    BatchConfig,
    BatchError,
    BatchSamplingParams,
    NativeBatchEngine,
    SequenceInfo,
};
pub use batch_ffi::BitNetSeqId;

// Re-export SequenceState from sequence module for backward compatibility
pub use sequence::SequenceState;

// Scheduler API
pub use scheduler::{BatchScheduler, SchedulerConfig, SchedulerHandle, SchedulerStats};
pub use sequence::{
    FinishReason, InferenceRequest, InferenceResponse, StreamToken,
};

// RadixCache API (prefix caching for KV cache reuse)
pub use radix_cache::{RadixCache, RadixCacheConfig, EvictionPolicy, MatchResult, RadixCacheStats};
pub use radix_tree::{RadixTreeNode, TokenVec};
