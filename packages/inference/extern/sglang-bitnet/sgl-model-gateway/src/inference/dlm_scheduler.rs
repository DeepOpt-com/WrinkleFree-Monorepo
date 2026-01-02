//! Batch scheduler for Fast-dLLM v2 block diffusion inference.
//!
//! This module implements the Fast-dLLM v2 algorithm (arXiv:2509.26328) for
//! parallel token generation using block diffusion.
//!
//! ## Key Features
//!
//! - **Block Diffusion**: Generate `block_size` (default 32) tokens in parallel
//! - **Confidence Thresholding**: Unmask tokens above confidence threshold
//! - **DualCache**: Hierarchical KV cache for sub-block reuse
//!
//! ## Algorithm Overview
//!
//! 1. Initialize block with mask tokens
//! 2. Forward pass to get logits for masked positions
//! 3. Sample tokens and compute confidence (softmax probability)
//! 4. Unmask tokens where confidence > threshold
//! 5. Always unmask at least one token (highest confidence)
//! 6. Repeat until all tokens unmasked
//! 7. Move to next block

use super::batch_engine::{Batch, BatchConfig, BatchError, NativeBatchEngine};
use super::batch_ffi::BitNetSeqId;
use super::dlm_config::DlmConfig;
use super::radix_cache::{RadixCache, RadixCacheConfig};
use super::sequence::{
    FinishReason, InferenceRequest, InferenceResponse, SequencePhase, SequenceState, StreamToken,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// DualCache for Fast-dLLM v2 sub-block KV reuse.
///
/// Maintains both prefix and block-level caches to enable efficient
/// recomputation as additional tokens are revealed within a block.
#[derive(Debug)]
struct DualCache {
    /// KV cache position for completed blocks (prefix)
    prefix_kv_pos: i32,
    /// KV cache position for current block (block-level)
    block_kv_pos: Option<i32>,
    /// Which small blocks have been computed
    computed_small_blocks: Vec<bool>,
}

impl DualCache {
    fn new(prefix_kv_pos: i32, num_small_blocks: usize) -> Self {
        Self {
            prefix_kv_pos,
            block_kv_pos: None,
            computed_small_blocks: vec![false; num_small_blocks],
        }
    }

    /// Check if we can reuse cache for a small block.
    ///
    /// Cache reuse is possible when:
    /// 1. Block KV cache exists
    /// 2. First token of small block is already unmasked
    /// 3. Small block has been computed before
    fn can_reuse(&self, small_block_idx: usize, first_token_unmasked: bool) -> bool {
        self.block_kv_pos.is_some()
            && first_token_unmasked
            && small_block_idx < self.computed_small_blocks.len()
            && self.computed_small_blocks[small_block_idx]
    }

    /// Mark a small block as computed.
    fn mark_computed(&mut self, small_block_idx: usize, kv_end_pos: i32) {
        if small_block_idx < self.computed_small_blocks.len() {
            self.computed_small_blocks[small_block_idx] = true;
        }
        self.block_kv_pos = Some(kv_end_pos);
    }

    /// Reset block cache (called when moving to next block).
    fn reset_block(&mut self, new_prefix_pos: i32, num_small_blocks: usize) {
        self.prefix_kv_pos = new_prefix_pos;
        self.block_kv_pos = None;
        self.computed_small_blocks = vec![false; num_small_blocks];
    }
}

/// Scheduler configuration for DLM.
#[derive(Debug, Clone)]
pub struct DlmSchedulerConfig {
    /// Maximum concurrent sequences
    pub max_sequences: usize,
    /// DLM-specific configuration
    pub dlm: DlmConfig,
    /// Enable RadixCache for prefix caching across requests
    pub enable_radix_cache: bool,
    /// Maximum tokens in RadixCache
    pub radix_cache_max_tokens: usize,
}

impl Default for DlmSchedulerConfig {
    fn default() -> Self {
        Self {
            max_sequences: 16,
            dlm: DlmConfig::default(),
            enable_radix_cache: true,
            radix_cache_max_tokens: 100_000,
        }
    }
}

/// Handle for submitting requests to the DLM scheduler.
#[derive(Clone)]
pub struct DlmSchedulerHandle {
    request_tx: mpsc::Sender<InferenceRequest>,
}

impl DlmSchedulerHandle {
    /// Submit a request to the scheduler.
    pub async fn submit(&self, request: InferenceRequest) -> Result<(), BatchError> {
        self.request_tx
            .send(request)
            .await
            .map_err(|_| BatchError::NotInitialized)
    }
}

/// DLM block state tracking.
#[derive(Debug)]
struct DlmBlockState {
    /// Current block tokens (includes masks and unmasked tokens)
    tokens: Vec<i32>,
    /// Position in sequence where block starts
    start_pos: i32,
    /// Number of iterations performed
    iterations: usize,
}

/// Extended sequence state for DLM.
struct DlmSequenceState {
    /// Base sequence state
    base: SequenceState,
    /// Current block being decoded
    current_block: Option<DlmBlockState>,
    /// DualCache for sub-block reuse
    dual_cache: Option<DualCache>,
}

impl DlmSequenceState {
    fn new(seq_id: BitNetSeqId, request: InferenceRequest) -> Self {
        Self {
            base: SequenceState::new(seq_id, request),
            current_block: None,
            dual_cache: None,
        }
    }
}

/// DLM Scheduler implementing Fast-dLLM v2 block diffusion.
pub struct DlmScheduler {
    config: DlmSchedulerConfig,
    engine: Arc<NativeBatchEngine>,
    radix_cache: RadixCache,

    request_rx: mpsc::Receiver<InferenceRequest>,
    sequences: HashMap<BitNetSeqId, DlmSequenceState>,
    pending_queue: VecDeque<InferenceRequest>,

    batch: Batch,
    eos_token_id: i32,
}

impl DlmScheduler {
    /// Create a new DLM scheduler.
    pub fn new(
        config: DlmSchedulerConfig,
        engine: Arc<NativeBatchEngine>,
    ) -> (Self, DlmSchedulerHandle) {
        let (request_tx, request_rx) = mpsc::channel(1024);

        // Create batch with capacity for prefill + block decode
        // Use the engine's max context per sequence to ensure we can prefill long prompts
        let batch_size = 2048; // Matches BatchConfig::default().n_ctx
        let batch = Batch::new(batch_size, 1).expect("Failed to create batch");

        let eos_token_id = engine.eos_token_id();

        // Create RadixCache
        let radix_cache = RadixCache::new(RadixCacheConfig {
            max_cached_tokens: config.radix_cache_max_tokens,
            enabled: config.enable_radix_cache,
            ..Default::default()
        });

        if config.enable_radix_cache {
            info!(
                "DLM RadixCache enabled with {} max tokens",
                config.radix_cache_max_tokens
            );
        }

        let scheduler = Self {
            config,
            engine,
            radix_cache,
            request_rx,
            sequences: HashMap::new(),
            pending_queue: VecDeque::new(),
            batch,
            eos_token_id,
        };

        let handle = DlmSchedulerHandle { request_tx };

        (scheduler, handle)
    }

    /// Run the scheduler loop.
    pub async fn run(&mut self) {
        info!("DLM scheduler starting (block_size={})", self.config.dlm.block_size);

        loop {
            // Phase 1: Receive new requests
            self.receive_requests().await;

            let pending = self.pending_queue.len();
            let active = self.sequences.len();
            if pending > 0 || active > 0 {
                debug!("Loop: pending={}, active={}", pending, active);
            }

            // Phase 2: Assign slots to pending requests
            self.assign_slots();

            // Phase 3: Run prefill for new sequences
            self.run_prefill_phase();

            // Phase 4: Run block decode for decoding sequences
            self.run_block_decode_phase().await;

            // Phase 5: Cleanup finished sequences
            self.cleanup_finished();

            // Yield if no work
            if self.sequences.is_empty() && self.pending_queue.is_empty() {
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            }
        }
    }

    /// Receive requests from the channel.
    async fn receive_requests(&mut self) {
        use tokio::time::{timeout, Duration};

        // Non-blocking drain
        loop {
            match self.request_rx.try_recv() {
                Ok(req) => {
                    debug!("Received request {}", req.request_id);
                    self.pending_queue.push_back(req);
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        // If idle, wait for requests
        if self.pending_queue.is_empty() && self.sequences.is_empty() {
            let deadline = Duration::from_millis(10);
            if let Ok(Some(req)) = timeout(deadline, self.request_rx.recv()).await {
                debug!("Received request {} (blocking)", req.request_id);
                self.pending_queue.push_back(req);
            }
        }
    }

    /// Assign sequence slots to pending requests.
    fn assign_slots(&mut self) {
        while !self.pending_queue.is_empty() {
            let seq_id = match self.engine.alloc_sequence() {
                Ok(id) => id,
                Err(BatchError::NoAvailableSlots) => {
                    warn!("No available slots for pending requests");
                    break;
                }
                Err(e) => {
                    error!("Failed to allocate slot: {}", e);
                    break;
                }
            };

            let request = self.pending_queue.pop_front().unwrap();
            let request_id = request.request_id.clone();

            info!("Assigned slot {} to request {}", seq_id, request_id);

            let state = DlmSequenceState::new(seq_id, request);
            self.sequences.insert(seq_id, state);
        }
    }

    /// Run prefill phase for new sequences.
    fn run_prefill_phase(&mut self) {
        let seq_ids: Vec<BitNetSeqId> = self.sequences.keys().copied().collect();

        for seq_id in seq_ids {
            let state = self.sequences.get_mut(&seq_id).unwrap();
            if !state.base.is_prefilling() {
                continue;
            }

            info!("Prefilling seq {} with {} prompt tokens", seq_id, state.base.prompt_tokens.len());

            // Clear any stale KV cache for this sequence before prefill
            self.engine.kv_cache_seq_rm(seq_id, -1, -1);

            // Prefill all prompt tokens
            self.batch.clear();
            let prompt_len = state.base.prompt_tokens.len();

            for (i, &token) in state.base.prompt_tokens.iter().enumerate() {
                let pos = i as i32;
                let is_last = i == prompt_len - 1;
                self.batch.add(token, pos, seq_id, is_last);
            }

            if let Err(e) = self.engine.decode(&self.batch) {
                error!("Prefill failed for seq {}: {}", seq_id, e);
                // Mark as failed so cleanup can handle it
                state.base.phase = SequencePhase::Finished(FinishReason::Error);
                continue;
            }

            info!("Prefill decode completed for seq {}", seq_id);

            // Sample first token
            let batch_idx = (prompt_len - 1) as i32;
            info!("Sampling first token for seq {} at batch_idx {}", seq_id, batch_idx);
            match self.engine.sample(batch_idx, &state.base.params) {
                Ok(first_token) => {
                    info!("Sampled first token {} for seq {}", first_token, seq_id);
                    let is_eos = self.engine.is_eos(first_token);
                    state.base.add_generated_token(first_token, is_eos);
                    state.base.position = prompt_len as i32;

                    // Transition to decoding
                    if !is_eos {
                        state.base.phase = SequencePhase::Decoding;
                        info!("Seq {} transitioned to Decoding phase", seq_id);

                        // Initialize DualCache for block decoding
                        let num_small_blocks = self.config.dlm.num_small_blocks();
                        state.dual_cache = Some(DualCache::new(
                            state.base.position,
                            num_small_blocks,
                        ));
                    } else {
                        info!("First token is EOS for seq {}", seq_id);
                    }

                    // Stream the first token
                    let text = self.engine.token_to_piece(first_token).unwrap_or_default();
                    state.base.send_token(first_token, text, is_eos);
                }
                Err(e) => {
                    error!("Failed to sample first token for seq {}: {}", seq_id, e);
                    state.base.phase = SequencePhase::Finished(FinishReason::Error);
                }
            }
        }
    }

    /// Run block decode phase for decoding sequences.
    async fn run_block_decode_phase(&mut self) {
        let seq_ids: Vec<BitNetSeqId> = self
            .sequences
            .keys()
            .copied()
            .collect();

        for seq_id in seq_ids {
            let state = self.sequences.get(&seq_id).unwrap();
            if !state.base.is_decoding() {
                continue;
            }

            // Check if we should generate more
            let max_tokens = state.base.max_tokens as usize;
            let generated = state.base.generated_tokens.len();
            debug!("Block decode for seq {}: {}/{} tokens", seq_id, generated, max_tokens);

            if generated >= max_tokens {
                info!("Seq {} reached max_tokens, finishing", seq_id);
                let state = self.sequences.get_mut(&seq_id).unwrap();
                state.base.phase = SequencePhase::Finished(FinishReason::Length);
                continue;
            }

            // Decode one block
            info!("Starting block decode for seq {}", seq_id);
            self.decode_block(seq_id).await;
        }
    }

    /// Decode a single block using Fast-dLLM v2 algorithm.
    async fn decode_block(&mut self, seq_id: BitNetSeqId) {
        let block_size = self.config.dlm.block_size;
        let small_block_size = self.config.dlm.small_block_size;
        let num_small_blocks = self.config.dlm.num_small_blocks();
        let threshold = self.config.dlm.threshold;
        let mask_id = self.config.dlm.mask_token_id;

        debug!("decode_block: seq={}, block_size={}, small_block_size={}, num_small_blocks={}",
               seq_id, block_size, small_block_size, num_small_blocks);

        let state = self.sequences.get_mut(&seq_id).unwrap();
        let block_start_pos = state.base.position;

        // Initialize block with mask tokens
        let mut block_tokens: Vec<i32> = vec![mask_id; block_size];

        // Get or create DualCache
        if state.dual_cache.is_none() {
            state.dual_cache = Some(DualCache::new(block_start_pos, num_small_blocks));
        }

        // Process each small block
        for small_block_idx in 0..num_small_blocks {
            let start_idx = small_block_idx * small_block_size;
            let end_idx = start_idx + small_block_size;
            debug!("Processing small block {} (indices {}..{})", small_block_idx, start_idx, end_idx);

            // Iterate until all masks in this small block are resolved
            let mut iteration = 0;
            loop {
                iteration += 1;
                let small_block_masks: usize = block_tokens[start_idx..end_idx]
                    .iter()
                    .filter(|&&t| t == mask_id)
                    .count();

                debug!("  Iteration {}: {} masks remaining", iteration, small_block_masks);

                if small_block_masks == 0 {
                    break;
                }

                // Check DualCache for reuse opportunity
                let first_token_unmasked = block_tokens[start_idx] != mask_id;
                let state = self.sequences.get(&seq_id).unwrap();
                let use_cached = state
                    .dual_cache
                    .as_ref()
                    .map(|c| c.can_reuse(small_block_idx, first_token_unmasked))
                    .unwrap_or(false);

                self.batch.clear();
                let logit_positions: Vec<usize>;
                // Track batch index mapping:
                // - masked_only=true: batch_idx = enumerate index
                // - masked_only=false: batch_idx = block_idx - batch_offset
                let masked_only: bool;
                let batch_offset: usize;

                if use_cached && first_token_unmasked {
                    // DualCache hit: Only compute for remaining masked positions
                    // Clear KV cache for just the masked positions we're recomputing
                    for i in start_idx..end_idx {
                        if block_tokens[i] == mask_id {
                            let pos = block_start_pos + i as i32;
                            self.engine.kv_cache_seq_rm(seq_id, pos, pos + 1);
                        }
                    }

                    logit_positions = (start_idx..end_idx)
                        .filter(|&i| block_tokens[i] == mask_id)
                        .collect();

                    // Compact batch: only add masked tokens
                    for &i in &logit_positions {
                        let pos = block_start_pos + i as i32;
                        self.batch.add(block_tokens[i], pos, seq_id, true);
                    }
                    masked_only = true;
                    batch_offset = 0; // Not used when masked_only=true
                } else {
                    // Full block forward - clear KV cache from current small block onwards
                    // Only clear from current position to preserve valid cache from previous small blocks
                    let current_pos = block_start_pos + start_idx as i32;
                    let block_end_pos = block_start_pos + block_size as i32;
                    self.engine.kv_cache_seq_rm(seq_id, current_pos, block_end_pos);

                    // Add tokens from current small block onwards only
                    // Previous small blocks already have valid KV cache
                    for i in start_idx..block_size {
                        let token = block_tokens[i];
                        let pos = block_start_pos + i as i32;
                        let need_logits = i < end_idx && token == mask_id;
                        self.batch.add(token, pos, seq_id, need_logits);
                    }

                    logit_positions = (start_idx..end_idx)
                        .filter(|&i| block_tokens[i] == mask_id)
                        .collect();
                    masked_only = false;
                    batch_offset = start_idx; // Batch indices are relative to start_idx

                    // Update DualCache
                    let kv_end_pos = block_start_pos + block_size as i32;
                    let state = self.sequences.get_mut(&seq_id).unwrap();
                    if let Some(ref mut cache) = state.dual_cache {
                        cache.mark_computed(small_block_idx, kv_end_pos);
                    }
                }

                // Forward pass
                debug!("Running forward pass");
                if let Err(e) = self.engine.decode(&self.batch) {
                    error!("Block decode failed for seq {}: {}", seq_id, e);
                    // Mark as failed so cleanup can handle it
                    let state = self.sequences.get_mut(&seq_id).unwrap();
                    state.base.phase = SequencePhase::Finished(FinishReason::Error);
                    return;
                }
                debug!("Forward pass completed");

                // Process masked positions with confidence thresholding
                let mut candidates: Vec<(usize, i32, f32)> = Vec::new();

                for (enum_idx, &block_idx) in logit_positions.iter().enumerate() {
                    // Determine batch index based on batch type:
                    // - masked_only: sequential indices (0, 1, 2...)
                    // - partial block: block_idx - batch_offset
                    let batch_idx = if masked_only {
                        enum_idx as i32
                    } else {
                        (block_idx - batch_offset) as i32
                    };

                    if let Some(logits) = self.engine.get_logits(batch_idx) {
                        let probs = softmax(logits);
                        let token_id = argmax(logits);
                        let confidence = probs[token_id as usize];
                        candidates.push((block_idx, token_id, confidence));
                    }
                }

                // Unmask high-confidence tokens
                debug!("Candidates: {} (threshold={})", candidates.len(), threshold);
                if !candidates.is_empty() {
                    let max_conf = candidates.iter().map(|c| c.2).fold(0.0f32, f32::max);
                    debug!("  Max confidence: {}", max_conf);
                }

                let mut unmasked_any = false;
                for &(idx, token, conf) in &candidates {
                    if conf > threshold {
                        debug!("  Unmasking idx {} with token {} (conf={})", idx, token, conf);
                        block_tokens[idx] = token;
                        unmasked_any = true;

                        // Stream this token
                        self.emit_token(seq_id, token);
                    }
                }

                // Always unmask at least one (highest confidence)
                if !unmasked_any && !candidates.is_empty() {
                    let (idx, token, conf) = candidates
                        .iter()
                        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Less))
                        .copied()
                        .unwrap();

                    debug!("  Force unmasking idx {} with token {} (conf={})", idx, token, conf);
                    block_tokens[idx] = token;
                    self.emit_token(seq_id, token);
                }
            }
        }

        // Finalize block
        let state = self.sequences.get_mut(&seq_id).unwrap();
        let new_pos = block_start_pos + block_size as i32;
        state.base.position = new_pos;

        // Add generated tokens
        for &token in &block_tokens {
            let is_eos = self.engine.is_eos(token);
            state.base.generated_tokens.push(token);

            if is_eos {
                state.base.phase = SequencePhase::Finished(FinishReason::EOS);
                return;
            }
        }

        // Reset DualCache for next block
        if let Some(ref mut cache) = state.dual_cache {
            cache.reset_block(new_pos, num_small_blocks);
        }

        // Check max tokens
        if state.base.generated_tokens.len() >= state.base.max_tokens as usize {
            state.base.phase = SequencePhase::Finished(FinishReason::Length);
        }
    }

    /// Emit a token to the response stream.
    fn emit_token(&self, seq_id: BitNetSeqId, token: i32) {
        if let Some(state) = self.sequences.get(&seq_id) {
            let text = self.engine.token_to_piece(token).unwrap_or_default();
            state.base.send_token(token, text, false);
        }
    }

    /// Cleanup finished sequences.
    fn cleanup_finished(&mut self) {
        let finished_ids: Vec<BitNetSeqId> = self
            .sequences
            .iter()
            .filter(|(_, s)| s.base.is_finished())
            .map(|(&id, _)| id)
            .collect();

        for seq_id in finished_ids {
            if let Some(mut state) = self.sequences.remove(&seq_id) {
                debug!(
                    "DLM sequence {} finished, generated {} tokens",
                    seq_id,
                    state.base.generated_tokens.len()
                );

                // Insert into RadixCache for future reuse
                let all_tokens: Vec<i32> = state
                    .base
                    .prompt_tokens
                    .iter()
                    .chain(state.base.generated_tokens.iter())
                    .copied()
                    .collect();

                let newly_cached = self.radix_cache.insert(&all_tokens, seq_id);
                if newly_cached > 0 {
                    debug!(
                        "Cached {} new tokens from DLM seq {}",
                        newly_cached, seq_id
                    );
                }

                // Free resources
                self.engine.kv_cache_seq_rm(seq_id, -1, -1);
                self.engine.free_sequence(seq_id);

                // Send final response
                let text = self
                    .engine
                    .detokenize(&state.base.generated_tokens)
                    .unwrap_or_default();
                state.base.complete(text);
            }
        }
    }

    /// Get scheduler statistics.
    pub fn stats(&self) -> DlmSchedulerStats {
        DlmSchedulerStats {
            active_sequences: self.sequences.len(),
            pending_requests: self.pending_queue.len(),
            radix_cache_tokens: self.radix_cache.cached_tokens(),
            block_size: self.config.dlm.block_size,
        }
    }
}

/// DLM scheduler statistics.
#[derive(Debug, Clone)]
pub struct DlmSchedulerStats {
    pub active_sequences: usize,
    pub pending_requests: usize,
    pub radix_cache_tokens: usize,
    pub block_size: usize,
}

/// Compute softmax of logits.
///
/// Uses numerically stable implementation with max subtraction.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_vals: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    exp_vals.iter().map(|x| x / sum).collect()
}

/// Find index of maximum value in logits.
/// Handles NaN values by treating them as less than any other value.
pub fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_correctness() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax sum: {}", sum);

        // Monotonic with logits
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values shouldn't overflow
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);

        assert!(!probs.iter().any(|p| p.is_nan() || p.is_infinite()));
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_negative_values() {
        let logits = vec![-100.0, -50.0, 0.0];
        let probs = softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let logits = vec![0.0; 100];
        let probs = softmax(&logits);

        // All should be equal
        let expected = 1.0 / 100.0;
        for p in &probs {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_argmax() {
        let logits = vec![0.1, 0.9, 0.5, 0.3];
        let idx = argmax(&logits);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_argmax_negative() {
        let logits = vec![-5.0, -3.0, -10.0, -1.0];
        let idx = argmax(&logits);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_argmax_first() {
        let logits = vec![10.0, 5.0, 1.0];
        let idx = argmax(&logits);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_always_unmask_at_least_one() {
        // Simulate all confidences below threshold
        let candidates = vec![
            (0usize, 100i32, 0.3f32), // idx, token, confidence
            (1, 200, 0.4),
            (2, 300, 0.2),
        ];
        let threshold = 0.95;

        let above_threshold: Vec<_> = candidates
            .iter()
            .filter(|(_, _, c)| *c > threshold)
            .collect();

        assert!(above_threshold.is_empty());

        // Should pick highest confidence (idx=1, conf=0.4)
        let best = candidates
            .iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap();

        assert_eq!(best.0, 1);
        assert_eq!(best.1, 200);
    }

    #[test]
    fn test_dual_cache_reuse_logic() {
        let num_small_blocks = 4;
        let mut cache = DualCache::new(0, num_small_blocks);

        // Initially, no cache to reuse
        assert!(!cache.can_reuse(0, true));

        // After computing first block
        cache.mark_computed(0, 32);

        // Can reuse if first token is unmasked
        assert!(cache.can_reuse(0, true));
        assert!(!cache.can_reuse(0, false)); // First token still masked
        assert!(!cache.can_reuse(1, true)); // Small block 1 not computed yet

        // After computing second block
        cache.mark_computed(1, 32);
        assert!(cache.can_reuse(1, true));
    }

    #[test]
    fn test_dual_cache_reset() {
        let mut cache = DualCache::new(0, 4);
        cache.mark_computed(0, 32);
        cache.mark_computed(1, 32);

        assert!(cache.can_reuse(0, true));
        assert!(cache.can_reuse(1, true));

        // Reset for next block
        cache.reset_block(32, 4);

        assert!(!cache.can_reuse(0, true));
        assert!(!cache.can_reuse(1, true));
        assert_eq!(cache.prefix_kv_pos, 32);
    }

    #[test]
    fn test_mask_detection() {
        let mask_id = 128256;
        let block: Vec<i32> = vec![mask_id, 100, mask_id, 200, mask_id];

        let mask_positions: Vec<usize> = block
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == mask_id)
            .map(|(i, _)| i)
            .collect();

        assert_eq!(mask_positions, vec![0, 2, 4]);
    }

    #[test]
    fn test_dlm_scheduler_config_defaults() {
        let config = DlmSchedulerConfig::default();
        assert_eq!(config.max_sequences, 16);
        assert!(config.enable_radix_cache);
    }
}
