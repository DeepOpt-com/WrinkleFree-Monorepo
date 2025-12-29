//! Batch scheduler for continuous batching.
//!
//! The scheduler manages request queuing, batch formation, and token generation
//! for multiple concurrent sequences.

use super::batch_engine::{Batch, BatchConfig, BatchError, NativeBatchEngine};
use super::batch_ffi::BitNetSeqId;
use super::radix_cache::{RadixCache, RadixCacheConfig};
use super::sequence::{
    FinishReason, InferenceRequest, InferenceResponse, SequencePhase, SequenceState, StreamToken,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum tokens per batch
    pub max_batch_size: usize,
    /// Maximum concurrent sequences
    pub max_sequences: usize,
    /// Time to wait for request accumulation (ms)
    pub accumulation_window_ms: u64,
    /// Maximum tokens to prefill at once per sequence
    pub prefill_chunk_size: usize,
    /// Minimum tokens reserved for decode operations
    pub min_decode_budget: usize,
    /// Enable RadixCache for prefix caching (improves long sequence performance)
    pub enable_radix_cache: bool,
    /// Maximum tokens to cache in RadixCache
    pub radix_cache_max_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 512,
            max_sequences: 16,
            accumulation_window_ms: 10,
            prefill_chunk_size: 128,
            min_decode_budget: 8,
            enable_radix_cache: true,  // Enabled by default for long sequence efficiency
            radix_cache_max_tokens: 100_000,
        }
    }
}

/// Handle for submitting requests to the scheduler
#[derive(Clone)]
pub struct SchedulerHandle {
    request_tx: mpsc::Sender<InferenceRequest>,
}

impl SchedulerHandle {
    /// Submit a request to the scheduler
    pub async fn submit(&self, request: InferenceRequest) -> Result<(), BatchError> {
        self.request_tx
            .send(request)
            .await
            .map_err(|_| BatchError::NotInitialized)
    }
}

/// The batch scheduler
pub struct BatchScheduler {
    config: SchedulerConfig,
    engine: Arc<NativeBatchEngine>,

    /// Incoming requests
    request_rx: mpsc::Receiver<InferenceRequest>,

    /// Active sequences
    sequences: HashMap<BitNetSeqId, SequenceState>,

    /// Pending requests waiting for a slot
    pending_queue: VecDeque<InferenceRequest>,

    /// Reusable batch structure
    batch: Batch,

    /// Token decoder (placeholder - would use tokenizer in real impl)
    eos_token_id: i32,

    /// RadixCache for prefix sharing (KV cache reuse)
    radix_cache: RadixCache,
}

impl BatchScheduler {
    /// Create a new scheduler
    pub fn new(
        config: SchedulerConfig,
        engine: Arc<NativeBatchEngine>,
    ) -> (Self, SchedulerHandle) {
        let (request_tx, request_rx) = mpsc::channel(1024);

        let batch = Batch::new(config.max_batch_size as i32, 1)
            .expect("Failed to create batch");

        let eos_token_id = engine.eos_token_id();

        // Create RadixCache for prefix sharing
        let radix_cache = RadixCache::new(RadixCacheConfig {
            max_cached_tokens: config.radix_cache_max_tokens,
            enabled: config.enable_radix_cache,
            ..Default::default()
        });

        if config.enable_radix_cache {
            info!("RadixCache enabled with {} max tokens", config.radix_cache_max_tokens);
        }

        let scheduler = Self {
            config,
            engine,
            request_rx,
            sequences: HashMap::new(),
            pending_queue: VecDeque::new(),
            batch,
            eos_token_id,
            radix_cache,
        };

        let handle = SchedulerHandle { request_tx };

        (scheduler, handle)
    }

    /// Run the scheduler loop
    pub async fn run(&mut self) {
        info!("Batch scheduler starting");

        loop {
            // Phase 1: Receive new requests
            self.receive_requests().await;

            // Phase 2: Assign slots to pending requests
            self.assign_slots();

            // Phase 3: Form batch
            self.batch.clear();
            let tokens_added = self.form_batch();

            if tokens_added == 0 {
                // No work to do - wait a bit
                if self.sequences.is_empty() && self.pending_queue.is_empty() {
                    // Idle - wait for requests
                    tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                    continue;
                }
            }

            // Phase 4: Execute batch
            if tokens_added > 0 {
                match self.engine.decode(&self.batch) {
                    Ok(()) => {
                        debug!("Decoded batch of {} tokens", tokens_added);
                    }
                    Err(BatchError::KVCacheFull) => {
                        warn!("KV cache full, need to preempt sequences");
                        // TODO: Implement preemption
                        continue;
                    }
                    Err(e) => {
                        error!("Batch decode failed: {}", e);
                        continue;
                    }
                }

                // Phase 5: Sample and distribute results
                self.sample_and_distribute().await;
            }

            // Phase 6: Cleanup finished sequences
            self.cleanup_finished();
        }
    }

    /// Receive requests from the channel
    async fn receive_requests(&mut self) {
        use tokio::time::{timeout, Duration};

        // First: Non-blocking drain of all pending requests
        // This ensures we batch requests that arrived during processing
        loop {
            match self.request_rx.try_recv() {
                Ok(req) => {
                    debug!("Received request {} (non-blocking)", req.request_id);
                    self.pending_queue.push_back(req);
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        // If we have work to do (pending requests or active sequences), don't wait
        if !self.pending_queue.is_empty() || !self.sequences.is_empty() {
            return;
        }

        // Otherwise, wait for first request with accumulation window
        let deadline = Duration::from_millis(self.config.accumulation_window_ms);
        match timeout(deadline, self.request_rx.recv()).await {
            Ok(Some(req)) => {
                debug!("Received request {} (blocking)", req.request_id);
                self.pending_queue.push_back(req);

                // After receiving first request, immediately drain any others
                // that arrived while we were waiting
                loop {
                    match self.request_rx.try_recv() {
                        Ok(req) => {
                            debug!("Received additional request {}", req.request_id);
                            self.pending_queue.push_back(req);
                        }
                        Err(_) => break,
                    }
                }
            }
            Ok(None) => {
                // Channel closed
            }
            Err(_) => {
                // Timeout - no requests
            }
        }
    }

    /// Assign sequence slots to pending requests
    fn assign_slots(&mut self) {
        while !self.pending_queue.is_empty() {
            // Peek at the request to check for prefix match before allocating
            let request = self.pending_queue.front().unwrap();

            // Step 1: Check for prefix match in RadixCache
            let match_result = self.radix_cache.match_prefix(&request.input_ids);

            // Step 2: Try to allocate a sequence slot
            let seq_id = match self.engine.alloc_sequence() {
                Ok(id) => id,
                Err(BatchError::NoAvailableSlots) => {
                    // No more slots available
                    break;
                }
                Err(e) => {
                    error!("Failed to allocate slot: {}", e);
                    break;
                }
            };

            // Now we can safely remove the request from the queue
            let request = self.pending_queue.pop_front().unwrap();
            let request_id = request.request_id;

            // Step 3: Create sequence state, potentially with prefix reuse
            let state = if match_result.has_match() {
                // Found a prefix match! Copy KV cache from existing sequence.
                let src_seq_id = match_result.reuse_seq_id.unwrap();
                let prefix_len = match_result.matched_len;
                let kv_end_pos = match_result.kv_end_pos;

                // Zero-copy KV cache sharing via llama.cpp
                // This adds seq_id to the KV cache cells' seq_id set without copying data
                self.engine.kv_cache_seq_cp(src_seq_id, seq_id, 0, kv_end_pos);

                // Lock the prefix node to prevent eviction while in use
                self.radix_cache.inc_lock_ref(&match_result.last_node);

                info!(
                    "Request {} reusing {} tokens from seq {} (saved {} prefill tokens)",
                    request_id, prefix_len, src_seq_id, prefix_len
                );

                SequenceState::new_with_prefix(
                    seq_id,
                    request,
                    match_result.last_node,
                    prefix_len,
                    kv_end_pos,
                )
            } else {
                // No prefix match - standard sequence creation
                debug!("Assigned slot {} to request {} (no prefix match)", seq_id, request_id);
                SequenceState::new(seq_id, request)
            };

            self.sequences.insert(seq_id, state);
        }
    }

    /// Form a batch from active sequences
    fn form_batch(&mut self) -> usize {
        let mut tokens_added = 0;
        let max_tokens = self.config.max_batch_size;
        let max_prefill = max_tokens - self.config.min_decode_budget;

        // Priority 1: Prefilling sequences (chunked)
        let seq_ids: Vec<BitNetSeqId> = self.sequences.keys().copied().collect();

        for seq_id in &seq_ids {
            if tokens_added >= max_prefill {
                break;
            }

            let state = self.sequences.get_mut(seq_id).unwrap();
            if !state.is_prefilling() {
                continue;
            }

            let remaining_budget = max_prefill - tokens_added;
            let chunk_size = remaining_budget.min(self.config.prefill_chunk_size);

            if let Some(chunk) = state.get_prefill_chunk(chunk_size) {
                let start_pos = state.position;

                for (i, &token) in chunk.iter().enumerate() {
                    let pos = start_pos + i as i32;
                    let is_last = i == chunk.len() - 1;

                    // Only output logits for last token of prefill
                    self.batch.add(token, pos, *seq_id, is_last);
                    tokens_added += 1;

                    if is_last {
                        // Store the ACTUAL batch position (0-indexed) where logits=true
                        // llama.cpp's llama_get_logits_ith expects the batch position
                        state.batch_idx = Some((tokens_added - 1) as i32);
                    }
                }

                // Update position
                state.position = start_pos + chunk.len() as i32;
            }
        }

        // Priority 2: Decoding sequences (1 token each)
        for seq_id in &seq_ids {
            if tokens_added >= max_tokens {
                break;
            }

            let state = self.sequences.get_mut(seq_id).unwrap();
            if !state.is_decoding() {
                continue;
            }

            // Skip if already processed in prefill phase (just transitioned to decode)
            // This can happen when prefill exhausts all tokens and transitions mid-batch
            if state.batch_idx.is_some() {
                continue;
            }

            let token = state.last_token;
            let pos = state.position;

            self.batch.add(token, pos, *seq_id, true);
            tokens_added += 1;

            // Store the ACTUAL batch position where logits=true
            state.batch_idx = Some((tokens_added - 1) as i32);
        }

        tokens_added
    }

    /// Sample tokens and distribute to sequences
    async fn sample_and_distribute(&mut self) {
        let seq_ids: Vec<BitNetSeqId> = self.sequences.keys().copied().collect();

        for seq_id in seq_ids {
            let state = self.sequences.get_mut(&seq_id).unwrap();

            // Skip if no logits were requested for this sequence
            let batch_idx = match state.batch_idx.take() {
                Some(idx) => idx,
                None => continue,
            };

            // Sample token
            let token_id = match self.engine.sample(batch_idx, &state.params) {
                Ok(t) => t,
                Err(e) => {
                    error!("Sampling failed for seq {}: {}", seq_id, e);
                    continue;
                }
            };

            // Check for EOS using engine's is_eos (handles all EOG tokens)
            let is_eos = self.engine.is_eos(token_id);

            // Record token
            state.add_generated_token(token_id, is_eos);

            // Update position for decode phase
            if state.is_decoding() {
                state.position += 1;
            }

            // Decode token to text using engine's tokenizer
            let text = self.engine.token_to_piece(token_id).unwrap_or_default();
            state.send_token(token_id, text, state.is_finished());
        }
    }

    /// Cleanup finished sequences
    fn cleanup_finished(&mut self) {
        let finished_ids: Vec<BitNetSeqId> = self
            .sequences
            .iter()
            .filter(|(_, s)| s.is_finished())
            .map(|(&id, _)| id)
            .collect();

        for seq_id in finished_ids {
            if let Some(mut state) = self.sequences.remove(&seq_id) {
                debug!(
                    "Sequence {} finished, generated {} tokens (prefix reused: {})",
                    seq_id,
                    state.generated_tokens.len(),
                    state.prefix_reused_len
                );

                // Insert completed sequence into RadixCache for future reuse
                // This includes both prompt and generated tokens
                let all_tokens: Vec<i32> = state.prompt_tokens.iter()
                    .chain(state.generated_tokens.iter())
                    .copied()
                    .collect();

                let newly_cached = self.radix_cache.insert(&all_tokens, seq_id);
                if newly_cached > 0 {
                    debug!(
                        "Cached {} new tokens from seq {} (total cached: {})",
                        newly_cached,
                        seq_id,
                        self.radix_cache.cached_tokens()
                    );
                }

                // Unlock the prefix node if we were reusing one
                if let Some(ref prefix_node) = state.prefix_node {
                    self.radix_cache.dec_lock_ref(prefix_node);
                }

                // If we cached tokens, keep the KV cache and sequence slot alive
                // for future prefix reuse. Otherwise, free everything.
                if newly_cached == 0 {
                    // No new tokens cached - safe to remove KV cache and free slot
                    self.engine.kv_cache_seq_rm(seq_id, -1, -1);
                    self.engine.free_sequence(seq_id);
                } else {
                    // Tokens were cached - keep KV cache for prefix reuse
                    // Don't free the sequence slot to prevent seq_id conflicts
                    debug!(
                        "Keeping seq {} KV cache for prefix reuse ({} tokens)",
                        seq_id, newly_cached
                    );
                }

                // Decode all tokens to text using engine's tokenizer
                let text = self
                    .engine
                    .detokenize(&state.generated_tokens)
                    .unwrap_or_else(|_| {
                        // Fallback: decode token by token
                        state
                            .generated_tokens
                            .iter()
                            .filter_map(|&t| self.engine.token_to_piece(t).ok())
                            .collect()
                    });
                state.complete(text);
            }
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            active_sequences: self.sequences.len(),
            pending_requests: self.pending_queue.len(),
            prefilling_sequences: self
                .sequences
                .values()
                .filter(|s| s.is_prefilling())
                .count(),
            decoding_sequences: self
                .sequences
                .values()
                .filter(|s| s.is_decoding())
                .count(),
            kv_cache_used: self.engine.kv_cache_used_cells() as usize,
            kv_cache_capacity: self.engine.kv_cache_capacity() as usize,
            radix_cache_tokens: self.radix_cache.cached_tokens(),
            radix_cache_enabled: self.radix_cache.is_enabled(),
        }
    }

    /// Get RadixCache statistics
    pub fn radix_cache_stats(&self) -> super::radix_cache::RadixCacheStats {
        self.radix_cache.stats()
    }

    /// Clear the RadixCache
    pub fn clear_radix_cache(&self) {
        self.radix_cache.clear();
    }
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub active_sequences: usize,
    pub pending_requests: usize,
    pub prefilling_sequences: usize,
    pub decoding_sequences: usize,
    pub kv_cache_used: usize,
    pub kv_cache_capacity: usize,
    /// Number of tokens cached in RadixCache
    pub radix_cache_tokens: usize,
    /// Whether RadixCache is enabled
    pub radix_cache_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    #[tokio::test]
    #[ignore = "Requires C++ library"]
    async fn test_scheduler_creation() {
        let engine = Arc::new(
            NativeBatchEngine::new("microsoft/bitnet-b1.58-2B-4T", None).unwrap(),
        );
        let config = SchedulerConfig::default();
        let (scheduler, handle) = BatchScheduler::new(config, engine);

        assert_eq!(scheduler.sequences.len(), 0);
        assert_eq!(scheduler.pending_queue.len(), 0);
    }
}
