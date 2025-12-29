//! Batch scheduler for continuous batching.
//!
//! The scheduler manages request queuing, batch formation, and token generation
//! for multiple concurrent sequences.

use super::batch_engine::{Batch, BatchConfig, BatchError, NativeBatchEngine};
use super::batch_ffi::BitNetSeqId;
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
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 512,
            max_sequences: 16,
            accumulation_window_ms: 10,
            prefill_chunk_size: 128,
            min_decode_budget: 8,
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

        let scheduler = Self {
            config,
            engine,
            request_rx,
            sequences: HashMap::new(),
            pending_queue: VecDeque::new(),
            batch,
            eos_token_id,
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

        let deadline = Duration::from_millis(self.config.accumulation_window_ms);

        // Try to receive as many requests as possible within the window
        loop {
            match timeout(deadline, self.request_rx.recv()).await {
                Ok(Some(req)) => {
                    debug!("Received request {}", req.request_id);
                    self.pending_queue.push_back(req);
                }
                Ok(None) => {
                    // Channel closed
                    break;
                }
                Err(_) => {
                    // Timeout - accumulation window expired
                    break;
                }
            }

            // Don't wait forever if we have pending work
            if !self.sequences.is_empty() {
                break;
            }
        }
    }

    /// Assign sequence slots to pending requests
    fn assign_slots(&mut self) {
        while !self.pending_queue.is_empty() {
            // Try to allocate a slot
            match self.engine.alloc_sequence() {
                Ok(seq_id) => {
                    let request = self.pending_queue.pop_front().unwrap();
                    let request_id = request.request_id;

                    let state = SequenceState::new(seq_id, request);
                    self.sequences.insert(seq_id, state);

                    debug!("Assigned slot {} to request {}", seq_id, request_id);
                }
                Err(BatchError::NoAvailableSlots) => {
                    // No more slots available
                    break;
                }
                Err(e) => {
                    error!("Failed to allocate slot: {}", e);
                    break;
                }
            }
        }
    }

    /// Form a batch from active sequences
    fn form_batch(&mut self) -> usize {
        let mut tokens_added = 0;
        let max_tokens = self.config.max_batch_size;
        let max_prefill = max_tokens - self.config.min_decode_budget;

        // Track batch indices for logits
        let mut batch_idx = 0;

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

                    if is_last {
                        state.batch_idx = Some(batch_idx);
                        batch_idx += 1;
                    }

                    tokens_added += 1;
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

            let token = state.last_token;
            let pos = state.position;

            self.batch.add(token, pos, *seq_id, true);
            state.batch_idx = Some(batch_idx);
            batch_idx += 1;

            tokens_added += 1;
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

            let is_eos = token_id == self.eos_token_id;

            // Record token
            state.add_generated_token(token_id, is_eos);

            // Update position for decode phase
            if state.is_decoding() {
                state.position += 1;
            }

            // Send streaming token
            // TODO: Decode token to text using tokenizer
            let text = format!("<token:{}>", token_id); // Placeholder
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
                    "Sequence {} finished, generated {} tokens",
                    seq_id,
                    state.generated_tokens.len()
                );

                // Clear KV cache
                self.engine.kv_cache_seq_rm(seq_id, -1, -1);

                // Free slot
                self.engine.free_sequence(seq_id);

                // Send final response
                // TODO: Decode tokens to text
                let text = state
                    .generated_tokens
                    .iter()
                    .map(|t| format!("<token:{}>", t))
                    .collect::<Vec<_>>()
                    .join("");
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
        }
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
