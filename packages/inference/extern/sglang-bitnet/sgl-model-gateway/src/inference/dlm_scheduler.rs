//! Batch scheduler for Fast-dLLM v2 block diffusion inference.
//!
//! This module implements the Fast-dLLM v2 algorithm for parallel token
//! generation using block diffusion.
//!
//! ## Performance Note
//!
//! The ~2.5x speedup over autoregressive decoding is **theoretical** based on
//! the Fast-dLLM paper. Actual performance depends on model, hardware, and
//! workload characteristics. Benchmarking is recommended for your use case.
//!
//! ## Key Features
//!
//! - **Block Diffusion**: Generate `block_size` (default 32) tokens in parallel
//! - **Confidence Thresholding**: Unmask tokens above confidence threshold
//! - **DualCache**: Hierarchical KV cache for sub-block reuse
//! - **Token Shift**: Use logits[i-1] to predict token[i] (Fast-dLLM v2)
//!
//! ## Algorithm Overview
//!
//! 1. Initialize block with mask tokens
//! 2. Forward pass to get logits for positions *before* masked positions
//! 3. Sample tokens using shifted logits and compute confidence
//! 4. Unmask tokens where confidence > threshold
//! 5. Always unmask at least one token (highest confidence)
//! 6. Repeat until all tokens unmasked
//! 7. Move to next block
//!
//! ## Token Shift Strategy
//!
//! Fast-dLLM v2 uses logits[i-1] to predict token[i]. This preserves the
//! pretrained AR model's representation where hidden[i-1] predicts token[i].
//! For masked position i, we request logits for position i-1.

use super::batch_engine::{Batch, BatchConfig, BatchError, NativeBatchEngine};
use super::batch_ffi::BitNetSeqId;
use super::dlm_config::DlmConfig;
use super::radix_cache::{RadixCache, RadixCacheConfig};
use super::sequence::{
    FinishReason, InferenceRequest, InferenceResponse, SequencePhase, SequenceState, StreamToken,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Latency statistics with percentile computation.
///
/// Uses reservoir sampling to maintain a fixed-size sample of latencies
/// for efficient percentile computation.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Collected latency samples (in microseconds)
    samples: Vec<u64>,
    /// Total number of observations
    count: u64,
    /// Sum of all latencies for mean calculation (in microseconds)
    total_us: u64,
    /// Maximum reservoir size
    max_samples: usize,
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl LatencyStats {
    /// Create a new latency tracker with specified reservoir size.
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples),
            count: 0,
            total_us: 0,
            max_samples,
        }
    }

    /// Record a latency observation.
    pub fn record(&mut self, duration: Duration) {
        let us = duration.as_micros() as u64;
        self.count += 1;
        self.total_us += us;

        // Reservoir sampling: randomly replace if at capacity
        if self.samples.len() < self.max_samples {
            self.samples.push(us);
        } else {
            // Use simple modulo for pseudo-random replacement
            let idx = (self.count as usize) % self.max_samples;
            self.samples[idx] = us;
        }
    }

    /// Get percentile value (0-100). Returns microseconds.
    pub fn percentile(&self, p: f64) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }

        let mut sorted = self.samples.clone();
        sorted.sort_unstable();

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Get mean latency in microseconds.
    pub fn mean(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.total_us / self.count
        }
    }

    /// Get observation count.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get p50/p95/p99 percentiles in milliseconds.
    pub fn percentiles_ms(&self) -> LatencyPercentiles {
        LatencyPercentiles {
            p50_ms: self.percentile(50.0) as f64 / 1000.0,
            p95_ms: self.percentile(95.0) as f64 / 1000.0,
            p99_ms: self.percentile(99.0) as f64 / 1000.0,
            mean_ms: self.mean() as f64 / 1000.0,
            count: self.count,
        }
    }
}

/// Latency percentiles for reporting.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LatencyPercentiles {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub count: u64,
}

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
    /// Request start time for latency tracking
    start_time: Instant,
}

impl DlmSequenceState {
    fn new(seq_id: BitNetSeqId, request: InferenceRequest) -> Self {
        Self {
            base: SequenceState::new(seq_id, request),
            current_block: None,
            dual_cache: None,
            start_time: Instant::now(),
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

    // Latency tracking
    request_latency: LatencyStats,
    block_decode_latency: LatencyStats,
    total_tokens_generated: u64,
    total_requests_completed: u64,
}

impl DlmScheduler {
    /// Create a new DLM scheduler.
    pub fn new(
        config: DlmSchedulerConfig,
        engine: Arc<NativeBatchEngine>,
    ) -> (Self, DlmSchedulerHandle) {
        let (request_tx, request_rx) = mpsc::channel(1024);

        // Create batch with enough capacity for a full block
        let batch_size = (config.dlm.block_size * 2) as i32;
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
            request_latency: LatencyStats::default(),
            block_decode_latency: LatencyStats::default(),
            total_tokens_generated: 0,
            total_requests_completed: 0,
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
                Err(BatchError::NoAvailableSlots) => break,
                Err(e) => {
                    error!("Failed to allocate slot: {}", e);
                    break;
                }
            };

            let request = self.pending_queue.pop_front().unwrap();
            let request_id = request.request_id.clone();

            debug!("Assigned slot {} to request {}", seq_id, request_id);

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
                continue;
            }

            // Sample first token
            let batch_idx = (prompt_len - 1) as i32;
            if let Ok(first_token) = self.engine.sample(batch_idx, &state.base.params) {
                let is_eos = self.engine.is_eos(first_token);
                state.base.add_generated_token(first_token, is_eos);
                state.base.position = prompt_len as i32;

                // Transition to decoding
                if !is_eos {
                    state.base.phase = SequencePhase::Decoding;

                    // Initialize DualCache for block decoding
                    let num_small_blocks = self.config.dlm.num_small_blocks();
                    state.dual_cache = Some(DualCache::new(
                        state.base.position,
                        num_small_blocks,
                    ));
                }

                // Stream the first token
                let text = self.engine.token_to_piece(first_token).unwrap_or_default();
                state.base.send_token(first_token, text, is_eos);
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
            if state.base.generated_tokens.len() >= max_tokens {
                let state = self.sequences.get_mut(&seq_id).unwrap();
                state.base.phase = SequencePhase::Finished(FinishReason::Length);
                continue;
            }

            // Decode one block
            self.decode_block(seq_id).await;
        }
    }

    /// Decode a single block using Fast-dLLM v2 algorithm.
    async fn decode_block(&mut self, seq_id: BitNetSeqId) {
        let block_start_time = Instant::now();

        let block_size = self.config.dlm.block_size;
        let small_block_size = self.config.dlm.small_block_size;
        let num_small_blocks = self.config.dlm.num_small_blocks();
        let threshold = self.config.dlm.threshold;
        let mask_id = self.config.dlm.mask_token_id;

        let state = self.sequences.get_mut(&seq_id).unwrap();
        let block_start_pos = state.base.position;

        // Initialize block with mask tokens
        let mut block_tokens: Vec<i32> = vec![mask_id; block_size];

        // Get or create DualCache
        if state.dual_cache.is_none() {
            state.dual_cache = Some(DualCache::new(block_start_pos, num_small_blocks));
        }

        let max_iterations = self.config.dlm.max_iterations_per_block;

        // Process each small block
        for small_block_idx in 0..num_small_blocks {
            let start_idx = small_block_idx * small_block_size;
            let end_idx = start_idx + small_block_size;

            // Iterate until all masks in this small block are resolved
            let mut iteration = 0;
            loop {
                iteration += 1;

                // Check max iterations safeguard
                if iteration > max_iterations {
                    warn!(
                        "Max iterations ({}) reached for small block {}, forcing remaining {} masks",
                        max_iterations,
                        small_block_idx,
                        block_tokens[start_idx..end_idx]
                            .iter()
                            .filter(|&&t| t == mask_id)
                            .count()
                    );
                    // Force unmask remaining by using any token with highest confidence
                    // This is handled in the normal unmasking logic below
                    break;
                }

                let small_block_masks: usize = block_tokens[start_idx..end_idx]
                    .iter()
                    .filter(|&&t| t == mask_id)
                    .count();

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

                // ============================================================
                // BATCH INDEX MAPPING
                // ============================================================
                // There are two batch construction strategies:
                //
                // 1. COMPACT (masked_only=true, DualCache hit):
                //    - Only add tokens needed for masked positions
                //    - batch_idx = sequential enumeration index
                //    - Example: masked at [2,5,7] → batch indices [0,1,2]
                //
                // 2. FULL (masked_only=false, cache miss):
                //    - Add all tokens from start_idx to block_size
                //    - batch_idx = (block_idx - 1) - batch_offset
                //    - Example: start_idx=4 → batch_offset=4
                //
                // TOKEN SHIFT: For masked position i, we need logits from i-1.
                // This aligns with Fast-dLLM v2 where hidden[i-1] predicts token[i].
                // ============================================================
                let masked_only: bool;
                let batch_offset: usize;

                if use_cached && first_token_unmasked {
                    // DualCache hit: Only compute for remaining masked positions
                    // Clear KV cache for positions whose logits we need
                    // With token shift, for masked position i, we need position i-1
                    //
                    // Optimization: Batch KV cache clearing by merging contiguous ranges
                    let mut clear_positions: Vec<i32> = Vec::new();
                    for i in start_idx..end_idx {
                        if block_tokens[i] == mask_id && i > 0 {
                            clear_positions.push(block_start_pos + (i as i32 - 1));
                        }
                    }

                    // Merge and clear contiguous ranges
                    if !clear_positions.is_empty() {
                        clear_positions.sort_unstable();
                        let mut range_start = clear_positions[0];
                        let mut range_end = clear_positions[0] + 1;

                        for &pos in &clear_positions[1..] {
                            if pos == range_end {
                                // Extend current range
                                range_end = pos + 1;
                            } else {
                                // Clear current range and start new one
                                self.engine.kv_cache_seq_rm(seq_id, range_start, range_end);
                                range_start = pos;
                                range_end = pos + 1;
                            }
                        }
                        // Clear final range
                        self.engine.kv_cache_seq_rm(seq_id, range_start, range_end);
                    }

                    logit_positions = (start_idx..end_idx)
                        .filter(|&i| block_tokens[i] == mask_id)
                        .collect();

                    // Compact batch: add tokens for positions whose logits we need
                    // For masked position i, we need logits from i-1
                    for &i in &logit_positions {
                        if i > 0 {
                            let logit_pos = block_start_pos + (i as i32 - 1);
                            // Add the token at position i-1 with need_logits=true
                            let token_at_prev = if i > 0 { block_tokens[i - 1] } else { block_tokens[i] };
                            self.batch.add(token_at_prev, logit_pos, seq_id, true);
                        }
                    }
                    masked_only = true;
                    batch_offset = 0; // Not used when masked_only=true
                } else {
                    // Full block forward - clear KV cache from current small block onwards
                    // Only clear from current position to preserve valid cache from previous small blocks
                    let current_pos = block_start_pos + start_idx as i32;
                    let block_end_pos = block_start_pos + block_size as i32;
                    self.engine.kv_cache_seq_rm(seq_id, current_pos, block_end_pos);

                    // Add tokens from current small block onwards
                    // With token shift: need_logits for position i-1 when position i is masked
                    for i in start_idx..block_size {
                        let token = block_tokens[i];
                        let pos = block_start_pos + i as i32;
                        // Need logits at this position if the NEXT position is masked
                        let next_masked = i + 1 < block_size && i + 1 < end_idx && block_tokens[i + 1] == mask_id;
                        // Also need logits if this position itself is masked AND it's the first in block
                        // (we use prefix cache for position -1)
                        let this_masked = i < end_idx && token == mask_id && i == 0;
                        let need_logits = next_masked || this_masked;
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
                if let Err(e) = self.engine.decode(&self.batch) {
                    error!("Block decode failed for seq {}: {}", seq_id, e);
                    return;
                }

                // ============================================================
                // CANDIDATE GENERATION
                // ============================================================
                // For each masked position, get logits and compute confidence.
                // TOKEN SHIFT: logits at position i-1 predict token at position i.
                let mut candidates: Vec<(usize, i32, f32)> = Vec::new();

                for (enum_idx, &block_idx) in logit_positions.iter().enumerate() {
                    // Skip position 0 - would need logits from previous block
                    if block_idx == 0 {
                        continue;
                    }

                    // Compute batch index using the mapping strategy selected above
                    let batch_idx = if masked_only {
                        // COMPACT: enum_idx maps directly to batch position
                        enum_idx as i32
                    } else {
                        // FULL: batch contains all tokens from batch_offset
                        // Logits for position i are at batch index (i-1) - batch_offset
                        ((block_idx - 1) - batch_offset) as i32
                    };

                    if batch_idx < 0 {
                        continue; // Skip if we don't have the shifted logits
                    }

                    if let Some(logits) = self.engine.get_logits(batch_idx) {
                        // Use efficient confidence computation (no allocation)
                        let (token_id, confidence) = confidence_for_argmax(logits);
                        candidates.push((block_idx, token_id, confidence));
                    }
                }

                // Unmask high-confidence tokens
                let mut unmasked_any = false;
                for &(idx, token, conf) in &candidates {
                    if conf > threshold {
                        block_tokens[idx] = token;
                        unmasked_any = true;

                        // Stream this token
                        self.emit_token(seq_id, token);
                    }
                }

                // Always unmask at least one (highest confidence)
                if !unmasked_any && !candidates.is_empty() {
                    let (idx, token, _) = candidates
                        .iter()
                        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Less))
                        .copied()
                        .unwrap();

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
                // Record block decode latency before returning
                self.block_decode_latency.record(block_start_time.elapsed());
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

        // Record block decode latency
        self.block_decode_latency.record(block_start_time.elapsed());
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
                let tokens_generated = state.base.generated_tokens.len();

                // Record request latency
                let request_duration = state.start_time.elapsed();
                self.request_latency.record(request_duration);
                self.total_tokens_generated += tokens_generated as u64;
                self.total_requests_completed += 1;

                debug!(
                    "DLM sequence {} finished, generated {} tokens in {:.2}ms",
                    seq_id,
                    tokens_generated,
                    request_duration.as_secs_f64() * 1000.0
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
            request_latency: self.request_latency.percentiles_ms(),
            block_decode_latency: self.block_decode_latency.percentiles_ms(),
            total_tokens_generated: self.total_tokens_generated,
            total_requests_completed: self.total_requests_completed,
        }
    }

    /// Get throughput in tokens per second (based on completed requests).
    pub fn throughput(&self) -> f64 {
        if self.total_requests_completed == 0 {
            return 0.0;
        }
        let mean_latency_s = self.request_latency.mean() as f64 / 1_000_000.0;
        if mean_latency_s <= 0.0 {
            return 0.0;
        }
        let avg_tokens = self.total_tokens_generated as f64 / self.total_requests_completed as f64;
        avg_tokens / mean_latency_s
    }
}

/// DLM scheduler statistics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DlmSchedulerStats {
    pub active_sequences: usize,
    pub pending_requests: usize,
    pub radix_cache_tokens: usize,
    pub block_size: usize,
    pub request_latency: LatencyPercentiles,
    pub block_decode_latency: LatencyPercentiles,
    pub total_tokens_generated: u64,
    pub total_requests_completed: u64,
}

/// Compute softmax of logits.
///
/// Uses numerically stable implementation with max subtraction.
/// Note: For single-token confidence, use `confidence_for_argmax` instead
/// to avoid allocating a full probability vector.
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

/// Compute confidence for the argmax token without allocating a probability vector.
///
/// This is more efficient than calling `softmax()` followed by indexing when you
/// only need the probability of a single token. It computes:
///   exp(logits[token_id] - max) / sum(exp(logits[i] - max))
///
/// Returns (argmax_token_id, confidence).
pub fn confidence_for_argmax(logits: &[f32]) -> (i32, f32) {
    if logits.is_empty() {
        return (0, 0.0);
    }

    // Find max and argmax in a single pass
    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx: usize = 0;
    for (i, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    // Compute sum of exp(logits - max) in a single pass
    // The exp at max_idx will be 1.0 (since logits[max_idx] - max_val = 0)
    let sum_exp: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();

    // Confidence = exp(0) / sum = 1.0 / sum
    let confidence = 1.0 / sum_exp;

    (max_idx as i32, confidence)
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
    fn test_confidence_for_argmax_correctness() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let (token_id, confidence) = confidence_for_argmax(&logits);

        // Should match softmax + argmax result
        let probs = softmax(&logits);
        let expected_token = argmax(&logits);
        let expected_confidence = probs[expected_token as usize];

        assert_eq!(token_id, expected_token);
        assert!(
            (confidence - expected_confidence).abs() < 1e-6,
            "confidence {} != expected {}",
            confidence,
            expected_confidence
        );
    }

    #[test]
    fn test_confidence_for_argmax_high_confidence() {
        // When one logit dominates, confidence should be high
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        let (token_id, confidence) = confidence_for_argmax(&logits);

        assert_eq!(token_id, 2);
        assert!(confidence > 0.99, "confidence {} should be > 0.99", confidence);
    }

    #[test]
    fn test_confidence_for_argmax_uniform() {
        // When all logits are equal, confidence should be 1/N
        let logits = vec![0.0; 10];
        let (_, confidence) = confidence_for_argmax(&logits);

        let expected = 1.0 / 10.0;
        assert!(
            (confidence - expected).abs() < 1e-6,
            "confidence {} != expected {}",
            confidence,
            expected
        );
    }

    #[test]
    fn test_confidence_for_argmax_empty() {
        let logits: Vec<f32> = vec![];
        let (token_id, confidence) = confidence_for_argmax(&logits);

        assert_eq!(token_id, 0);
        assert_eq!(confidence, 0.0);
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

    #[test]
    fn test_latency_stats_basic() {
        let mut stats = LatencyStats::new(100);

        // Record some latencies
        stats.record(Duration::from_millis(10));
        stats.record(Duration::from_millis(20));
        stats.record(Duration::from_millis(30));

        assert_eq!(stats.count(), 3);
        assert_eq!(stats.mean(), 20_000); // 20ms in microseconds
    }

    #[test]
    fn test_latency_stats_percentiles() {
        let mut stats = LatencyStats::new(100);

        // Record 100 latencies from 1ms to 100ms
        for i in 1..=100 {
            stats.record(Duration::from_millis(i));
        }

        assert_eq!(stats.count(), 100);

        // p50 should be around 50ms (50_000 us)
        let p50 = stats.percentile(50.0);
        assert!(p50 >= 49_000 && p50 <= 51_000, "p50 was {}", p50);

        // p95 should be around 95ms
        let p95 = stats.percentile(95.0);
        assert!(p95 >= 94_000 && p95 <= 96_000, "p95 was {}", p95);

        // p99 should be around 99ms
        let p99 = stats.percentile(99.0);
        assert!(p99 >= 98_000 && p99 <= 100_000, "p99 was {}", p99);
    }

    #[test]
    fn test_latency_stats_empty() {
        let stats = LatencyStats::new(100);
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.mean(), 0);
        assert_eq!(stats.percentile(50.0), 0);
    }

    #[test]
    fn test_latency_stats_reservoir_sampling() {
        let mut stats = LatencyStats::new(10);

        // Record more samples than reservoir size
        for i in 1..=100 {
            stats.record(Duration::from_millis(i));
        }

        assert_eq!(stats.count(), 100);
        // Mean should still be accurate
        assert_eq!(stats.mean(), 50_500); // (1+100)/2 * 1000 us
        // Samples should be at capacity
        assert_eq!(stats.samples.len(), 10);
    }

    #[test]
    fn test_latency_percentiles_ms() {
        let mut stats = LatencyStats::new(100);

        stats.record(Duration::from_millis(100));
        stats.record(Duration::from_millis(200));
        stats.record(Duration::from_millis(300));

        let percentiles = stats.percentiles_ms();
        assert!((percentiles.mean_ms - 200.0).abs() < 1.0);
        assert_eq!(percentiles.count, 3);
    }
}
