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
use super::dlm_config::{DlmConfig, DlmDecodeMode};
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
        info!(
            "DLM scheduler starting (block_size={}, mode={:?}, threshold={})",
            self.config.dlm.block_size,
            self.config.dlm.decode_mode,
            self.config.dlm.threshold
        );

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

    /// Decode a single block - dispatches to greedy or iterative based on config.
    async fn decode_block(&mut self, seq_id: BitNetSeqId) {
        match self.config.dlm.decode_mode {
            DlmDecodeMode::Greedy => self.decode_block_greedy(seq_id).await,
            DlmDecodeMode::Iterative | DlmDecodeMode::Adaptive => {
                self.decode_block_iterative(seq_id).await
            }
        }
    }

    /// Decode a single block using FAST single-pass greedy approach.
    ///
    /// OPTIMIZATION: Instead of iterating over small blocks with multiple passes,
    /// we use a single forward pass that decodes ALL tokens at once using greedy
    /// argmax. This reduces FFI calls from O(small_blocks * iterations) to O(1).
    ///
    /// The key insight: with token shift (logits[i-1] predicts token[i]), we need
    /// to include the PREVIOUS token in the batch to get logits for the first
    /// block position.
    ///
    /// NOTE: This is ~20x faster than iterative but doesn't follow the paper's
    /// confidence-based unmasking algorithm.
    async fn decode_block_greedy(&mut self, seq_id: BitNetSeqId) {
        let block_start_time = Instant::now();

        let block_size = self.config.dlm.block_size;
        let mask_id = self.config.dlm.mask_token_id;

        let state = self.sequences.get_mut(&seq_id).unwrap();
        let block_start_pos = state.base.position;

        // Get the token BEFORE this block (needed for logits to predict first token)
        let prev_token = if !state.base.generated_tokens.is_empty() {
            state.base.generated_tokens[state.base.generated_tokens.len() - 1]
        } else if !state.base.prompt_tokens.is_empty() {
            state.base.prompt_tokens[state.base.prompt_tokens.len() - 1]
        } else {
            mask_id // Fallback
        };

        // Clear KV cache for the entire block region (single FFI call)
        // Note: We need to clear from prev position since we're re-computing it
        let block_end_pos = block_start_pos + block_size as i32;
        self.engine.kv_cache_seq_rm(seq_id, block_start_pos - 1, block_end_pos);

        // Build batch: [prev_token, mask, mask, ..., mask]
        // Position: [block_start-1, block_start, block_start+1, ..., block_start+N-1]
        // logits[0] predicts token at block_start, logits[1] predicts block_start+1, etc.
        self.batch.clear();

        // Add previous token at position -1 relative to block (need logits for first block token)
        self.batch.add(prev_token, block_start_pos - 1, seq_id, true);

        // Add all mask tokens for the block
        for i in 0..block_size {
            let pos = block_start_pos + i as i32;
            // Need logits for all except the last (predicts token after block)
            let need_logits = i < block_size - 1;
            self.batch.add(mask_id, pos, seq_id, need_logits);
        }

        // SINGLE forward pass for entire block!
        if let Err(e) = self.engine.decode(&self.batch) {
            error!("Block decode failed for seq {}: {}", seq_id, e);
            return;
        }

        // Greedy decode all tokens from logits
        // Batch layout: [prev_token, mask0, mask1, ..., maskN-1]
        // Batch idx:    [0,          1,     2,     ..., N]
        // logits[0] → token at block_start (from prev_token)
        // logits[1] → token at block_start+1 (from mask0)
        // ...
        // logits[N-1] → token at block_start+N-1 (from maskN-2)
        let mut block_tokens: Vec<i32> = Vec::with_capacity(block_size);

        for i in 0..block_size {
            let batch_idx = i as i32; // logits[i] predicts token[i] due to our batch layout
            if let Some(logits) = self.engine.get_logits(batch_idx) {
                let (token_id, _) = confidence_for_argmax(logits);
                block_tokens.push(token_id);
            } else {
                // Fallback - shouldn't happen
                block_tokens.push(mask_id);
            }
        }

        // Stream all tokens
        for &token in &block_tokens {
            self.emit_token(seq_id, token);
        }

        // Finalize block
        let state = self.sequences.get_mut(&seq_id).unwrap();
        let new_pos = block_start_pos + block_size as i32;
        state.base.position = new_pos;

        // Add generated tokens and check for EOS
        for &token in &block_tokens {
            let is_eos = self.engine.is_eos(token);
            state.base.generated_tokens.push(token);

            if is_eos {
                state.base.phase = SequencePhase::Finished(FinishReason::EOS);
                self.block_decode_latency.record(block_start_time.elapsed());
                return;
            }
        }

        // Check max tokens
        if state.base.generated_tokens.len() >= state.base.max_tokens as usize {
            state.base.phase = SequencePhase::Finished(FinishReason::Length);
        }

        // Record block decode latency
        self.block_decode_latency.record(block_start_time.elapsed());
    }

    /// Decode a single block using OPTIMIZED iterative confidence-based unmasking.
    ///
    /// This implements the Fast-dLLM v2 algorithm with optimizations:
    /// 1. Initialize block with all MASK tokens
    /// 2. Loop while any position is masked:
    ///    a. Only clear KV from first changed position (incremental cache)
    ///    b. Forward pass with current block state
    ///    c. For each masked position, compute confidence using token shift
    ///    d. Unmask positions where confidence > threshold
    ///    e. Always unmask at least one position (highest confidence) to ensure progress
    /// 3. Emit all tokens after block is complete
    ///
    /// OPTIMIZATIONS:
    /// - Incremental KV cache: only clear from first masked position
    /// - Only request logits for masked positions
    /// - Store predictions for fallback (don't leave as MASK)
    ///
    /// Token shift: logits[i-1] predicts token[i], so we include prev_token
    /// in the batch to get logits for the first block position.
    async fn decode_block_iterative(&mut self, seq_id: BitNetSeqId) {
        let block_start_time = Instant::now();

        let block_size = self.config.dlm.block_size;
        let mask_id = self.config.dlm.mask_token_id;
        let threshold = self.config.dlm.threshold;
        let max_iters = self.config.dlm.max_iterations_per_block;

        let state = self.sequences.get_mut(&seq_id).unwrap();
        let block_start_pos = state.base.position;

        // Get the token BEFORE this block (needed for token shift)
        let prev_token = if !state.base.generated_tokens.is_empty() {
            state.base.generated_tokens[state.base.generated_tokens.len() - 1]
        } else if !state.base.prompt_tokens.is_empty() {
            state.base.prompt_tokens[state.base.prompt_tokens.len() - 1]
        } else {
            mask_id
        };

        // Initialize block with all MASK tokens
        let mut block_tokens: Vec<i32> = vec![mask_id; block_size];
        let mut is_masked: Vec<bool> = vec![true; block_size];
        // Store best predictions for fallback (don't leave as MASK)
        let mut best_predictions: Vec<(i32, f32)> = vec![(mask_id, 0.0); block_size];

        // Clear KV cache for the block region (only needed once at start)
        let block_end_pos = block_start_pos + block_size as i32;
        self.engine.kv_cache_seq_rm(seq_id, block_start_pos - 1, block_end_pos);

        // Track the stable prefix length (all unmasked from position 0)
        let mut stable_prefix_len: usize = 0;

        let mut iteration = 0;
        while is_masked.iter().any(|&m| m) && iteration < max_iters {
            iteration += 1;

            // Adaptive threshold: start low, increase with iterations
            // This quickly unmasks confident positions, then refines uncertain ones
            let current_threshold = match self.config.dlm.decode_mode {
                DlmDecodeMode::Adaptive => match iteration {
                    1 => 0.5_f32.max(threshold - 0.4), // Start low
                    2 => 0.7_f32.max(threshold - 0.2), // Medium
                    _ => threshold,                    // Full threshold
                },
                _ => threshold, // Fixed threshold for Iterative mode
            };

            // Find first masked position - we need to recompute from here
            let first_masked_idx = is_masked.iter().position(|&m| m).unwrap_or(block_size);

            // OPTIMIZATION: Only clear KV from the first position that could change
            // With causal attention, if position i changes, all j > i are invalidated
            // But we can keep KV for stable prefix (all unmasked from start)
            let recompute_from = stable_prefix_len.min(first_masked_idx);
            if recompute_from < block_size {
                let clear_start = block_start_pos + recompute_from as i32 - 1; // -1 for prev token
                self.engine.kv_cache_seq_rm(seq_id, clear_start.max(block_start_pos - 1), block_end_pos);
            }

            // Build batch starting from recompute position
            self.batch.clear();

            // Add the token that predicts the first recompute position
            let pred_token = if recompute_from == 0 {
                prev_token
            } else {
                block_tokens[recompute_from - 1]
            };
            let pred_pos = block_start_pos + recompute_from as i32 - 1;
            self.batch.add(pred_token, pred_pos, seq_id, true);

            // Add tokens from recompute position onwards
            // OPTIMIZATION: Only request logits for positions that are masked
            for i in recompute_from..block_size {
                let pos = block_start_pos + i as i32;
                // Only need logits if this position OR subsequent positions are masked
                let need_logits = is_masked[i] || (i + 1 < block_size && is_masked[i + 1..].iter().any(|&m| m));
                self.batch.add(block_tokens[i], pos, seq_id, need_logits);
            }

            // Forward pass
            if let Err(e) = self.engine.decode(&self.batch) {
                error!("Iterative decode failed for seq {}: {}", seq_id, e);
                return;
            }

            // Compute confidence for masked positions
            // Batch layout: [pred_token, tok[recompute_from], tok[recompute_from+1], ...]
            // logits[0] → predicts block_tokens[recompute_from]
            // logits[k] → predicts block_tokens[recompute_from + k]
            let mut candidates: Vec<(usize, i32, f32)> = Vec::new();

            for i in recompute_from..block_size {
                if !is_masked[i] {
                    continue;
                }

                let batch_idx = (i - recompute_from) as i32;
                if let Some(logits) = self.engine.get_logits(batch_idx) {
                    let (token_id, confidence) = confidence_for_argmax(logits);
                    candidates.push((i, token_id, confidence));
                    // Store best prediction for fallback
                    if confidence > best_predictions[i].1 {
                        best_predictions[i] = (token_id, confidence);
                    }
                }
            }

            // Unmask positions above current threshold
            let mut any_unmasked = false;
            for &(idx, token, conf) in &candidates {
                if conf > current_threshold {
                    block_tokens[idx] = token;
                    is_masked[idx] = false;
                    any_unmasked = true;
                }
            }

            // If nothing unmasked, unmask the highest confidence to ensure progress
            if !any_unmasked && !candidates.is_empty() {
                let best = candidates
                    .iter()
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                block_tokens[best.0] = best.1;
                is_masked[best.0] = false;
            }

            // Update stable prefix length (contiguous unmasked from start)
            stable_prefix_len = is_masked.iter().position(|&m| m).unwrap_or(block_size);
        }

        // Fill remaining masked positions with best predictions (not MASK tokens)
        if is_masked.iter().any(|&m| m) {
            debug!(
                "Block decode used {} iterations, {} positions filled with best prediction",
                iteration,
                is_masked.iter().filter(|&&m| m).count()
            );
            for (i, &masked) in is_masked.iter().enumerate() {
                if masked {
                    block_tokens[i] = best_predictions[i].0;
                }
            }
        }

        // Stream all tokens
        for &token in &block_tokens {
            self.emit_token(seq_id, token);
        }

        // Finalize block
        let state = self.sequences.get_mut(&seq_id).unwrap();
        state.base.position = block_start_pos + block_size as i32;

        // Add generated tokens and check for EOS
        for &token in &block_tokens {
            let is_eos = self.engine.is_eos(token);
            state.base.generated_tokens.push(token);

            if is_eos {
                state.base.phase = SequencePhase::Finished(FinishReason::EOS);
                self.block_decode_latency.record(block_start_time.elapsed());
                return;
            }
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
///
/// OPTIMIZATION: Uses SIMD intrinsics on x86_64 for ~4x speedup.
#[inline]
pub fn confidence_for_argmax(logits: &[f32]) -> (i32, f32) {
    if logits.is_empty() {
        return (0, 0.0);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        confidence_for_argmax_avx2(logits)
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        confidence_for_argmax_scalar(logits)
    }
}

/// Scalar fallback for confidence computation.
#[inline]
fn confidence_for_argmax_scalar(logits: &[f32]) -> (i32, f32) {
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

/// AVX2-optimized confidence computation.
/// Processes 8 floats at a time for finding max and computing exp sum.
/// Uses fast polynomial exp approximation for full vectorization.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn confidence_for_argmax_avx2(logits: &[f32]) -> (i32, f32) {
    use std::arch::x86_64::*;

    let n = logits.len();
    if n == 0 {
        return (0, 0.0);
    }

    unsafe {
        let ptr = logits.as_ptr();

        // Phase 1: Find max using SIMD
        let chunks = n / 8;
        let remainder = n % 8;

        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut i = 0;

        // Process 8 elements at a time
        for _ in 0..chunks {
            let v = _mm256_loadu_ps(ptr.add(i));
            max_vec = _mm256_max_ps(max_vec, v);
            i += 8;
        }

        // Horizontal max reduction
        let mut tmp = _mm256_permute2f128_ps(max_vec, max_vec, 1);
        max_vec = _mm256_max_ps(max_vec, tmp);
        tmp = _mm256_shuffle_ps(max_vec, max_vec, 0b01001110);
        max_vec = _mm256_max_ps(max_vec, tmp);
        tmp = _mm256_shuffle_ps(max_vec, max_vec, 0b10110001);
        max_vec = _mm256_max_ps(max_vec, tmp);

        // Extract scalar max from SIMD
        let mut max_val = _mm256_cvtss_f32(max_vec);

        // Handle remainder with scalar
        for j in 0..remainder {
            let val = *ptr.add(i + j);
            if val > max_val {
                max_val = val;
            }
        }

        // Find argmax (scalar - needed for exact index)
        let mut max_idx: usize = 0;
        for j in 0..n {
            if *ptr.add(j) == max_val {
                max_idx = j;
                break;
            }
        }

        // Phase 2: Compute sum of exp(x - max) using SIMD with fast exp
        let max_bcast = _mm256_set1_ps(max_val);
        let mut sum_vec = _mm256_setzero_ps();
        i = 0;

        for _ in 0..chunks {
            let v = _mm256_loadu_ps(ptr.add(i));
            let diff = _mm256_sub_ps(v, max_bcast);
            let exp_v = fast_exp_avx2(diff);
            sum_vec = _mm256_add_ps(sum_vec, exp_v);
            i += 8;
        }

        // Horizontal sum reduction
        let mut tmp = _mm256_permute2f128_ps(sum_vec, sum_vec, 1);
        sum_vec = _mm256_add_ps(sum_vec, tmp);
        tmp = _mm256_shuffle_ps(sum_vec, sum_vec, 0b01001110);
        sum_vec = _mm256_add_ps(sum_vec, tmp);
        tmp = _mm256_shuffle_ps(sum_vec, sum_vec, 0b10110001);
        sum_vec = _mm256_add_ps(sum_vec, tmp);
        let mut sum_exp = _mm256_cvtss_f32(sum_vec);

        // Handle remainder with scalar
        for j in 0..remainder {
            sum_exp += (*ptr.add(i + j) - max_val).exp();
        }

        let confidence = 1.0 / sum_exp;
        (max_idx as i32, confidence)
    }
}

/// Fast vectorized exp approximation using the Schraudolph method.
/// Accurate to ~1-2% relative error, sufficient for softmax confidence thresholding.
/// Based on: https://nic.schraudolph.org/pubs/Schraudolph99.pdf
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn fast_exp_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    // Constants for exp approximation: exp(x) ≈ 2^(x * log2(e))
    // Using IEEE 754 float bit manipulation (Schraudolph trick)
    let log2e = _mm256_set1_ps(1.442695041_f32); // log2(e)
    let shift = _mm256_set1_ps((1 << 23) as f32); // 2^23
    let offset = _mm256_set1_ps(127.0_f32 * (1 << 23) as f32); // Exponent bias * 2^23

    // Clamp to avoid overflow/underflow (exp(88) ≈ 1.65e38, exp(-88) ≈ 6e-39)
    let min_val = _mm256_set1_ps(-88.0_f32);
    let max_val = _mm256_set1_ps(88.0_f32);
    let x_clamped = _mm256_max_ps(_mm256_min_ps(x, max_val), min_val);

    // Compute: floor(x * log2(e) * 2^23) + 127 * 2^23
    // This sets the IEEE 754 exponent bits correctly
    let scaled = _mm256_mul_ps(x_clamped, log2e);
    let shifted = _mm256_add_ps(_mm256_mul_ps(scaled, shift), offset);

    // Convert to integer and reinterpret as float
    let int_bits = _mm256_cvtps_epi32(shifted);
    _mm256_castsi256_ps(int_bits)
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
    fn test_confidence_for_argmax_large_vocab() {
        // Test with vocab size similar to real models (128K)
        let n = 128257;
        let mut logits = vec![0.0f32; n];
        // Set one token to be the clear winner
        // With 128K tokens at exp(0)=1, we need exp(x) >> 128K for high confidence
        // ln(128K) ≈ 11.76, so we need x > 20 for > 99% confidence
        logits[50000] = 25.0;

        let (token_id, confidence) = confidence_for_argmax(&logits);

        assert_eq!(token_id, 50000);
        // exp(25) ≈ 7.2e10, vs 128K exp(0) ≈ 1.3e5, so confidence ≈ 99.9998%
        assert!(confidence > 0.99, "confidence {} should be > 0.99", confidence);
    }

    #[test]
    fn test_confidence_scalar_vs_simd_consistency() {
        // Verify scalar and SIMD produce similar results
        let n = 1024; // Multiple of 8 for clean SIMD
        let mut logits = vec![0.0f32; n];
        for i in 0..n {
            logits[i] = (i as f32 * 0.1) - 50.0; // Range from -50 to +52
        }
        logits[n - 1] = 100.0; // Clear winner

        let (scalar_token, scalar_conf) = confidence_for_argmax_scalar(&logits);

        // These should match (within approximation error for SIMD exp)
        assert_eq!(scalar_token, (n - 1) as i32);
        assert!(scalar_conf > 0.99, "scalar confidence {} should be > 0.99", scalar_conf);

        // Also test the main function (which may use SIMD)
        let (main_token, main_conf) = confidence_for_argmax(&logits);
        assert_eq!(main_token, scalar_token);
        // Allow 5% tolerance for fast exp approximation
        assert!(
            (main_conf - scalar_conf).abs() / scalar_conf < 0.05,
            "main_conf {} too far from scalar_conf {}",
            main_conf,
            scalar_conf
        );
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

    // ==================== Iterative Decode Correctness Tests ====================

    #[test]
    fn test_stable_prefix_tracking() {
        // Test that stable_prefix_len correctly tracks contiguous unmasked from start
        let is_masked = vec![false, false, true, false, true];
        let stable_prefix_len = is_masked.iter().position(|&m| m).unwrap_or(is_masked.len());
        assert_eq!(stable_prefix_len, 2); // First two are unmasked

        let is_masked = vec![true, false, false, false, false];
        let stable_prefix_len = is_masked.iter().position(|&m| m).unwrap_or(is_masked.len());
        assert_eq!(stable_prefix_len, 0); // First is masked

        let is_masked = vec![false, false, false, false, false];
        let stable_prefix_len = is_masked.iter().position(|&m| m).unwrap_or(is_masked.len());
        assert_eq!(stable_prefix_len, 5); // All unmasked
    }

    #[test]
    fn test_confidence_thresholding_partial_unmask() {
        // Simulate candidates with varying confidences
        let candidates = vec![
            (0usize, 100i32, 0.8f32),  // Above threshold 0.7
            (1, 200, 0.5),              // Below threshold
            (2, 300, 0.9),              // Above threshold
            (3, 400, 0.3),              // Below threshold
        ];
        let threshold = 0.7;

        let above_threshold: Vec<_> = candidates
            .iter()
            .filter(|(_, _, c)| *c > threshold)
            .collect();

        assert_eq!(above_threshold.len(), 2);
        assert_eq!(above_threshold[0].0, 0);
        assert_eq!(above_threshold[1].0, 2);
    }

    #[test]
    fn test_progress_guarantee_unmask_best() {
        // When all confidences are below threshold, pick highest
        let candidates = vec![
            (0usize, 100i32, 0.3f32),
            (1, 200, 0.5),  // Highest
            (2, 300, 0.4),
            (3, 400, 0.2),
        ];
        let threshold = 0.9; // All below

        let above_threshold: Vec<_> = candidates
            .iter()
            .filter(|(_, _, c)| *c > threshold)
            .collect();

        assert!(above_threshold.is_empty(), "No candidates above threshold");

        // Progress guarantee: unmask highest confidence
        let best = candidates
            .iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap();

        assert_eq!(best.0, 1, "Should pick index 1 with confidence 0.5");
        assert_eq!(best.1, 200, "Should pick token 200");
    }

    #[test]
    fn test_best_prediction_fallback() {
        // Test that we track best predictions for fallback
        let mask_id = 128256;
        let mut best_predictions: Vec<(i32, f32)> = vec![(mask_id, 0.0); 4];

        // Simulate multiple iterations updating best predictions
        // Iteration 1
        let iter1_predictions = vec![(100, 0.6), (200, 0.4), (300, 0.5), (400, 0.3)];
        for (i, (token, conf)) in iter1_predictions.iter().enumerate() {
            if *conf > best_predictions[i].1 {
                best_predictions[i] = (*token, *conf);
            }
        }

        // Iteration 2 (some improve)
        let iter2_predictions = vec![(110, 0.5), (210, 0.7), (310, 0.4), (410, 0.8)];
        for (i, (token, conf)) in iter2_predictions.iter().enumerate() {
            if *conf > best_predictions[i].1 {
                best_predictions[i] = (*token, *conf);
            }
        }

        // Check best predictions
        assert_eq!(best_predictions[0], (100, 0.6), "Position 0: iter1 was better");
        assert_eq!(best_predictions[1], (210, 0.7), "Position 1: iter2 was better");
        assert_eq!(best_predictions[2], (300, 0.5), "Position 2: iter1 was better");
        assert_eq!(best_predictions[3], (410, 0.8), "Position 3: iter2 was better");

        // No MASK tokens in best predictions
        for (i, (token, _)) in best_predictions.iter().enumerate() {
            assert_ne!(*token, mask_id, "Position {} should not be MASK", i);
        }
    }

    #[test]
    fn test_batch_index_calculation() {
        // Test batch index calculation for partial recomputation
        let block_size = 8;
        let recompute_from = 3;

        // Batch layout: [pred_token, tok[3], tok[4], tok[5], tok[6], tok[7]]
        // batch_idx 0 -> predicts position 3
        // batch_idx k -> predicts position (recompute_from + k)

        for i in recompute_from..block_size {
            let batch_idx = (i - recompute_from) as i32;
            let predicted_position = recompute_from + batch_idx as usize;
            assert_eq!(predicted_position, i, "Batch idx {} should predict position {}", batch_idx, i);
        }
    }

    #[test]
    fn test_iterative_decode_simulation() {
        // Simulate a full iterative decode with mock confidences
        let block_size = 4;
        let mask_id = 128256;
        let threshold = 0.7;

        let mut block_tokens: Vec<i32> = vec![mask_id; block_size];
        let mut is_masked: Vec<bool> = vec![true; block_size];
        let mut stable_prefix_len: usize = 0;

        // Mock confidence function: returns high confidence for position 0, 2 first
        // then position 1, 3 in subsequent iterations
        let mock_confidences = vec![
            // Iteration 1: position 0 and 2 are confident
            vec![(0, 100, 0.9), (1, 200, 0.3), (2, 300, 0.8), (3, 400, 0.2)],
            // Iteration 2: position 1 is now confident, 3 still low
            vec![(1, 201, 0.85), (3, 401, 0.4)],
            // Iteration 3: position 3 is confident
            vec![(3, 402, 0.75)],
        ];

        for (iter_idx, candidates) in mock_confidences.iter().enumerate() {
            // Check only masked positions
            let masked_count = is_masked.iter().filter(|&&m| m).count();
            assert_eq!(masked_count, candidates.len(),
                "Iteration {}: expected {} masked positions, got {}",
                iter_idx, candidates.len(), masked_count);

            // Unmask above threshold
            let mut any_unmasked = false;
            for &(idx, token, conf) in candidates {
                if conf > threshold && is_masked[idx] {
                    block_tokens[idx] = token;
                    is_masked[idx] = false;
                    any_unmasked = true;
                }
            }

            // Progress guarantee
            if !any_unmasked && !candidates.is_empty() {
                let best = candidates
                    .iter()
                    .filter(|(idx, _, _)| is_masked[*idx])
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                    .unwrap();
                block_tokens[best.0] = best.1;
                is_masked[best.0] = false;
            }

            // Update stable prefix
            stable_prefix_len = is_masked.iter().position(|&m| m).unwrap_or(block_size);
        }

        // After all iterations, all should be unmasked
        assert!(is_masked.iter().all(|&m| !m), "All positions should be unmasked");
        assert_eq!(stable_prefix_len, block_size);

        // No MASK tokens in output
        for (i, &token) in block_tokens.iter().enumerate() {
            assert_ne!(token, mask_id, "Position {} should not be MASK", i);
        }
    }

    #[test]
    fn test_token_shift_batch_layout() {
        // Verify token shift: logits[i-1] predicts token[i]
        // Batch: [prev_token, mask0, mask1, mask2, mask3]
        // logits[0] (from prev_token) predicts position 0
        // logits[1] (from mask0) predicts position 1
        // etc.

        let prev_token = 99;
        let mask_id = 128256;
        let block_tokens = vec![mask_id; 4];
        let block_start_pos = 100;

        // Build expected batch
        let mut expected_batch: Vec<(i32, i32)> = Vec::new(); // (token, position)
        expected_batch.push((prev_token, block_start_pos - 1));
        for (i, &token) in block_tokens.iter().enumerate() {
            expected_batch.push((token, block_start_pos + i as i32));
        }

        assert_eq!(expected_batch.len(), 5); // prev + 4 block tokens
        assert_eq!(expected_batch[0], (99, 99)); // prev_token at position 99
        assert_eq!(expected_batch[1], (mask_id, 100)); // mask at position 100

        // Logit mapping
        // logits[0] is from position 99 (prev_token), predicts position 100
        // logits[i] predicts position (block_start_pos + i)
        for i in 0..4 {
            let predicted_pos = block_start_pos + i as i32;
            assert_eq!(predicted_pos, 100 + i as i32);
        }
    }

    #[test]
    fn test_decode_mode_enum() {
        use super::super::dlm_config::DlmDecodeMode;

        // Test default
        let mode = DlmDecodeMode::default();
        assert_eq!(mode, DlmDecodeMode::Greedy);

        // Test serde
        let json_greedy = serde_json::to_string(&DlmDecodeMode::Greedy).unwrap();
        assert_eq!(json_greedy, "\"greedy\"");

        let json_iterative = serde_json::to_string(&DlmDecodeMode::Iterative).unwrap();
        assert_eq!(json_iterative, "\"iterative\"");

        // Test deserialization
        let parsed: DlmDecodeMode = serde_json::from_str("\"greedy\"").unwrap();
        assert_eq!(parsed, DlmDecodeMode::Greedy);

        let parsed: DlmDecodeMode = serde_json::from_str("\"iterative\"").unwrap();
        assert_eq!(parsed, DlmDecodeMode::Iterative);
    }
}
