//! BitNet transformer model for inference.
//!
//! Implements the forward pass using native ternary SIMD kernels.

use std::path::Path;
use std::time::Instant;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::gguf::{GgufReader, GgmlQuantType, GgufError, repack_ternary_weights, NativeWeightFormat};

/// Global profiling state
static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);
static PROFILE_QKV_US: AtomicU64 = AtomicU64::new(0);
static PROFILE_ROPE_US: AtomicU64 = AtomicU64::new(0);
static PROFILE_ATTN_SCORES_US: AtomicU64 = AtomicU64::new(0);
static PROFILE_O_PROJ_US: AtomicU64 = AtomicU64::new(0);
static PROFILE_FFN_US: AtomicU64 = AtomicU64::new(0);
static PROFILE_NORM_US: AtomicU64 = AtomicU64::new(0);
static PROFILE_OUTPUT_US: AtomicU64 = AtomicU64::new(0);

/// Enable profiling
pub fn enable_profiling() {
    PROFILING_ENABLED.store(true, Ordering::SeqCst);
}

/// Print profiling results
pub fn print_profile_results() {
    let qkv = PROFILE_QKV_US.load(Ordering::SeqCst);
    let rope = PROFILE_ROPE_US.load(Ordering::SeqCst);
    let attn = PROFILE_ATTN_SCORES_US.load(Ordering::SeqCst);
    let o_proj = PROFILE_O_PROJ_US.load(Ordering::SeqCst);
    let ffn = PROFILE_FFN_US.load(Ordering::SeqCst);
    let norm = PROFILE_NORM_US.load(Ordering::SeqCst);
    let output = PROFILE_OUTPUT_US.load(Ordering::SeqCst);
    let total = qkv + rope + attn + o_proj + ffn + norm + output;

    println!("\n=== Detailed Profile ===");
    println!("Q/K/V projection: {:>8.2} ms ({:>5.1}%)", qkv as f64 / 1000.0, 100.0 * qkv as f64 / total as f64);
    println!("RoPE encoding:    {:>8.2} ms ({:>5.1}%)", rope as f64 / 1000.0, 100.0 * rope as f64 / total as f64);
    println!("Attention scores: {:>8.2} ms ({:>5.1}%)", attn as f64 / 1000.0, 100.0 * attn as f64 / total as f64);
    println!("O projection:     {:>8.2} ms ({:>5.1}%)", o_proj as f64 / 1000.0, 100.0 * o_proj as f64 / total as f64);
    println!("FFN:              {:>8.2} ms ({:>5.1}%)", ffn as f64 / 1000.0, 100.0 * ffn as f64 / total as f64);
    println!("Norms:            {:>8.2} ms ({:>5.1}%)", norm as f64 / 1000.0, 100.0 * norm as f64 / total as f64);
    println!("Output proj:      {:>8.2} ms ({:>5.1}%)", output as f64 / 1000.0, 100.0 * output as f64 / total as f64);
    println!("Total tracked:    {:>8.2} ms", total as f64 / 1000.0);
    println!("========================\n");
}

/// Reset profiling counters
pub fn reset_profile() {
    PROFILE_QKV_US.store(0, Ordering::SeqCst);
    PROFILE_ROPE_US.store(0, Ordering::SeqCst);
    PROFILE_ATTN_SCORES_US.store(0, Ordering::SeqCst);
    PROFILE_O_PROJ_US.store(0, Ordering::SeqCst);
    PROFILE_FFN_US.store(0, Ordering::SeqCst);
    PROFILE_NORM_US.store(0, Ordering::SeqCst);
    PROFILE_OUTPUT_US.store(0, Ordering::SeqCst);
}
use crate::kernels::BitNetKernel;
use crate::kernels::simd::{rms_norm_with_scale, silu_inplace, softmax_inplace};
use rayon::prelude::*;

use super::kv_cache::{KVCache, KVCacheConfig};
use super::sampling::{SamplingConfig, sample_token};

/// BitNet model configuration
#[derive(Debug, Clone)]
pub struct BitNetConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Intermediate (FFN) dimension
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta (frequency base)
    pub rope_theta: f32,
    /// Head dimension (hidden_size / num_heads)
    pub head_dim: usize,
}

impl Default for BitNetConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 8,
            intermediate_size: 8192,
            max_seq_len: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: 128,
        }
    }
}

/// Weights for a single transformer layer
pub struct LayerWeights {
    // Attention
    pub attn_norm: Vec<f32>,
    pub q_proj: NativeWeightFormat,
    pub k_proj: NativeWeightFormat,
    pub v_proj: NativeWeightFormat,
    pub o_proj: NativeWeightFormat,

    // FFN
    pub ffn_norm: Vec<f32>,
    pub gate_proj: NativeWeightFormat,
    pub up_proj: NativeWeightFormat,
    pub down_proj: NativeWeightFormat,

    // LRC corrections (optional)
    pub q_lrc_u: Option<Vec<f32>>,
    pub q_lrc_v: Option<Vec<f32>>,
    // ... other LRC matrices if needed
}

/// BitNet inference engine
pub struct BitNetEngine {
    /// Model configuration
    pub config: BitNetConfig,
    /// Token embeddings [vocab_size, hidden_size] - kept in F32 for lookup
    pub embed_tokens: Vec<f32>,
    /// Token embeddings in BF16 for output projection (half memory bandwidth)
    embed_tokens_bf16: Vec<u16>,
    /// Output norm (final RMS norm)
    pub output_norm: Vec<f32>,
    /// Output projection (lm_head) - may be tied with embed_tokens
    pub output_proj: Option<Vec<f32>>,
    /// Output projection in BF16 (if not tied with embeddings)
    output_proj_bf16: Option<Vec<u16>>,
    /// Per-layer weights
    pub layers: Vec<LayerWeights>,
    /// KV cache
    pub kv_cache: KVCache,
    /// Tokenizer vocabulary (for decoding)
    pub vocab: Option<Vec<String>>,
    /// RoPE frequency cache
    rope_freqs: Vec<f32>,
    /// Native BitNet SIMD kernel
    kernel: BitNetKernel,
}

/// Error type for engine operations
#[derive(Debug)]
pub enum EngineError {
    /// GGUF parsing error
    Gguf(GgufError),
    /// Missing required tensor
    MissingTensor(String),
    /// Invalid model configuration
    InvalidConfig(String),
    /// Generation error
    GenerationError(String),
}

impl From<GgufError> for EngineError {
    fn from(e: GgufError) -> Self {
        Self::Gguf(e)
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf(e) => write!(f, "GGUF error: {}", e),
            Self::MissingTensor(name) => write!(f, "Missing tensor: {}", name),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::GenerationError(msg) => write!(f, "Generation error: {}", msg),
        }
    }
}

impl std::error::Error for EngineError {}

impl BitNetEngine {
    /// Load a model from a GGUF file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, EngineError> {
        Self::load_with_context_len(path, None)
    }

    /// Load a model from a GGUF file with optional context length override.
    ///
    /// Use `context_len` to limit KV cache memory for large models with
    /// very long default context lengths (e.g., 128K).
    pub fn load_with_context_len<P: AsRef<Path>>(
        path: P,
        context_len: Option<usize>,
    ) -> Result<Self, EngineError> {
        let reader = GgufReader::open(path)?;
        reader.print_summary();

        // Load embeddings first to infer vocab_size if needed
        let embed_tokens = load_f32_tensor(&reader, "token_embd.weight")?;

        // Extract configuration
        let gguf_config = &reader.config;
        let hidden_size = gguf_config.hidden_size as usize;

        // Infer vocab_size from embedding tensor shape if not in metadata
        // embed_tokens shape is [hidden_size, vocab_size]
        let vocab_size = if gguf_config.vocab_size > 0 {
            gguf_config.vocab_size as usize
        } else {
            embed_tokens.len() / hidden_size
        };
        tracing::info!("Vocab size: {} (inferred from embedding shape)", vocab_size);

        // Use context_len override if provided, otherwise use model's default
        let max_seq_len = context_len.unwrap_or(gguf_config.max_seq_len as usize);
        if context_len.is_some() {
            tracing::info!(
                "Context length: {} (overridden from model default {})",
                max_seq_len,
                gguf_config.max_seq_len
            );
        } else {
            tracing::info!("Context length: {} (from model)", max_seq_len);
        }

        let config = BitNetConfig {
            vocab_size,
            hidden_size,
            num_layers: gguf_config.num_layers as usize,
            num_heads: gguf_config.num_heads as usize,
            num_kv_heads: gguf_config.num_kv_heads as usize,
            intermediate_size: gguf_config.intermediate_size as usize,
            max_seq_len,
            rms_norm_eps: gguf_config.rms_norm_eps,
            rope_theta: gguf_config.rope_theta,
            head_dim: gguf_config.hidden_size as usize / gguf_config.num_heads as usize,
        };

        // Validate config
        if config.hidden_size == 0 || config.num_layers == 0 || config.vocab_size == 0 {
            return Err(EngineError::InvalidConfig(
                "Missing required model dimensions".to_string(),
            ));
        }

        // Load output norm
        let output_norm = load_f32_tensor(&reader, "output_norm.weight")?;

        // Try to load output projection (may be tied with embeddings)
        let output_proj = load_f32_tensor(&reader, "output.weight").ok();

        // Convert embeddings to BF16 for efficient output projection
        // This halves memory bandwidth for the vocab_size × hidden_size dot products
        tracing::info!("Converting embeddings to BF16 for output projection...");
        let embed_tokens_bf16 = f32_slice_to_bf16(&embed_tokens);

        // Convert output projection to BF16 if it exists (not tied with embeddings)
        let output_proj_bf16 = output_proj.as_ref().map(|proj| {
            tracing::info!("Converting output projection to BF16...");
            f32_slice_to_bf16(proj)
        });

        // Load layer weights
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = load_layer_weights(&reader, layer_idx, &config)?;
            layers.push(layer);
        }

        // Initialize KV cache
        let kv_config = KVCacheConfig {
            num_layers: config.num_layers,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            max_seq_len: config.max_seq_len,
        };
        let kv_cache = KVCache::new(kv_config);

        // Log KV cache memory usage
        let kv_memory_mb = kv_cache.memory_bytes() / (1024 * 1024);
        tracing::info!("KV cache allocated: {} MB", kv_memory_mb);

        // Load vocabulary
        let vocab = reader.get_vocab();

        // Precompute RoPE frequencies
        let rope_freqs = compute_rope_freqs(config.head_dim, config.rope_theta);

        // Initialize native BitNet kernel with auto-tuned tiles
        let kernel = BitNetKernel::with_auto_tune(
            config.hidden_size,
            config.hidden_size,
        );
        tracing::info!(
            "Native kernel initialized: {} (tiles: {}x{})",
            kernel.capabilities.description(),
            kernel.tile_config.bm,
            kernel.tile_config.bk,
        );

        Ok(Self {
            config,
            embed_tokens,
            embed_tokens_bf16,
            output_norm,
            output_proj,
            output_proj_bf16,
            layers,
            kv_cache,
            vocab,
            rope_freqs,
            kernel,
        })
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get hidden size.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Reset KV cache for new sequence.
    pub fn reset(&mut self) {
        self.kv_cache.clear();
    }

    /// Forward pass for a single position (decode step).
    ///
    /// # Arguments
    /// * `token_id` - Input token ID
    /// * `pos` - Position in sequence
    ///
    /// # Returns
    /// Logits [vocab_size]
    pub fn forward_one(&mut self, token_id: i32, pos: usize) -> Vec<f32> {
        // Get embedding
        let hidden = self.get_embedding(token_id as usize);

        // Forward through layers
        let hidden = self.forward_layers(hidden, pos, 1);

        // Apply output norm and projection
        self.forward_output(&hidden)
    }

    /// Forward pass for multiple positions (prefill).
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs
    /// * `start_pos` - Starting position
    ///
    /// # Returns
    /// Logits for last position [vocab_size]
    pub fn forward_prefill(&mut self, token_ids: &[i32], start_pos: usize) -> Vec<f32> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return vec![0.0; self.config.vocab_size];
        }

        // Get embeddings for all tokens
        let mut hidden = Vec::with_capacity(seq_len * self.config.hidden_size);
        for &token_id in token_ids {
            hidden.extend_from_slice(&self.get_embedding(token_id as usize));
        }

        // Forward through layers
        let hidden = self.forward_layers(hidden, start_pos, seq_len);

        // Get logits for last position only
        let last_hidden = &hidden[(seq_len - 1) * self.config.hidden_size..];
        self.forward_output(last_hidden)
    }

    /// Get token embedding.
    fn get_embedding(&self, token_id: usize) -> Vec<f32> {
        let offset = token_id * self.config.hidden_size;
        self.embed_tokens[offset..offset + self.config.hidden_size].to_vec()
    }

    /// Forward through all transformer layers.
    fn forward_layers(&mut self, mut hidden: Vec<f32>, start_pos: usize, seq_len: usize) -> Vec<f32> {
        for layer_idx in 0..self.config.num_layers {
            hidden = self.forward_layer(layer_idx, hidden, start_pos, seq_len);
        }
        hidden
    }

    /// Forward through a single transformer layer.
    fn forward_layer(
        &mut self,
        layer_idx: usize,
        hidden: Vec<f32>,
        start_pos: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        // Profile: pre-attention norm
        let norm_start = Instant::now();

        // Pre-attention norm - clone the norm weights to avoid borrow conflict
        let attn_norm = self.layers[layer_idx].attn_norm.clone();
        let normed = self.apply_norm(&hidden, &attn_norm, seq_len);

        if PROFILING_ENABLED.load(Ordering::Relaxed) {
            PROFILE_NORM_US.fetch_add(norm_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        // Attention (mutates kv_cache)
        let attn_out = self.attention(layer_idx, &normed, start_pos, seq_len);

        // Profile: pre-FFN norm
        let norm_start = Instant::now();

        // Residual connection
        let mut hidden: Vec<f32> = hidden.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // Pre-FFN norm - clone the norm weights to avoid borrow conflict
        let ffn_norm = self.layers[layer_idx].ffn_norm.clone();
        let normed = self.apply_norm(&hidden, &ffn_norm, seq_len);

        if PROFILING_ENABLED.load(Ordering::Relaxed) {
            PROFILE_NORM_US.fetch_add(norm_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        // FFN (immutable access to self)
        let ffn_out = self.ffn(layer_idx, &normed, seq_len);

        // Residual connection
        for (h, f) in hidden.iter_mut().zip(ffn_out.iter()) {
            *h += f;
        }

        hidden
    }

    /// Apply RMS normalization.
    fn apply_norm(&self, hidden: &[f32], gamma: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.config.hidden_size;
        let mut output = Vec::with_capacity(seq_len * h);

        for i in 0..seq_len {
            let start = i * h;
            let slice = &hidden[start..start + h];
            let normed = rms_norm_with_scale(slice, gamma, self.config.rms_norm_eps);
            output.extend_from_slice(&normed);
        }

        output
    }

    /// Compute attention for a layer.
    fn attention(
        &mut self,
        layer_idx: usize,
        hidden: &[f32],
        start_pos: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Profile: Q/K/V projections
        let qkv_start = Instant::now();

        // Quantize input ONCE for all Q/K/V projections (3x speedup on quantization)
        let (quantized_hidden, hidden_scale) = self.kernel.quantize_activations(hidden);

        // Project Q, K, V using pre-quantized activations
        let q = self.linear_forward_quantized(
            &self.layers[layer_idx].q_proj, &quantized_hidden, hidden_scale, seq_len
        );
        let k = self.linear_forward_quantized(
            &self.layers[layer_idx].k_proj, &quantized_hidden, hidden_scale, seq_len
        );
        let v = self.linear_forward_quantized(
            &self.layers[layer_idx].v_proj, &quantized_hidden, hidden_scale, seq_len
        );

        if PROFILING_ENABLED.load(Ordering::Relaxed) {
            PROFILE_QKV_US.fetch_add(qkv_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        // Profile: RoPE
        let rope_start = Instant::now();

        // Apply RoPE to Q and K
        let q = self.apply_rope(&q, start_pos, seq_len, num_heads, head_dim);
        let k = self.apply_rope(&k, start_pos, seq_len, num_kv_heads, head_dim);

        if PROFILING_ENABLED.load(Ordering::Relaxed) {
            PROFILE_ROPE_US.fetch_add(rope_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        // Update KV cache
        for i in 0..seq_len {
            let pos = start_pos + i;
            let k_slice = &k[i * kv_dim..(i + 1) * kv_dim];
            let v_slice = &v[i * kv_dim..(i + 1) * kv_dim];
            self.kv_cache.update(layer_idx, pos, k_slice, v_slice);
        }

        // Profile: attention scores
        let attn_start = Instant::now();

        // Compute attention scores and output
        let total_seq_len = start_pos + seq_len;
        let attn_out = self.compute_attention_scores(
            &q, layer_idx, seq_len, total_seq_len, num_heads, num_kv_heads, head_dim,
        );

        if PROFILING_ENABLED.load(Ordering::Relaxed) {
            PROFILE_ATTN_SCORES_US.fetch_add(attn_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        // Profile: output projection
        let o_proj_start = Instant::now();

        // Output projection
        let result = self.linear_forward_ref(&self.layers[layer_idx].o_proj, &attn_out, seq_len);

        if PROFILING_ENABLED.load(Ordering::Relaxed) {
            PROFILE_O_PROJ_US.fetch_add(o_proj_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        result
    }

    /// Compute attention scores and weighted values.
    fn compute_attention_scores(
        &self,
        q: &[f32],
        layer_idx: usize,
        q_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let h = self.config.hidden_size;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let heads_per_kv = num_heads / num_kv_heads;

        let layer_cache = self.kv_cache.layer(layer_idx);
        let all_keys = layer_cache.get_keys(kv_len);
        let all_values = layer_cache.get_values(kv_len);

        let mut output = vec![0.0; q_len * h];

        // For each query position
        for qi in 0..q_len {
            // For each head
            for head in 0..num_heads {
                let kv_head = head / heads_per_kv;

                // Get query for this head
                let q_start = qi * h + head * head_dim;
                let q_vec = &q[q_start..q_start + head_dim];

                // Compute attention scores with all keys
                let mut scores = vec![0.0; kv_len];
                for ki in 0..kv_len {
                    let k_start = ki * num_kv_heads * head_dim + kv_head * head_dim;
                    let k_vec = &all_keys[k_start..k_start + head_dim];
                    scores[ki] = dot(q_vec, k_vec) * scale;
                }

                // Apply causal mask (for positions after current)
                let current_pos = qi; // relative to start of this forward pass
                for ki in (current_pos + 1)..kv_len {
                    scores[ki] = f32::NEG_INFINITY;
                }

                // Softmax
                softmax_inplace(&mut scores);

                // Weighted sum of values
                let out_start = qi * h + head * head_dim;
                for ki in 0..kv_len {
                    let v_start = ki * num_kv_heads * head_dim + kv_head * head_dim;
                    let v_vec = &all_values[v_start..v_start + head_dim];
                    for (j, &v) in v_vec.iter().enumerate() {
                        output[out_start + j] += scores[ki] * v;
                    }
                }
            }
        }

        output
    }

    /// FFN forward pass.
    fn ffn(&self, layer_idx: usize, hidden: &[f32], seq_len: usize) -> Vec<f32> {
        // Profile: entire FFN
        let ffn_start = Instant::now();

        // Quantize input ONCE for gate and up projections (2x speedup on quantization)
        let (quantized_hidden, hidden_scale) = self.kernel.quantize_activations(hidden);

        // Gate and up projections using pre-quantized activations
        let gate = self.linear_forward_quantized(
            &self.layers[layer_idx].gate_proj, &quantized_hidden, hidden_scale, seq_len
        );
        let up = self.linear_forward_quantized(
            &self.layers[layer_idx].up_proj, &quantized_hidden, hidden_scale, seq_len
        );

        // Apply SiLU to gate and multiply with up
        let mut gate = gate;
        silu_inplace(&mut gate);
        let intermediate: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| g * u).collect();

        // Down projection (new input, must quantize fresh)
        let result = self.linear_forward_ref(&self.layers[layer_idx].down_proj, &intermediate, seq_len);

        if PROFILING_ENABLED.load(Ordering::Relaxed) {
            PROFILE_FFN_US.fetch_add(ffn_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        result
    }

    /// Apply linear projection using native BitNet kernels.
    ///
    /// This is the core operation that uses SIMD-optimized ternary GEMM:
    /// output = weights * quantize(input) * scale
    fn linear_forward_ref(
        &self,
        weights: &NativeWeightFormat,
        input: &[f32],
        seq_len: usize,
    ) -> Vec<f32> {
        let in_features = weights.in_features;
        let out_features = weights.out_features;

        // Validate dimensions
        debug_assert_eq!(
            input.len(),
            seq_len * in_features,
            "Input size mismatch: expected {} ({}x{}), got {}",
            seq_len * in_features,
            seq_len,
            in_features,
            input.len()
        );

        // Quantize activations to INT8
        // The kernel handles finding the optimal scale factor
        let (quantized_input, input_scale) = self.kernel.quantize_activations(input);

        // Compute GEMM: output = packed_weights @ quantized_input
        // Dimensions: [out_features x in_features] @ [in_features x seq_len] = [out_features x seq_len]
        //
        // The native kernel expects:
        // - M = out_features
        // - N = seq_len (batch size)
        // - K = in_features
        // - weights: [M x K/4] packed ternary
        // - activations: [K x N] INT8
        // - output: [M x N] FP32
        self.kernel.gemm(
            out_features,
            seq_len,
            in_features,
            &weights.data,
            &quantized_input,
            weights.scale * input_scale,
        )
    }

    /// Apply linear projection with pre-quantized activations.
    ///
    /// Use this when the same input is used for multiple projections
    /// (e.g., Q/K/V in attention, gate/up in FFN) to avoid redundant quantization.
    #[inline]
    fn linear_forward_quantized(
        &self,
        weights: &NativeWeightFormat,
        quantized_input: &[i8],
        input_scale: f32,
        seq_len: usize,
    ) -> Vec<f32> {
        let in_features = weights.in_features;
        let out_features = weights.out_features;

        debug_assert_eq!(
            quantized_input.len(),
            seq_len * in_features,
            "Quantized input size mismatch"
        );

        self.kernel.gemm(
            out_features,
            seq_len,
            in_features,
            &weights.data,
            quantized_input,
            weights.scale * input_scale,
        )
    }

    /// Apply RoPE positional encoding.
    fn apply_rope(
        &self,
        x: &[f32],
        start_pos: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = x.to_vec();
        let half_dim = head_dim / 2;

        for i in 0..seq_len {
            let pos = start_pos + i;
            for head in 0..num_heads {
                let offset = i * num_heads * head_dim + head * head_dim;

                for j in 0..half_dim {
                    let freq = self.rope_freqs[j];
                    let angle = pos as f32 * freq;
                    let cos = angle.cos();
                    let sin = angle.sin();

                    let x0 = output[offset + j];
                    let x1 = output[offset + j + half_dim];

                    output[offset + j] = x0 * cos - x1 * sin;
                    output[offset + j + half_dim] = x0 * sin + x1 * cos;
                }
            }
        }

        output
    }

    /// Forward through output norm and projection.
    /// Uses BF16 weights to halve memory bandwidth (major bottleneck).
    fn forward_output(&self, hidden: &[f32]) -> Vec<f32> {
        // Profile: output projection
        let output_start = Instant::now();

        // Apply final norm
        let normed = rms_norm_with_scale(hidden, &self.output_norm, self.config.rms_norm_eps);

        // Project to vocabulary using parallel computation with BF16 weights
        // BF16 halves memory bandwidth: 128K vocab × 2560 hidden × 2 bytes = 655MB
        // vs F32: 128K vocab × 2560 hidden × 4 bytes = 1.3GB
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        let logits = if let Some(ref output_proj_bf16) = self.output_proj_bf16 {
            // Use BF16 output projection weights (parallel)
            (0..vocab_size)
                .into_par_iter()
                .map(|i| {
                    let offset = i * hidden_size;
                    dot_bf16_simd(&normed, &output_proj_bf16[offset..offset + hidden_size])
                })
                .collect()
        } else {
            // Tied embeddings - use BF16 version (parallel)
            (0..vocab_size)
                .into_par_iter()
                .map(|i| {
                    let offset = i * hidden_size;
                    dot_bf16_simd(&normed, &self.embed_tokens_bf16[offset..offset + hidden_size])
                })
                .collect()
        };

        if PROFILING_ENABLED.load(Ordering::Relaxed) {
            PROFILE_OUTPUT_US.fetch_add(output_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        logits
    }

    /// Generate tokens.
    pub fn generate(
        &mut self,
        input_ids: &[i32],
        max_tokens: usize,
        sampling_config: &SamplingConfig,
    ) -> Vec<i32> {
        let mut rng = rand::thread_rng();
        let mut output_ids = Vec::with_capacity(max_tokens);

        // Prefill
        let logits = self.forward_prefill(input_ids, 0);
        let token_id = sample_token(&logits, sampling_config, &mut rng);
        output_ids.push(token_id as i32);

        // Decode
        let mut pos = input_ids.len();
        for _ in 1..max_tokens {
            let logits = self.forward_one(output_ids.last().copied().unwrap_or(0), pos);
            let token_id = sample_token(&logits, sampling_config, &mut rng);

            // Check for EOS (simplified)
            if token_id == 2 {
                break;
            }

            output_ids.push(token_id as i32);
            pos += 1;
        }

        output_ids
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Load an F32 tensor from GGUF.
fn load_f32_tensor(reader: &GgufReader, name: &str) -> Result<Vec<f32>, EngineError> {
    let tensor = reader
        .find_tensor(name)
        .ok_or_else(|| EngineError::MissingTensor(name.to_string()))?;

    let data = reader.tensor_data(tensor);

    match tensor.dtype {
        GgmlQuantType::F32 => {
            // Direct F32 data
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(floats)
        }
        GgmlQuantType::F16 => {
            // Convert F16 to F32
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            Ok(floats)
        }
        GgmlQuantType::BF16 => {
            // Convert BF16 to F32
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            Ok(floats)
        }
        _ => Err(EngineError::InvalidConfig(format!(
            "Unsupported dtype {:?} for tensor {}",
            tensor.dtype, name
        ))),
    }
}

/// Load quantized weights and repack to native format.
fn load_quantized_weights(
    reader: &GgufReader,
    name: &str,
) -> Result<NativeWeightFormat, EngineError> {
    let tensor = reader
        .find_tensor(name)
        .ok_or_else(|| EngineError::MissingTensor(name.to_string()))?;

    let data = reader.tensor_data(tensor);

    if !tensor.dtype.is_ternary() {
        return Err(EngineError::InvalidConfig(format!(
            "Expected ternary weights for {}, got {:?}",
            name, tensor.dtype
        )));
    }

    repack_ternary_weights(data, tensor.dtype, &tensor.shape).map_err(|e| e.into())
}

/// Load layer weights.
fn load_layer_weights(
    reader: &GgufReader,
    layer_idx: usize,
    config: &BitNetConfig,
) -> Result<LayerWeights, EngineError> {
    let prefix = format!("blk.{}", layer_idx);

    // Load attention norm
    let attn_norm = load_f32_tensor(reader, &format!("{}.attn_norm.weight", prefix))?;

    // Load attention projections
    let q_proj = load_quantized_weights_or_f32(reader, &format!("{}.attn_q.weight", prefix), config)?;
    let k_proj = load_quantized_weights_or_f32(reader, &format!("{}.attn_k.weight", prefix), config)?;
    let v_proj = load_quantized_weights_or_f32(reader, &format!("{}.attn_v.weight", prefix), config)?;
    let o_proj = load_quantized_weights_or_f32(reader, &format!("{}.attn_output.weight", prefix), config)?;

    // Load FFN norm
    let ffn_norm = load_f32_tensor(reader, &format!("{}.ffn_norm.weight", prefix))?;

    // Load FFN projections
    let gate_proj = load_quantized_weights_or_f32(reader, &format!("{}.ffn_gate.weight", prefix), config)?;
    let up_proj = load_quantized_weights_or_f32(reader, &format!("{}.ffn_up.weight", prefix), config)?;
    let down_proj = load_quantized_weights_or_f32(reader, &format!("{}.ffn_down.weight", prefix), config)?;

    Ok(LayerWeights {
        attn_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        ffn_norm,
        gate_proj,
        up_proj,
        down_proj,
        q_lrc_u: None,
        q_lrc_v: None,
    })
}

/// Load weights as either quantized or F32 (fallback).
fn load_quantized_weights_or_f32(
    reader: &GgufReader,
    name: &str,
    config: &BitNetConfig,
) -> Result<NativeWeightFormat, EngineError> {
    // Try quantized first
    if let Ok(weights) = load_quantized_weights(reader, name) {
        return Ok(weights);
    }

    // Fallback to F32 (convert to pseudo-quantized format)
    let tensor = reader
        .find_tensor(name)
        .ok_or_else(|| EngineError::MissingTensor(name.to_string()))?;

    let _f32_data = load_f32_tensor(reader, name)?;

    // Create dummy native format (weights stored as F32, scale = 1)
    // This is inefficient but allows F32 models to work
    // GGUF stores shapes in column-major order:
    // shape[0] = columns (in_features), shape[1] = rows (out_features)
    Ok(NativeWeightFormat {
        data: vec![0; tensor.n_elements / 4], // Placeholder
        scale: 1.0,
        in_features: tensor.shape.first().copied().unwrap_or(config.hidden_size),
        out_features: tensor.shape.get(1).copied().unwrap_or(0),
    })
}

/// Compute RoPE frequencies.
fn compute_rope_freqs(head_dim: usize, theta: f32) -> Vec<f32> {
    let half_dim = head_dim / 2;
    (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect()
}

/// Dot product (scalar fallback).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ============================================================================
// BF16 Support for Output Projection
// ============================================================================

/// Convert F32 to BF16 (just truncate mantissa)
#[inline]
fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Convert BF16 to F32 (just shift up)
#[inline]
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// Convert F32 slice to BF16 Vec
fn f32_slice_to_bf16(input: &[f32]) -> Vec<u16> {
    input.iter().map(|&x| f32_to_bf16(x)).collect()
}

/// Dot product with BF16 weights and F32 activations.
/// Weights are converted on-the-fly (saves memory bandwidth).
fn dot_bf16(a: &[f32], b_bf16: &[u16]) -> f32 {
    a.iter()
        .zip(b_bf16.iter())
        .map(|(&x, &y)| x * bf16_to_f32(y))
        .sum()
}

/// SIMD-optimized dot product for F32 activations × BF16 weights.
/// Uses NEON on ARM64 for efficient BF16→F32 conversion.
#[cfg(target_arch = "aarch64")]
fn dot_bf16_simd(a: &[f32], b_bf16: &[u16]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, b_bf16.len());

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0;

        // Process 16 elements at a time (4x unroll)
        while i + 16 <= n {
            // Load 4 BF16 values, convert to F32
            // BF16 -> F32: shift left by 16 bits
            let b0_u16: uint16x4_t = vld1_u16(b_bf16.as_ptr().add(i));
            let b1_u16: uint16x4_t = vld1_u16(b_bf16.as_ptr().add(i + 4));
            let b2_u16: uint16x4_t = vld1_u16(b_bf16.as_ptr().add(i + 8));
            let b3_u16: uint16x4_t = vld1_u16(b_bf16.as_ptr().add(i + 12));

            // Convert to u32 and shift left by 16
            let b0_u32: uint32x4_t = vshll_n_u16(b0_u16, 16);
            let b1_u32: uint32x4_t = vshll_n_u16(b1_u16, 16);
            let b2_u32: uint32x4_t = vshll_n_u16(b2_u16, 16);
            let b3_u32: uint32x4_t = vshll_n_u16(b3_u16, 16);

            // Reinterpret as f32
            let b0_f32: float32x4_t = vreinterpretq_f32_u32(b0_u32);
            let b1_f32: float32x4_t = vreinterpretq_f32_u32(b1_u32);
            let b2_f32: float32x4_t = vreinterpretq_f32_u32(b2_u32);
            let b3_f32: float32x4_t = vreinterpretq_f32_u32(b3_u32);

            // Load activations
            let a0: float32x4_t = vld1q_f32(a.as_ptr().add(i));
            let a1: float32x4_t = vld1q_f32(a.as_ptr().add(i + 4));
            let a2: float32x4_t = vld1q_f32(a.as_ptr().add(i + 8));
            let a3: float32x4_t = vld1q_f32(a.as_ptr().add(i + 12));

            // FMA: acc = a * b + acc
            acc0 = vfmaq_f32(acc0, a0, b0_f32);
            acc1 = vfmaq_f32(acc1, a1, b1_f32);
            acc2 = vfmaq_f32(acc2, a2, b2_f32);
            acc3 = vfmaq_f32(acc3, a3, b3_f32);

            i += 16;
        }

        // Combine accumulators
        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);

        // Horizontal sum
        let mut sum = vaddvq_f32(acc0);

        // Handle remaining elements
        while i < n {
            sum += a[i] * bf16_to_f32(b_bf16[i]);
            i += 1;
        }

        sum
    }
}

/// Fallback for non-ARM (x86, etc)
#[cfg(not(target_arch = "aarch64"))]
fn dot_bf16_simd(a: &[f32], b_bf16: &[u16]) -> f32 {
    dot_bf16(a, b_bf16)
}

/// SIMD-optimized dot product for f32 vectors.
/// Uses AVX2 on x86-64 for 8-way parallelism.
#[cfg(target_arch = "x86_64")]
fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut sum = 0.0f32;

    unsafe {
        // Process 8 floats at a time with AVX2
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // 4-way unrolled for better ILP (32 floats per iteration)
        let mut i = 0;
        while i + 32 <= n {
            let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);

            let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            acc1 = _mm256_fmadd_ps(a1, b1, acc1);

            let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
            let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
            acc2 = _mm256_fmadd_ps(a2, b2, acc2);

            let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
            let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
            acc3 = _mm256_fmadd_ps(a3, b3, acc3);

            i += 32;
        }

        // Process remaining 8-float chunks
        while i + 8 <= n {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(av, bv, acc0);
            i += 8;
        }

        // Combine accumulators
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

        // Horizontal sum of 8 floats
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(hi, lo);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        sum = _mm_cvtss_f32(sum32);

        // Handle remaining elements
        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }
    }

    sum
}

/// Fallback for non-x86 architectures
#[cfg(not(target_arch = "x86_64"))]
fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

/// Convert F16 to F32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormal
        let e = exp as i32 - 14;
        let m = mant as f32 / 1024.0;
        return if sign == 0 { m * 2.0f32.powi(e) } else { -m * 2.0f32.powi(e) };
    } else if exp == 31 {
        if mant == 0 {
            return if sign == 0 { f32::INFINITY } else { f32::NEG_INFINITY };
        }
        return f32::NAN;
    }

    let f32_exp = exp + 127 - 15;
    let f32_mant = mant << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
}

// bf16_to_f32 is defined earlier in this file (in BF16 Support section)
