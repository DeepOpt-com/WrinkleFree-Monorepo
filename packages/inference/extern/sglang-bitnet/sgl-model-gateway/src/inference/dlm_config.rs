//! Configuration for Diffusion LLM (Fast-dLLM v2) inference.
//!
//! This module provides configuration for block diffusion inference,
//! implementing the Fast-dLLM v2 algorithm (arXiv:2509.26328).

use super::batch_engine::NativeBatchEngine;

/// Decode mode for DLM block diffusion.
///
/// Fast-dLLM v2 uses iterative refinement by default, but we also support
/// a greedy mode that's ~20x faster at the cost of quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DlmDecodeMode {
    /// Single-pass greedy decode (FAST but doesn't follow paper).
    /// Uses one forward pass per block with argmax for all positions.
    /// ~60 tok/s on c3d-standard-32, but may produce lower quality output.
    #[default]
    Greedy,
    /// Iterative refinement with confidence thresholding (CORRECT per paper).
    /// Multiple forward passes per block, unmasking tokens above threshold.
    /// Slower but follows the Fast-dLLM v2 algorithm correctly.
    Iterative,
    /// Adaptive threshold that increases with iterations (FAST + QUALITY).
    /// - Iteration 1: θ=0.5 (unmask easy tokens quickly)
    /// - Iteration 2: θ=0.7 (unmask moderately confident)
    /// - Iteration 3+: θ=final_threshold (full refinement for hard tokens)
    /// This reduces total iterations while maintaining quality for uncertain positions.
    Adaptive,
}

/// Configuration for Diffusion LLM inference.
///
/// Fast-dLLM v2 uses block diffusion to decode multiple tokens in parallel.
/// The ~2.5x speedup over autoregressive decoding is theoretical; actual
/// performance depends on model, hardware, and workload characteristics.
#[derive(Debug, Clone)]
pub struct DlmConfig {
    /// Mask token ID (usually vocab_size - 1 for DLM-trained models)
    pub mask_token_id: i32,
    /// Block size for parallel decoding (default: 32)
    pub block_size: usize,
    /// Confidence threshold for unmasking (default: 0.95)
    pub threshold: f32,
    /// Small block size for sub-block iteration (default: 8)
    pub small_block_size: usize,
    /// Enable DualCache for sub-block KV reuse
    pub use_dual_cache: bool,
    /// Maximum iterations per small block (default: 10)
    /// Prevents runaway loops when confidence is uniformly low.
    pub max_iterations_per_block: usize,
    /// Decode mode: greedy (fast) or iterative (correct per paper)
    pub decode_mode: DlmDecodeMode,
}

impl Default for DlmConfig {
    fn default() -> Self {
        Self {
            mask_token_id: -1, // Must be set via detect() or explicitly
            block_size: 32, // Must match training block size (Fast-dLLM v2 default)
            threshold: 0.7, // Good balance: 89% of greedy speed with iterative correctness
            small_block_size: 8,
            use_dual_cache: true,
            max_iterations_per_block: 10,
            decode_mode: DlmDecodeMode::default(), // Greedy by default for speed
        }
    }
}

impl DlmConfig {
    /// Auto-detect DLM config from a loaded model.
    ///
    /// Attempts to find the mask token `|<MASK>|` in the model's vocabulary.
    /// Returns `None` if the model doesn't have a mask token (not DLM-trained).
    ///
    /// # Example
    /// ```ignore
    /// let engine = NativeBatchEngine::new("dlm-model.gguf", None)?;
    /// let dlm_config = DlmConfig::detect(&engine)
    ///     .expect("Model is not DLM-trained");
    /// ```
    pub fn detect(engine: &NativeBatchEngine) -> Option<Self> {
        // Try to tokenize the mask token
        let mask_token = "|<MASK>|";
        match engine.tokenize(mask_token, false) {
            Ok(tokens) if tokens.len() == 1 => {
                Some(Self {
                    mask_token_id: tokens[0],
                    ..Default::default()
                })
            }
            _ => {
                // Fallback: check if vocab_size - 1 is a special token
                // This is a common pattern for DLM models
                let vocab_size = engine.vocab_size();
                let candidate_mask_id = vocab_size - 1;

                // Try to decode it to verify it's the mask token
                if let Ok(piece) = engine.token_to_piece(candidate_mask_id) {
                    if piece.contains("MASK") || piece.contains("mask") {
                        return Some(Self {
                            mask_token_id: candidate_mask_id,
                            ..Default::default()
                        });
                    }
                }
                None
            }
        }
    }

    /// Create a new DlmConfig with explicit mask token ID.
    pub fn new(mask_token_id: i32) -> Self {
        Self {
            mask_token_id,
            ..Default::default()
        }
    }

    /// Set block size (tokens per parallel decode block).
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Set confidence threshold for unmasking.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set small block size for sub-block iteration.
    pub fn with_small_block_size(mut self, small_block_size: usize) -> Self {
        self.small_block_size = small_block_size;
        self
    }

    /// Enable or disable DualCache.
    pub fn with_dual_cache(mut self, enable: bool) -> Self {
        self.use_dual_cache = enable;
        self
    }

    /// Set maximum iterations per small block.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations_per_block = max_iterations;
        self
    }

    /// Set decode mode (greedy or iterative).
    pub fn with_decode_mode(mut self, mode: DlmDecodeMode) -> Self {
        self.decode_mode = mode;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.mask_token_id < 0 {
            return Err("mask_token_id must be non-negative".to_string());
        }
        if self.block_size == 0 {
            return Err("block_size must be positive".to_string());
        }
        if self.small_block_size == 0 {
            return Err("small_block_size must be positive".to_string());
        }
        if self.block_size % self.small_block_size != 0 {
            return Err(format!(
                "block_size ({}) must be divisible by small_block_size ({})",
                self.block_size, self.small_block_size
            ));
        }
        if self.threshold <= 0.0 || self.threshold >= 1.0 {
            return Err("threshold must be between 0 and 1 (exclusive)".to_string());
        }
        if self.max_iterations_per_block == 0 {
            return Err("max_iterations_per_block must be positive".to_string());
        }
        Ok(())
    }

    /// Get the number of small blocks per block.
    pub fn num_small_blocks(&self) -> usize {
        self.block_size / self.small_block_size
    }
}

/// Result of mask token detection with diagnostic information.
#[derive(Debug, Clone)]
pub struct MaskTokenDetectionResult {
    /// Detected mask token ID, if found
    pub mask_token_id: Option<i32>,
    /// How the mask token was detected
    pub detection_method: String,
    /// Candidates checked during detection (token_id, token_piece)
    pub candidates_checked: Vec<(i32, String)>,
    /// Vocab size of the model
    pub vocab_size: i32,
}

impl MaskTokenDetectionResult {
    /// Format a detailed error message for display.
    pub fn format_error(&self) -> String {
        let sep = "======================================================================";
        let mut msg = String::new();
        msg.push_str(sep);
        msg.push_str("\nMASK TOKEN DETECTION FAILED\n");
        msg.push_str(sep);
        msg.push_str("\n\nThis model does not appear to be a DLM (Diffusion LLM) model.\n");
        msg.push_str("DLM models require a special |<MASK>| token for block diffusion.\n\n");

        msg.push_str("Detection attempts:\n");
        msg.push_str(&format!("  1. Tokenized '|<MASK>|': {}\n", self.detection_method));

        if !self.candidates_checked.is_empty() {
            msg.push_str("  2. Checked vocab candidates:\n");
            for (id, piece) in &self.candidates_checked {
                msg.push_str(&format!("     - Token {}: '{}'\n", id, piece));
            }
        }

        msg.push_str(&format!("\nModel vocab size: {}\n\n", self.vocab_size));

        msg.push_str("SOLUTIONS:\n");
        msg.push_str("  1. Use a DLM-trained checkpoint (from WrinkleFree training)\n");
        msg.push_str("  2. Manually specify mask token: --mask-token-id <ID>\n");
        msg.push_str("  3. For regular BitNet (non-DLM), use native_server instead\n\n");

        msg.push_str("Expected mask token names: |<MASK>|, <mask>, [MASK]\n");
        msg.push_str(sep);
        msg.push('\n');
        msg
    }
}

impl DlmConfig {
    /// Auto-detect DLM config with detailed diagnostics.
    ///
    /// Unlike `detect()`, this always returns a result with diagnostic info,
    /// which is useful for generating helpful error messages.
    pub fn detect_with_diagnostics(engine: &NativeBatchEngine) -> MaskTokenDetectionResult {
        let vocab_size = engine.vocab_size();
        let mut candidates_checked = Vec::new();
        let mut detection_method = String::new();

        // Try to tokenize the mask token
        let mask_token = "|<MASK>|";
        match engine.tokenize(mask_token, false) {
            Ok(tokens) if tokens.len() == 1 => {
                detection_method = format!("Found as single token {}", tokens[0]);
                return MaskTokenDetectionResult {
                    mask_token_id: Some(tokens[0]),
                    detection_method,
                    candidates_checked,
                    vocab_size,
                };
            }
            Ok(tokens) => {
                detection_method = format!("Tokenized to {} tokens (expected 1)", tokens.len());
            }
            Err(e) => {
                detection_method = format!("Tokenization failed: {}", e);
            }
        }

        // Check vocab_size - 1 (common DLM pattern)
        let candidate_id = vocab_size - 1;
        if let Ok(piece) = engine.token_to_piece(candidate_id) {
            candidates_checked.push((candidate_id, piece.clone()));
            if piece.contains("MASK") || piece.contains("mask") {
                return MaskTokenDetectionResult {
                    mask_token_id: Some(candidate_id),
                    detection_method: format!("Found at vocab_size-1 ({})", candidate_id),
                    candidates_checked,
                    vocab_size,
                };
            }
        }

        // Check a few more special token positions
        for offset in [0, 1, 2, 3] {
            let candidate_id = vocab_size - 1 - offset;
            if candidate_id < 0 {
                continue;
            }
            if candidates_checked.iter().any(|(id, _)| *id == candidate_id) {
                continue;
            }
            if let Ok(piece) = engine.token_to_piece(candidate_id) {
                candidates_checked.push((candidate_id, piece.clone()));
                if piece.contains("MASK") || piece.contains("mask") {
                    return MaskTokenDetectionResult {
                        mask_token_id: Some(candidate_id),
                        detection_method: format!("Found at vocab_size-{} ({})", offset + 1, candidate_id),
                        candidates_checked,
                        vocab_size,
                    };
                }
            }
        }

        // Not found
        MaskTokenDetectionResult {
            mask_token_id: None,
            detection_method,
            candidates_checked,
            vocab_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dlm_config_defaults() {
        let config = DlmConfig::default();
        assert_eq!(config.block_size, 32); // Must match training
        assert!((config.threshold - 0.95).abs() < 1e-6);
        assert_eq!(config.small_block_size, 8);
        assert!(config.use_dual_cache);
    }

    #[test]
    fn test_dlm_config_builder() {
        let config = DlmConfig::new(128256)
            .with_block_size(64)
            .with_threshold(0.9)
            .with_small_block_size(16)
            .with_dual_cache(false);

        assert_eq!(config.mask_token_id, 128256);
        assert_eq!(config.block_size, 64);
        assert!((config.threshold - 0.9).abs() < 1e-6);
        assert_eq!(config.small_block_size, 16);
        assert!(!config.use_dual_cache);
    }

    #[test]
    fn test_dlm_config_validation() {
        // Valid config
        let config = DlmConfig::new(128256);
        assert!(config.validate().is_ok());

        // Invalid: negative mask token
        let config = DlmConfig::default();
        assert!(config.validate().is_err());

        // Invalid: block_size not divisible by small_block_size
        let config = DlmConfig::new(128256)
            .with_block_size(30)
            .with_small_block_size(8);
        assert!(config.validate().is_err());

        // Invalid: threshold out of range
        let config = DlmConfig::new(128256).with_threshold(1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_num_small_blocks() {
        let config = DlmConfig::new(128256)
            .with_block_size(32)
            .with_small_block_size(8);
        assert_eq!(config.num_small_blocks(), 4);

        let config = DlmConfig::new(128256)
            .with_block_size(64)
            .with_small_block_size(16);
        assert_eq!(config.num_small_blocks(), 4);
    }
}
