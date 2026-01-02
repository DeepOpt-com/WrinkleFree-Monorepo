//! Configuration for Diffusion LLM (Fast-dLLM v2) inference.
//!
//! This module provides configuration for block diffusion inference,
//! implementing the Fast-dLLM v2 algorithm (arXiv:2509.26328).

use super::batch_engine::NativeBatchEngine;

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
}

impl Default for DlmConfig {
    fn default() -> Self {
        Self {
            mask_token_id: -1, // Must be set via detect() or explicitly
            block_size: 32,
            threshold: 0.95,
            small_block_size: 8,
            use_dual_cache: true,
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
        Ok(())
    }

    /// Get the number of small blocks per block.
    pub fn num_small_blocks(&self) -> usize {
        self.block_size / self.small_block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dlm_config_defaults() {
        let config = DlmConfig::default();
        assert_eq!(config.block_size, 32);
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
