//! Native BitNet inference engine backend.
//!
//! This module provides a direct C++ backend that bypasses the Python gRPC server,
//! eliminating ~49ms of overhead per token for faster inference.

use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

#[cfg(feature = "native-inference")]
use crate::inference::{NativeEngine as CppEngine, SamplingParams as CppSamplingParams};

use crate::protocols::sampling_params::SamplingParams;

/// Result type for native engine operations
pub type NativeEngineResult<T> = Result<T, NativeEngineError>;

/// Errors from the native engine
#[derive(Debug, thiserror::Error)]
pub enum NativeEngineError {
    #[error("Native inference not enabled. Compile with --features native-inference")]
    NotEnabled,

    #[error("Engine creation failed: {0}")]
    EngineCreation(String),

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Model not loaded")]
    NotLoaded,
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_layers: i32,
    pub max_seq_len: i32,
    pub model_path: String,
}

/// Generation result
#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub text: String,
    pub token_ids: Vec<i32>,
    pub num_prompt_tokens: usize,
    pub num_generated_tokens: usize,
}

/// Native BitNet inference engine client.
///
/// This is a drop-in replacement for `SglangSchedulerClient` that uses
/// the native C++ engine instead of gRPC to Python.
#[derive(Clone)]
pub struct NativeEngineClient {
    #[cfg(feature = "native-inference")]
    engine: Arc<Mutex<Option<CppEngine>>>,

    #[cfg(not(feature = "native-inference"))]
    _phantom: std::marker::PhantomData<()>,

    model_info: Arc<Mutex<Option<ModelInfo>>>,
    tokenizer: Arc<Mutex<Option<tokenizers::Tokenizer>>>,
}

impl NativeEngineClient {
    /// Create a new native engine client.
    ///
    /// Unlike the gRPC client, this needs a model path to load the model directly.
    pub async fn new(model_path: &str) -> NativeEngineResult<Self> {
        #[cfg(feature = "native-inference")]
        {
            info!("Creating native BitNet engine for: {}", model_path);

            // Load tokenizer from HuggingFace
            let tokenizer = Self::load_tokenizer(model_path).await?;

            // Create C++ engine
            let engine = CppEngine::new(model_path, None)
                .map_err(|e| NativeEngineError::EngineCreation(e.to_string()))?;

            let model_info = ModelInfo {
                vocab_size: engine.vocab_size(),
                hidden_size: engine.hidden_size(),
                num_layers: engine.num_layers(),
                max_seq_len: engine.max_seq_len(),
                model_path: model_path.to_string(),
            };

            info!(
                "Native engine created: {} layers, vocab_size={}, max_seq_len={}",
                model_info.num_layers, model_info.vocab_size, model_info.max_seq_len
            );

            Ok(Self {
                engine: Arc::new(Mutex::new(Some(engine))),
                model_info: Arc::new(Mutex::new(Some(model_info))),
                tokenizer: Arc::new(Mutex::new(Some(tokenizer))),
            })
        }

        #[cfg(not(feature = "native-inference"))]
        {
            let _ = model_path;
            Err(NativeEngineError::NotEnabled)
        }
    }

    /// Load tokenizer from HuggingFace hub or local path
    async fn load_tokenizer(model_path: &str) -> NativeEngineResult<tokenizers::Tokenizer> {
        let local_path = std::path::Path::new(model_path);

        // If model_path is a file (e.g., a .gguf file), look for tokenizer in parent directory
        if local_path.is_file() {
            if let Some(parent) = local_path.parent() {
                let tokenizer_path = parent.join("tokenizer.json");
                if tokenizer_path.exists() {
                    info!("Loading tokenizer from: {}", tokenizer_path.display());
                    return tokenizers::Tokenizer::from_file(&tokenizer_path)
                        .map_err(|e| NativeEngineError::TokenizerError(e.to_string()));
                }
            }
        }

        // Try loading from local path (if model_path is a directory)
        if local_path.is_dir() {
            let tokenizer_path = local_path.join("tokenizer.json");
            if tokenizer_path.exists() {
                info!("Loading tokenizer from: {}", tokenizer_path.display());
                return tokenizers::Tokenizer::from_file(&tokenizer_path)
                    .map_err(|e| NativeEngineError::TokenizerError(e.to_string()));
            }
        }

        // Try loading from HuggingFace cache
        let home = std::env::var("HOME")
            .map_err(|_| NativeEngineError::TokenizerError("HOME not set".into()))?;
        let cache_dir = std::path::PathBuf::from(home).join(".cache/huggingface/hub");

        // Convert model ID to cache path format (e.g., "microsoft/bitnet-b1.58-2B-4T" -> "models--microsoft--bitnet-b1.58-2B-4T")
        let model_cache_name = format!("models--{}", model_path.replace('/', "--"));
        let cache_path = cache_dir.join(&model_cache_name).join("snapshots");

        if cache_path.exists() {
            // Find the latest snapshot
            if let Ok(entries) = std::fs::read_dir(&cache_path) {
                for entry in entries.flatten() {
                    let tokenizer_path = entry.path().join("tokenizer.json");
                    if tokenizer_path.exists() {
                        info!("Loading tokenizer from: {}", tokenizer_path.display());
                        return tokenizers::Tokenizer::from_file(&tokenizer_path)
                            .map_err(|e| NativeEngineError::TokenizerError(e.to_string()));
                    }
                }
            }
        }

        Err(NativeEngineError::TokenizerError(format!(
            "Could not find tokenizer for model '{}'. Try downloading with: huggingface-cli download {}",
            model_path, model_path
        )))
    }

    /// Generate text from a prompt.
    #[cfg(feature = "native-inference")]
    pub async fn generate(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> NativeEngineResult<GenerateResult> {
        // Tokenize input
        let token_ids = {
            let tokenizer_guard = self.tokenizer.lock().await;
            let tokenizer = tokenizer_guard
                .as_ref()
                .ok_or(NativeEngineError::NotLoaded)?;

            let encoding = tokenizer
                .encode(prompt, false)
                .map_err(|e| NativeEngineError::TokenizerError(e.to_string()))?;

            encoding.get_ids().iter().map(|&id| id as i32).collect::<Vec<_>>()
        };

        let num_prompt_tokens = token_ids.len();

        // Generate with C++ engine
        let output_ids = {
            let mut engine_guard = self.engine.lock().await;
            let engine = engine_guard
                .as_mut()
                .ok_or(NativeEngineError::NotLoaded)?;

            let cpp_params = CppSamplingParams {
                temperature: params.temperature.unwrap_or(0.7) as f32,
                top_p: params.top_p.unwrap_or(0.9) as f32,
                top_k: params.top_k.map(|k| k as i32).unwrap_or(0),
                repetition_penalty: params.repetition_penalty.unwrap_or(1.0) as f32,
                max_tokens: params.max_new_tokens.unwrap_or(256) as i32,
            };

            let result = engine
                .generate(&token_ids, &cpp_params)
                .map_err(|e| NativeEngineError::GenerationFailed(e.to_string()))?;

            result.output_ids
        };

        let num_generated_tokens = output_ids.len();

        // Decode output
        let text = {
            let tokenizer_guard = self.tokenizer.lock().await;
            let tokenizer = tokenizer_guard
                .as_ref()
                .ok_or(NativeEngineError::NotLoaded)?;

            let ids: Vec<u32> = output_ids.iter().map(|&id| id as u32).collect();
            tokenizer
                .decode(&ids, true)
                .map_err(|e| NativeEngineError::TokenizerError(e.to_string()))?
        };

        debug!(
            "Generated {} tokens from {} prompt tokens",
            num_generated_tokens, num_prompt_tokens
        );

        Ok(GenerateResult {
            text,
            token_ids: output_ids,
            num_prompt_tokens,
            num_generated_tokens,
        })
    }

    #[cfg(not(feature = "native-inference"))]
    pub async fn generate(
        &self,
        _prompt: &str,
        _params: &SamplingParams,
    ) -> NativeEngineResult<GenerateResult> {
        Err(NativeEngineError::NotEnabled)
    }

    /// Get model information.
    pub async fn get_model_info(&self) -> NativeEngineResult<ModelInfo> {
        let guard = self.model_info.lock().await;
        guard.clone().ok_or(NativeEngineError::NotLoaded)
    }

    /// Health check - returns true if engine is loaded.
    pub async fn health_check(&self) -> bool {
        #[cfg(feature = "native-inference")]
        {
            let guard = self.engine.lock().await;
            guard.is_some()
        }

        #[cfg(not(feature = "native-inference"))]
        false
    }

    /// Encode text to token IDs.
    #[cfg(feature = "native-inference")]
    pub async fn encode(&self, text: &str) -> NativeEngineResult<Vec<i32>> {
        let tokenizer_guard = self.tokenizer.lock().await;
        let tokenizer = tokenizer_guard
            .as_ref()
            .ok_or(NativeEngineError::NotLoaded)?;

        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| NativeEngineError::TokenizerError(e.to_string()))?;

        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }

    #[cfg(not(feature = "native-inference"))]
    pub async fn encode(&self, _text: &str) -> NativeEngineResult<Vec<i32>> {
        Err(NativeEngineError::NotEnabled)
    }

    /// Decode token IDs to text.
    #[cfg(feature = "native-inference")]
    pub async fn decode(&self, token_ids: &[i32]) -> NativeEngineResult<String> {
        let tokenizer_guard = self.tokenizer.lock().await;
        let tokenizer = tokenizer_guard
            .as_ref()
            .ok_or(NativeEngineError::NotLoaded)?;

        let ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
        tokenizer
            .decode(&ids, true)
            .map_err(|e| NativeEngineError::TokenizerError(e.to_string()))
    }

    #[cfg(not(feature = "native-inference"))]
    pub async fn decode(&self, _token_ids: &[i32]) -> NativeEngineResult<String> {
        Err(NativeEngineError::NotEnabled)
    }

    /// Prefill phase - process input tokens and populate KV cache.
    #[cfg(feature = "native-inference")]
    pub async fn prefill(&self, token_ids: &[i32]) -> NativeEngineResult<()> {
        let mut engine_guard = self.engine.lock().await;
        let engine = engine_guard
            .as_mut()
            .ok_or(NativeEngineError::NotLoaded)?;

        engine
            .prefill(token_ids)
            .map_err(|e| NativeEngineError::GenerationFailed(e.to_string()))
    }

    #[cfg(not(feature = "native-inference"))]
    pub async fn prefill(&self, _token_ids: &[i32]) -> NativeEngineResult<()> {
        Err(NativeEngineError::NotEnabled)
    }

    /// Single decode step - generate one token.
    #[cfg(feature = "native-inference")]
    pub async fn decode_step(&self, position: i32, params: &SamplingParams) -> NativeEngineResult<i32> {
        let mut engine_guard = self.engine.lock().await;
        let engine = engine_guard
            .as_mut()
            .ok_or(NativeEngineError::NotLoaded)?;

        let cpp_params = CppSamplingParams {
            temperature: params.temperature.unwrap_or(0.7) as f32,
            top_p: params.top_p.unwrap_or(0.9) as f32,
            top_k: params.top_k.map(|k| k as i32).unwrap_or(0),
            repetition_penalty: params.repetition_penalty.unwrap_or(1.0) as f32,
            max_tokens: params.max_new_tokens.unwrap_or(256) as i32,
        };

        engine
            .decode_step(position, &cpp_params)
            .map_err(|e| NativeEngineError::GenerationFailed(e.to_string()))
    }

    #[cfg(not(feature = "native-inference"))]
    pub async fn decode_step(&self, _position: i32, _params: &SamplingParams) -> NativeEngineResult<i32> {
        Err(NativeEngineError::NotEnabled)
    }

    /// Reset the KV cache (start new sequence).
    #[cfg(feature = "native-inference")]
    pub async fn reset_cache(&self) {
        let mut engine_guard = self.engine.lock().await;
        if let Some(engine) = engine_guard.as_mut() {
            engine.reset_cache();
        }
    }

    #[cfg(not(feature = "native-inference"))]
    pub async fn reset_cache(&self) {}

    /// Get the EOS token ID.
    #[cfg(feature = "native-inference")]
    pub async fn eos_token_id(&self) -> NativeEngineResult<i32> {
        let tokenizer_guard = self.tokenizer.lock().await;
        let tokenizer = tokenizer_guard
            .as_ref()
            .ok_or(NativeEngineError::NotLoaded)?;

        // Try to get EOS token from tokenizer
        // BitNet uses <|eot_id|> as EOS
        if let Some(id) = tokenizer.token_to_id("<|eot_id|>") {
            return Ok(id as i32);
        }
        if let Some(id) = tokenizer.token_to_id("</s>") {
            return Ok(id as i32);
        }
        if let Some(id) = tokenizer.token_to_id("<|end_of_text|>") {
            return Ok(id as i32);
        }
        // Default EOS token ID for llama-based models
        Ok(2)
    }

    #[cfg(not(feature = "native-inference"))]
    pub async fn eos_token_id(&self) -> NativeEngineResult<i32> {
        Err(NativeEngineError::NotEnabled)
    }
}

impl Default for NativeEngineClient {
    fn default() -> Self {
        Self {
            #[cfg(feature = "native-inference")]
            engine: Arc::new(Mutex::new(None)),

            #[cfg(not(feature = "native-inference"))]
            _phantom: std::marker::PhantomData,

            model_info: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "Requires model to be downloaded"]
    async fn test_native_engine_creation() {
        let result = NativeEngineClient::new("microsoft/bitnet-b1.58-2B-4T").await;
        // Should succeed with native-inference feature, or return NotEnabled without
        assert!(result.is_ok() || matches!(result, Err(NativeEngineError::NotEnabled)));
    }
}
