//! DLM server with Fast-dLLM v2 block diffusion inference.
//!
//! This server implements Fast-dLLM v2 (arXiv:2509.26328) for ~2.5x faster
//! inference via parallel block decoding.
//!
//! Usage:
//!   cargo run --release --features native-inference --bin dlm_server -- \
//!     --model-path /path/to/dlm-model.gguf --port 30000

// Use mimalloc as the global allocator for better performance
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tracing::{error, info};

#[cfg(feature = "native-inference")]
use sgl_model_gateway::inference::{
    BatchConfig, BatchSamplingParams, DlmConfig, DlmScheduler, DlmSchedulerConfig,
    DlmSchedulerHandle, InferenceRequest, NativeBatchEngine,
};

async fn health() -> &'static str {
    "ok"
}

// OpenAI-compatible request/response types
#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: i32,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
}

fn default_max_tokens() -> i32 {
    256
}

#[derive(Debug, Serialize)]
struct ChatCompletionChoice {
    index: i32,
    message: ChatCompletionMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: ChatCompletionUsage,
}

#[cfg(feature = "native-inference")]
struct AppState {
    handle: DlmSchedulerHandle,
    engine: Arc<NativeBatchEngine>,
    request_counter: AtomicU64,
}

#[cfg(feature = "native-inference")]
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    // Build prompt using Llama 3 chat format
    // Note: Don't add BOS - tokenizer adds it automatically via add_bos=true
    let mut prompt = String::new();
    for msg in &req.messages {
        prompt.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

    // Tokenize
    let input_ids = state
        .engine
        .tokenize(&prompt, true)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let prompt_tokens = input_ids.len() as i32;

    // Create request
    let request_id = state.request_counter.fetch_add(1, Ordering::Relaxed);
    let (tx, rx) = oneshot::channel();

    let inference_req = InferenceRequest {
        request_id,
        input_ids,
        params: BatchSamplingParams {
            temperature: req.temperature.unwrap_or(0.7),
            top_p: req.top_p.unwrap_or(0.9),
            ..Default::default()
        },
        max_tokens: req.max_tokens,
        stream: false,
        response_tx: Some(tx),
        token_tx: None,
    };

    // Submit to scheduler
    state
        .handle
        .submit(inference_req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Wait for response
    let response = rx
        .await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Request cancelled".to_string()))?;

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", request_id),
        object: "chat.completion".to_string(),
        created,
        model: "dlm".to_string(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: response.text,
            },
            finish_reason: response.finish_reason.to_string(),
        }],
        usage: ChatCompletionUsage {
            prompt_tokens,
            completion_tokens: response.completion_tokens,
            total_tokens: prompt_tokens + response.completion_tokens,
        },
    }))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    // Parse args
    let mut model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "dlm-bitnet.gguf".to_string());
    let mut port: u16 = 30000;
    let mut host = String::from("0.0.0.0");
    let mut max_sequences: usize = 16;
    let mut block_size: usize = 32;
    let mut threshold: f32 = 0.95;
    let mut small_block_size: usize = 8;
    let mut mask_token_id: Option<i32> = None;
    let mut force_unmask: bool = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" | "-m" => {
                model_path = args.get(i + 1).cloned().unwrap_or(model_path);
                i += 2;
            }
            "--port" | "-p" => {
                port = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(port);
                i += 2;
            }
            "--host" | "-h" => {
                host = args.get(i + 1).cloned().unwrap_or(host);
                i += 2;
            }
            "--max-sequences" => {
                max_sequences = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(max_sequences);
                i += 2;
            }
            "--block-size" => {
                block_size = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(block_size);
                i += 2;
            }
            "--threshold" => {
                threshold = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(threshold);
                i += 2;
            }
            "--small-block-size" => {
                small_block_size = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(small_block_size);
                i += 2;
            }
            "--mask-token-id" => {
                mask_token_id = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--help" => {
                println!("DLM Server - Fast-dLLM v2 Block Diffusion Inference");
                println!();
                println!("Usage: dlm_server [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -m, --model-path PATH    Path to DLM model (GGUF format)");
                println!("  -p, --port PORT          Server port (default: 30000)");
                println!("  -h, --host HOST          Server host (default: 0.0.0.0)");
                println!("  --max-sequences N        Max concurrent sequences (default: 16)");
                println!("  --block-size N           Block size for parallel decode (default: 32)");
                println!("  --threshold F            Confidence threshold 0-1 (default: 0.95)");
                println!("  --small-block-size N     Sub-block size (default: 8)");
                println!("  --mask-token-id ID       Override mask token ID (for benchmarking non-DLM models)");
                println!("  --help                   Show this help");
                return;
            }
            _ => i += 1,
        }
    }

    info!("=== DLM Server (Fast-dLLM v2) ===");
    info!("Model: {}", model_path);
    info!("Host:  {}:{}", host, port);
    info!("Max sequences: {}", max_sequences);
    info!("Block size: {} (small: {})", block_size, small_block_size);
    info!("Threshold: {}", threshold);

    #[cfg(feature = "native-inference")]
    {
        use std::sync::Arc;

        // Load engine
        let batch_config = BatchConfig {
            max_sequences: max_sequences as i32,
            ..Default::default()
        };

        let engine = match NativeBatchEngine::new(&model_path, Some(batch_config)) {
            Ok(e) => Arc::new(e),
            Err(e) => {
                error!("Failed to load model: {}", e);
                std::process::exit(1);
            }
        };

        info!("Model loaded: vocab_size={}", engine.vocab_size());

        // Detect or configure DLM
        let dlm_config = if let Some(manual_mask_id) = mask_token_id {
            // Manual mask token ID provided - use it directly
            info!(
                "Using manual mask_token_id={} (benchmark mode)",
                manual_mask_id
            );
            DlmConfig::new(manual_mask_id)
                .with_block_size(block_size)
                .with_threshold(threshold)
                .with_small_block_size(small_block_size)
        } else {
            // Auto-detect DLM model
            match DlmConfig::detect(&engine) {
                Some(mut config) => {
                    info!(
                        "Detected DLM model: mask_token_id={}",
                        config.mask_token_id
                    );
                    config.block_size = block_size;
                    config.threshold = threshold;
                    config.small_block_size = small_block_size;
                    config
                }
                None => {
                    error!("Model does not appear to be DLM-trained (no mask token found)");
                    error!("Use --mask-token-id to manually specify a mask token for benchmarking");
                    std::process::exit(1);
                }
            }
        };

        if let Err(e) = dlm_config.validate() {
            error!("Invalid DLM config: {}", e);
            std::process::exit(1);
        }

        let scheduler_config = DlmSchedulerConfig {
            max_sequences,
            dlm: dlm_config,
            enable_radix_cache: true,
            radix_cache_max_tokens: 100_000,
        };

        let (mut scheduler, handle) = DlmScheduler::new(scheduler_config, engine.clone());

        info!("DLM scheduler initialized");

        // Create app state
        let app_state = Arc::new(AppState {
            handle,
            engine,
            request_counter: AtomicU64::new(0),
        });

        // Create router with chat completions endpoint
        let app = Router::new()
            .route("/health", get(health))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(app_state);

        let addr: SocketAddr = format!("{}:{}", host, port).parse().unwrap();
        info!("Server listening on http://{}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

        // Run server and scheduler concurrently
        let scheduler_handle = tokio::spawn(async move {
            scheduler.run().await;
        });

        tokio::select! {
            _ = axum::serve(listener, app) => {
                info!("Server shut down");
            }
            _ = scheduler_handle => {
                info!("Scheduler shut down");
            }
        }
    }

    #[cfg(not(feature = "native-inference"))]
    {
        error!("Native inference not enabled. Build with --features native-inference");
        std::process::exit(1);
    }
}
