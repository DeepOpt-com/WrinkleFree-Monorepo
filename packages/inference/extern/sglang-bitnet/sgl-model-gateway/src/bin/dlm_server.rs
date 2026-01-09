//! DLM server with Fast-dLLM v2 block diffusion inference.
//!
//! This server implements Fast-dLLM v2 (arXiv:2509.26328) for ~2.5x faster
//! inference via parallel block decoding.
//!
//! Usage:
//!   cargo run --release --features native-inference --bin dlm_server -- \
//!     --model-path /path/to/dlm-model.gguf --port 30000
//!
//! Or with YAML config:
//!   cargo run --release --features native-inference --bin dlm_server -- \
//!     --config /path/to/dlm_config.yaml

// Use mimalloc as the global allocator for better performance
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::net::SocketAddr;
use std::path::Path;
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
use wf_inference::inference::{
    BatchConfig, BatchSamplingParams, DlmConfig, DlmDecodeMode, DlmScheduler, DlmSchedulerConfig,
    DlmSchedulerHandle, InferenceRequest, NativeBatchEngine,
};

/// Server configuration loaded from YAML or CLI args.
///
/// Example YAML config (dlm_config.yaml):
/// ```yaml
/// model_path: /path/to/model.gguf
/// host: 0.0.0.0
/// port: 30000
///
/// dlm:
///   block_size: 32        # Must match training
///   threshold: 0.95
///   small_block_size: 8
///   mask_token_id: null   # Auto-detect if null
///
/// scheduler:
///   max_sequences: 16
///   enable_radix_cache: true
///   radix_cache_max_tokens: 100000
///
/// benchmark:
///   enabled: false
///   iterations: 50
///   max_tokens: 64
///   prompt: "What is the meaning of life?"
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Path to GGUF model file
    pub model_path: String,
    /// Server host
    pub host: String,
    /// Server port
    pub port: u16,
    /// DLM-specific settings
    pub dlm: DlmSettings,
    /// Scheduler settings
    pub scheduler: SchedulerSettings,
    /// Benchmark settings
    pub benchmark: BenchmarkSettings,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct DlmSettings {
    /// Block size for parallel decoding (must match training)
    pub block_size: usize,
    /// Confidence threshold for unmasking (0.0-1.0)
    pub threshold: f32,
    /// Small block size for sub-block iteration
    pub small_block_size: usize,
    /// Mask token ID (null = auto-detect)
    pub mask_token_id: Option<i32>,
    /// Maximum iterations per block (safety limit)
    pub max_iterations_per_block: usize,
    /// Decode mode: "greedy" (fast, ~120 tok/s) or "iterative" (correct per paper)
    pub decode_mode: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct SchedulerSettings {
    /// Maximum concurrent sequences
    pub max_sequences: usize,
    /// Enable RadixCache for prefix caching
    pub enable_radix_cache: bool,
    /// Maximum tokens in RadixCache
    pub radix_cache_max_tokens: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct BenchmarkSettings {
    /// Run in benchmark mode (no server)
    pub enabled: bool,
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Max tokens per iteration
    pub max_tokens: i32,
    /// Prompt for benchmarking
    pub prompt: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_path: "dlm-bitnet.gguf".to_string(),
            host: "0.0.0.0".to_string(),
            port: 30000,
            dlm: DlmSettings::default(),
            scheduler: SchedulerSettings::default(),
            benchmark: BenchmarkSettings::default(),
        }
    }
}

impl Default for DlmSettings {
    fn default() -> Self {
        Self {
            block_size: 32, // Must match training (Fast-dLLM v2 default)
            threshold: 0.7, // Good balance: 89% of greedy speed with iterative correctness
            small_block_size: 8,
            mask_token_id: None, // Auto-detect
            max_iterations_per_block: 10,
            decode_mode: "greedy".to_string(), // Fast mode by default
        }
    }
}

impl Default for SchedulerSettings {
    fn default() -> Self {
        Self {
            max_sequences: 16,
            enable_radix_cache: true,
            radix_cache_max_tokens: 100_000,
        }
    }
}

impl Default for BenchmarkSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            iterations: 50,
            max_tokens: 64,
            prompt: "What is the meaning of life?".to_string(),
        }
    }
}

impl ServerConfig {
    /// Load config from YAML file.
    pub fn from_yaml(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;
        serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse YAML config: {}", e))
    }

    /// Generate example YAML config.
    pub fn example_yaml() -> String {
        let config = Self::default();
        serde_yaml::to_string(&config).unwrap_or_default()
    }
}

#[cfg(feature = "native-inference")]
impl DlmSettings {
    /// Parse decode_mode string to DlmDecodeMode enum.
    pub fn parse_decode_mode(&self) -> DlmDecodeMode {
        match self.decode_mode.to_lowercase().as_str() {
            "iterative" | "iter" => DlmDecodeMode::Iterative,
            "adaptive" | "adapt" => DlmDecodeMode::Adaptive,
            _ => DlmDecodeMode::Greedy, // Default to greedy for speed
        }
    }
}

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
    #[allow(dead_code)] // Reserved for future streaming support
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
    // Build prompt from messages using Llama 3 format
    let mut prompt = String::new();
    for msg in &req.messages {
        prompt.push_str(&format!("{}: {}<|eot_id|>", msg.role.to_uppercase(), msg.content));
    }
    prompt.push_str("ASSISTANT:");

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

/// Parse CLI args and merge with YAML config.
/// Priority: CLI args > YAML config > defaults
fn parse_config() -> ServerConfig {
    let args: Vec<String> = std::env::args().collect();

    // First pass: look for --config to load base config
    let mut config = ServerConfig::default();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--config" || args[i] == "-c" {
            if let Some(path) = args.get(i + 1) {
                match ServerConfig::from_yaml(Path::new(path)) {
                    Ok(yaml_config) => {
                        info!("Loaded config from: {}", path);
                        config = yaml_config;
                    }
                    Err(e) => {
                        eprintln!("Error loading config: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            break;
        }
        i += 1;
    }

    // Override with environment variable
    if let Ok(model_path) = std::env::var("MODEL_PATH") {
        config.model_path = model_path;
    }

    // Second pass: override with CLI args
    i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" => {
                i += 2; // Already handled
            }
            "--model-path" | "-m" => {
                if let Some(v) = args.get(i + 1) {
                    config.model_path = v.clone();
                }
                i += 2;
            }
            "--port" | "-p" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    config.port = v;
                }
                i += 2;
            }
            "--host" => {
                if let Some(v) = args.get(i + 1) {
                    config.host = v.clone();
                }
                i += 2;
            }
            "--max-sequences" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    config.scheduler.max_sequences = v;
                }
                i += 2;
            }
            "--block-size" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    config.dlm.block_size = v;
                }
                i += 2;
            }
            "--threshold" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    config.dlm.threshold = v;
                }
                i += 2;
            }
            "--small-block-size" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    config.dlm.small_block_size = v;
                }
                i += 2;
            }
            "--mask-token-id" => {
                config.dlm.mask_token_id = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--decode-mode" => {
                if let Some(v) = args.get(i + 1) {
                    config.dlm.decode_mode = v.clone();
                }
                i += 2;
            }
            "--benchmark" => {
                config.benchmark.enabled = true;
                i += 1;
            }
            "--benchmark-iterations" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    config.benchmark.iterations = v;
                }
                i += 2;
            }
            "--benchmark-prompt" => {
                if let Some(v) = args.get(i + 1) {
                    config.benchmark.prompt = v.clone();
                }
                i += 2;
            }
            "--benchmark-max-tokens" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    config.benchmark.max_tokens = v;
                }
                i += 2;
            }
            "--generate-config" => {
                // Print example YAML config and exit
                println!("# DLM Server Configuration");
                println!("# Save this to a file and use with --config");
                println!("{}", ServerConfig::example_yaml());
                std::process::exit(0);
            }
            "--help" | "-h" => {
                println!("DLM Server - Fast-dLLM v2 Block Diffusion Inference");
                println!();
                println!("Usage: dlm_server [OPTIONS]");
                println!();
                println!("Config:");
                println!("  -c, --config PATH        Load config from YAML file");
                println!("  --generate-config        Print example YAML config and exit");
                println!();
                println!("Server Options (override config):");
                println!("  -m, --model-path PATH    Path to DLM model (GGUF format)");
                println!("  -p, --port PORT          Server port (default: 30000)");
                println!("  --host HOST              Server host (default: 0.0.0.0)");
                println!();
                println!("DLM Options:");
                println!("  --block-size N           Block size for parallel decode (default: 32)");
                println!("  --threshold F            Confidence threshold 0-1 (default: 0.7)");
                println!("  --small-block-size N     Sub-block size (default: 8)");
                println!("  --mask-token-id ID       Override mask token ID");
                println!("  --decode-mode MODE       greedy | iterative | adaptive");
                println!();
                println!("Scheduler Options:");
                println!("  --max-sequences N        Max concurrent sequences (default: 16)");
                println!();
                println!("Benchmark Options:");
                println!("  --benchmark              Run benchmark mode (no server)");
                println!("  --benchmark-iterations N Iterations (default: 50)");
                println!("  --benchmark-prompt TEXT  Prompt for benchmarking");
                println!("  --benchmark-max-tokens N Max tokens (default: 64)");
                println!();
                println!("  -h, --help               Show this help");
                std::process::exit(0);
            }
            _ => i += 1,
        }
    }

    config
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config = parse_config();

    info!("=== DLM Server (Fast-dLLM v2) ===");
    info!("Model: {}", config.model_path);
    info!("Host:  {}:{}", config.host, config.port);
    info!("Max sequences: {}", config.scheduler.max_sequences);
    info!("Block size: {} (small: {})", config.dlm.block_size, config.dlm.small_block_size);
    info!("Threshold: {}", config.dlm.threshold);
    info!("Decode mode: {}", config.dlm.decode_mode);

    #[cfg(feature = "native-inference")]
    {
        use std::sync::Arc;

        // Load engine
        let batch_config = BatchConfig {
            max_sequences: config.scheduler.max_sequences as i32,
            ..Default::default()
        };

        let engine = match NativeBatchEngine::new(&config.model_path, Some(batch_config)) {
            Ok(e) => Arc::new(e),
            Err(e) => {
                error!("Failed to load model: {}", e);
                std::process::exit(1);
            }
        };

        info!("Model loaded: vocab_size={}", engine.vocab_size());

        // Detect or configure DLM
        let dlm_config = if let Some(manual_mask_id) = config.dlm.mask_token_id {
            // Manual mask token ID provided - use it directly
            info!(
                "Using manual mask_token_id={} (benchmark mode)",
                manual_mask_id
            );
            // Validate mask token is within vocab range
            let vocab_size = engine.vocab_size();
            if manual_mask_id < 0 || manual_mask_id >= vocab_size {
                error!("mask_token_id {} is out of vocab range [0, {})", manual_mask_id, vocab_size);
                std::process::exit(1);
            }
            DlmConfig::new(manual_mask_id)
                .with_block_size(config.dlm.block_size)
                .with_threshold(config.dlm.threshold)
                .with_small_block_size(config.dlm.small_block_size)
                .with_max_iterations(config.dlm.max_iterations_per_block)
                .with_decode_mode(config.dlm.parse_decode_mode())
        } else {
            // Auto-detect DLM model with diagnostics
            let detection = DlmConfig::detect_with_diagnostics(&engine);

            match detection.mask_token_id {
                Some(mask_id) => {
                    info!("============================================================");
                    info!("DLM MODEL DETECTED");
                    info!("  Mask token ID: {}", mask_id);
                    info!("  Detection method: {}", detection.detection_method);
                    info!("  Vocab size: {}", detection.vocab_size);
                    info!("============================================================");
                    DlmConfig::new(mask_id)
                        .with_block_size(config.dlm.block_size)
                        .with_threshold(config.dlm.threshold)
                        .with_small_block_size(config.dlm.small_block_size)
                        .with_max_iterations(config.dlm.max_iterations_per_block)
                        .with_decode_mode(config.dlm.parse_decode_mode())
                }
                None => {
                    // Print detailed diagnostic error
                    eprintln!("{}", detection.format_error());
                    std::process::exit(1);
                }
            }
        };

        if let Err(e) = dlm_config.validate() {
            error!("Invalid DLM config: {}", e);
            std::process::exit(1);
        }

        let scheduler_config = DlmSchedulerConfig {
            max_sequences: config.scheduler.max_sequences,
            dlm: dlm_config,
            enable_radix_cache: config.scheduler.enable_radix_cache,
            radix_cache_max_tokens: config.scheduler.radix_cache_max_tokens,
        };

        let (mut scheduler, handle) = DlmScheduler::new(scheduler_config, engine.clone());

        info!("DLM scheduler initialized");

        // Benchmark mode: run benchmark and exit
        if config.benchmark.enabled {
            info!("=== Benchmark Mode ===");
            info!("Iterations: {}", config.benchmark.iterations);
            info!("Max tokens per iteration: {}", config.benchmark.max_tokens);
            info!("Prompt: {}", config.benchmark.prompt);

            // Run scheduler in background
            let scheduler_handle_bg = tokio::spawn(async move {
                scheduler.run().await;
            });

            // Run benchmark
            run_benchmark(
                handle,
                engine,
                &config.benchmark.prompt,
                config.benchmark.iterations,
                config.benchmark.max_tokens,
            )
            .await;

            // Clean shutdown
            drop(scheduler_handle_bg);
            return;
        }

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

        let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse().unwrap();
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

/// Run benchmark with the given parameters.
#[cfg(feature = "native-inference")]
async fn run_benchmark(
    handle: DlmSchedulerHandle,
    engine: Arc<NativeBatchEngine>,
    prompt: &str,
    iterations: usize,
    max_tokens: i32,
) {
    use std::time::Instant;
    use wf_inference::inference::LatencyStats;

    info!("Starting benchmark warmup...");

    // Format prompt as chat
    let formatted_prompt = format!("USER: {}<|eot_id|>ASSISTANT:", prompt);

    // Tokenize once
    let input_ids = match engine.tokenize(&formatted_prompt, true) {
        Ok(ids) => ids,
        Err(e) => {
            error!("Failed to tokenize prompt: {}", e);
            return;
        }
    };
    let prompt_tokens = input_ids.len();

    info!("Prompt tokens: {}", prompt_tokens);

    // Warmup (3 iterations)
    for _ in 0..3 {
        let (tx, rx) = oneshot::channel();
        let req = InferenceRequest {
            request_id: 0,
            input_ids: input_ids.clone(),
            params: BatchSamplingParams::default(),
            max_tokens,
            stream: false,
            response_tx: Some(tx),
            token_tx: None,
        };
        if handle.submit(req).await.is_ok() {
            let _ = rx.await;
        }
    }

    info!("Warmup complete. Running {} iterations...", iterations);

    let mut latency_stats = LatencyStats::new(iterations);
    let mut total_tokens = 0u64;
    let overall_start = Instant::now();

    for i in 0..iterations {
        let (tx, rx) = oneshot::channel();
        let req = InferenceRequest {
            request_id: i as u64,
            input_ids: input_ids.clone(),
            params: BatchSamplingParams::default(),
            max_tokens,
            stream: false,
            response_tx: Some(tx),
            token_tx: None,
        };

        let iter_start = Instant::now();

        if let Err(e) = handle.submit(req).await {
            error!("Failed to submit request {}: {}", i, e);
            continue;
        }

        match rx.await {
            Ok(response) => {
                let duration = iter_start.elapsed();
                latency_stats.record(duration);
                total_tokens += response.completion_tokens as u64;

                if (i + 1) % 10 == 0 {
                    info!(
                        "Progress: {}/{} iterations, {} tokens, {:.2}ms avg latency",
                        i + 1,
                        iterations,
                        total_tokens,
                        latency_stats.mean() as f64 / 1000.0
                    );
                }
            }
            Err(_) => {
                error!("Request {} cancelled", i);
            }
        }
    }

    let overall_duration = overall_start.elapsed();
    let percentiles = latency_stats.percentiles_ms();

    println!();
    println!("=== Benchmark Results ===");
    println!("Iterations:        {}", iterations);
    println!("Prompt tokens:     {}", prompt_tokens);
    println!("Max tokens:        {}", max_tokens);
    println!("Total tokens:      {}", total_tokens);
    println!();
    println!("Latency (ms):");
    println!("  Mean:            {:.2}", percentiles.mean_ms);
    println!("  p50:             {:.2}", percentiles.p50_ms);
    println!("  p95:             {:.2}", percentiles.p95_ms);
    println!("  p99:             {:.2}", percentiles.p99_ms);
    println!();
    let throughput = if overall_duration.as_secs_f64() > 0.0 {
        total_tokens as f64 / overall_duration.as_secs_f64()
    } else {
        0.0
    };
    let avg_tokens_per_request = if iterations > 0 {
        total_tokens as f64 / iterations as f64
    } else {
        0.0
    };
    println!("Throughput:");
    println!("  Tokens/sec:      {:.2}", throughput);
    println!("  Avg tokens/req:  {:.2}", avg_tokens_per_request);
    println!(
        "  Total time:      {:.2}s",
        overall_duration.as_secs_f64()
    );
    println!("=========================");
}
