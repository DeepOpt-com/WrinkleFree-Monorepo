//! WrinkleFree Inference Server
//!
//! Native BitNet inference server with GGUF support and DLM block diffusion.
//! Uses native ternary SIMD kernels for maximum inference speed.
//!
//! Usage:
//!   cargo run --release --features native-inference --bin wf_server -- \
//!     --model-path /path/to/model.gguf --port 30000
//!
//! The server implements OpenAI-compatible API at:
//!   POST /v1/chat/completions

// Use mimalloc as the global allocator for better performance
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

#[cfg(feature = "native-inference")]
use sgl_model_gateway::engine::{BitNetEngine, SamplingConfig};

#[cfg(feature = "native-inference")]
use sgl_model_gateway::gguf::GgufReader;

/// WrinkleFree Inference Server
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model_path: String,

    /// Server host
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Server port
    #[arg(short, long, default_value_t = 30000)]
    port: u16,

    /// Number of threads for inference
    #[arg(short, long, default_value_t = 0)]
    threads: usize,

    /// Run benchmark mode (no server)
    #[arg(long)]
    benchmark: bool,

    /// Number of benchmark iterations
    #[arg(long, default_value_t = 20)]
    benchmark_iterations: usize,

    /// Max tokens for benchmark
    #[arg(long, default_value_t = 64)]
    benchmark_max_tokens: usize,

    /// Prompt for benchmark
    #[arg(long, default_value = "What is the meaning of life?")]
    benchmark_prompt: String,
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
    engine: Arc<Mutex<BitNetEngine>>,
    request_counter: AtomicU64,
    model_name: String,
}

async fn health() -> &'static str {
    "ok"
}

#[cfg(feature = "native-inference")]
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    let start = Instant::now();

    // Build prompt from messages
    let mut prompt = String::new();
    for msg in &req.messages {
        prompt.push_str(&format!("{}: {}\n", msg.role.to_uppercase(), msg.content));
    }
    prompt.push_str("ASSISTANT:");

    // Simple tokenization (placeholder - real impl uses proper tokenizer)
    let input_ids: Vec<i32> = prompt
        .chars()
        .filter(|c| !c.is_whitespace())
        .take(512)
        .enumerate()
        .map(|(i, _)| i as i32)
        .collect();

    let prompt_tokens = input_ids.len() as i32;

    // Configure sampling
    let sampling_config = SamplingConfig {
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        ..Default::default()
    };

    // Generate
    let mut engine = state.engine.lock().await;
    engine.reset();
    let output_ids = engine.generate(&input_ids, req.max_tokens as usize, &sampling_config);
    let completion_tokens = output_ids.len() as i32;

    // Decode tokens (placeholder)
    let output_text = format!("[Generated {} tokens]", completion_tokens);

    let elapsed = start.elapsed();
    let tokens_per_sec = if elapsed.as_secs_f32() > 0.0 {
        completion_tokens as f32 / elapsed.as_secs_f32()
    } else {
        0.0
    };

    info!(
        "Generated {} tokens in {:.2}ms ({:.1} tok/s)",
        completion_tokens,
        elapsed.as_millis(),
        tokens_per_sec
    );

    let request_id = state.request_counter.fetch_add(1, Ordering::Relaxed);
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", request_id),
        object: "chat.completion".to_string(),
        created,
        model: state.model_name.clone(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: output_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: ChatCompletionUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// Run benchmark mode.
#[cfg(feature = "native-inference")]
async fn run_benchmark(
    mut engine: BitNetEngine,
    prompt: &str,
    iterations: usize,
    max_tokens: usize,
) {
    info!("=== Benchmark Mode ===");
    info!("Prompt: {}", prompt);
    info!("Iterations: {}", iterations);
    info!("Max tokens: {}", max_tokens);

    // Simple tokenization (placeholder)
    let input_ids: Vec<i32> = (0..prompt.len().min(256))
        .map(|i| i as i32)
        .collect();

    let prompt_tokens = input_ids.len();
    info!("Prompt tokens: {}", prompt_tokens);

    let sampling_config = SamplingConfig {
        temperature: 0.0, // Greedy for deterministic benchmark
        ..Default::default()
    };

    // Warmup (3 iterations)
    info!("Warmup...");
    for _ in 0..3 {
        engine.reset();
        let _ = engine.generate(&input_ids, max_tokens, &sampling_config);
    }

    // Benchmark
    info!("Running benchmark...");
    let mut prefill_times = Vec::with_capacity(iterations);
    let mut decode_times = Vec::with_capacity(iterations);
    let mut total_tokens = 0u64;

    let overall_start = Instant::now();

    for i in 0..iterations {
        engine.reset();

        // Measure prefill
        let prefill_start = Instant::now();
        let logits = engine.forward_prefill(&input_ids, 0);
        let prefill_time = prefill_start.elapsed();
        prefill_times.push(prefill_time.as_micros() as u64);

        // Measure decode
        let decode_start = Instant::now();
        let output_ids = engine.generate(&input_ids, max_tokens, &sampling_config);
        let decode_time = decode_start.elapsed() - prefill_time;
        decode_times.push(decode_time.as_micros() as u64);

        total_tokens += output_ids.len() as u64;

        if (i + 1) % 5 == 0 {
            info!("Progress: {}/{}", i + 1, iterations);
        }
    }

    let overall_duration = overall_start.elapsed();

    // Calculate statistics
    let avg_prefill_us = prefill_times.iter().sum::<u64>() / iterations as u64;
    let avg_decode_us = decode_times.iter().sum::<u64>() / iterations as u64;
    let prefill_tok_s = prompt_tokens as f64 / (avg_prefill_us as f64 / 1_000_000.0);
    let decode_tok_s = (max_tokens as f64) / (avg_decode_us as f64 / 1_000_000.0);
    let overall_tok_s = total_tokens as f64 / overall_duration.as_secs_f64();

    println!();
    println!("=== Benchmark Results ===");
    println!("Iterations:        {}", iterations);
    println!("Prompt tokens:     {}", prompt_tokens);
    println!("Max tokens:        {}", max_tokens);
    println!("Total tokens:      {}", total_tokens);
    println!();
    println!("Prefill:");
    println!("  Avg time:        {:.2} ms", avg_prefill_us as f64 / 1000.0);
    println!("  Throughput:      {:.1} tok/s", prefill_tok_s);
    println!();
    println!("Decode:");
    println!("  Avg time:        {:.2} ms", avg_decode_us as f64 / 1000.0);
    println!("  Throughput:      {:.1} tok/s", decode_tok_s);
    println!();
    println!("Overall:");
    println!("  Total time:      {:.2} s", overall_duration.as_secs_f64());
    println!("  Throughput:      {:.1} tok/s", overall_tok_s);
    println!("=========================");
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("=== WrinkleFree Inference Server ===");
    info!("Model: {}", args.model_path);

    #[cfg(feature = "native-inference")]
    {
        // Load model
        info!("Loading model...");
        let engine = match BitNetEngine::load(&args.model_path) {
            Ok(e) => e,
            Err(e) => {
                error!("Failed to load model: {}", e);
                std::process::exit(1);
            }
        };

        info!("Model loaded:");
        info!("  Vocab size: {}", engine.config.vocab_size);
        info!("  Hidden size: {}", engine.config.hidden_size);
        info!("  Num layers: {}", engine.config.num_layers);
        info!("  Num heads: {}", engine.config.num_heads);

        // Benchmark mode
        if args.benchmark {
            run_benchmark(
                engine,
                &args.benchmark_prompt,
                args.benchmark_iterations,
                args.benchmark_max_tokens,
            )
            .await;
            return;
        }

        // Get model name
        let model_name = Path::new(&args.model_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "wf-model".to_string());

        // Create app state
        let app_state = Arc::new(AppState {
            engine: Arc::new(Mutex::new(engine)),
            request_counter: AtomicU64::new(0),
            model_name,
        });

        // Create router
        let app = Router::new()
            .route("/health", get(health))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(app_state);

        let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse().unwrap();
        info!("Server listening on http://{}", addr);
        info!("API endpoint: http://{}/v1/chat/completions", addr);

        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    }

    #[cfg(not(feature = "native-inference"))]
    {
        error!("Native inference not enabled. Build with --features native-inference");
        std::process::exit(1);
    }
}
