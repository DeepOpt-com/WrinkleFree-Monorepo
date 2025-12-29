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

use axum::{routing::get, Router};
use tracing::{error, info};

#[cfg(feature = "native-inference")]
use sgl_model_gateway::inference::{
    BatchConfig, DlmConfig, DlmScheduler, DlmSchedulerConfig, NativeBatchEngine,
};
#[cfg(feature = "native-inference")]
use sgl_model_gateway::routers::BatchRouter;

async fn health() -> &'static str {
    "ok"
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
        let dlm_config = match DlmConfig::detect(&engine) {
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
                error!("Ensure the model was trained with Fast-dLLM v2 and has |<MASK>| token");
                std::process::exit(1);
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

        let (mut scheduler, handle) = DlmScheduler::new(scheduler_config, engine);

        info!("DLM scheduler initialized");

        // Create router using BatchRouter with DLM scheduler
        // Note: We reuse BatchRouter's HTTP API but with DLM scheduling
        // For a production system, you'd want a dedicated DlmRouter
        let app = Router::new().route("/health", get(health));

        // TODO: Add DLM-specific routes:
        // - POST /v1/chat/completions (with DLM scheduling)
        // - GET /v1/models
        // - GET /stats (DLM-specific stats)

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
