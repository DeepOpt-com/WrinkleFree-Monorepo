//! Batch server with continuous batching and RadixAttention prefix caching.
//!
//! This server uses the BatchScheduler with RadixAttention to enable:
//! - Continuous batching of multiple concurrent requests
//! - Prefix caching for efficient long sequence handling
//!
//! Usage:
//!   cargo run --release --features native-inference --bin batch_server -- \
//!     --model-path /path/to/model.gguf --port 30000

use std::net::SocketAddr;

use axum::{routing::get, Router};
use tracing::{error, info};

#[cfg(feature = "native-inference")]
use sgl_model_gateway::routers::BatchRouter;
#[cfg(feature = "native-inference")]
use sgl_model_gateway::inference::{BatchConfig, SchedulerConfig};

async fn health() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    // Parse args
    let mut model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "microsoft/bitnet-b1.58-2B-4T".to_string());
    let mut port: u16 = 30000;
    let mut host = String::from("0.0.0.0");
    let mut max_sequences: usize = 16;
    let mut enable_radix_cache = true;

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
                max_sequences = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(max_sequences);
                i += 2;
            }
            "--no-radix-cache" => {
                enable_radix_cache = false;
                i += 1;
            }
            _ => i += 1,
        }
    }

    info!("=== Batch BitNet Server with RadixAttention ===");
    info!("Model: {}", model_path);
    info!("Host:  {}:{}", host, port);
    info!("Max sequences: {}", max_sequences);
    info!("RadixCache enabled: {}", enable_radix_cache);

    #[cfg(feature = "native-inference")]
    {
        let batch_config = BatchConfig {
            max_sequences: max_sequences as i32,
            ..Default::default()
        };

        let scheduler_config = SchedulerConfig {
            max_sequences,
            enable_radix_cache,
            radix_cache_max_tokens: 100_000,
            ..Default::default()
        };

        match BatchRouter::new(
            &model_path,
            Some(batch_config),
            Some(scheduler_config),
            None,
        ) {
            Ok(router) => {
                info!("Engine loaded successfully");

                let (api_router, scheduler_handle) = router.into_router();

                let app = Router::new()
                    .route("/health", get(health))
                    .merge(api_router);

                let addr: SocketAddr = format!("{}:{}", host, port).parse().unwrap();
                info!("Server listening on http://{}", addr);

                let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

                // Run server and scheduler concurrently
                tokio::select! {
                    _ = axum::serve(listener, app) => {
                        info!("Server shut down");
                    }
                    _ = scheduler_handle => {
                        info!("Scheduler shut down");
                    }
                }
            }
            Err(e) => {
                error!("Failed to create batch router: {}", e);
                std::process::exit(1);
            }
        }
    }

    #[cfg(not(feature = "native-inference"))]
    {
        error!("Native inference not enabled. Build with --features native-inference");
        std::process::exit(1);
    }
}
