pub mod app_context;
pub mod config;
pub mod core;
pub mod data_connector;
pub mod grpc_client;
pub mod mcp;
pub mod middleware;
pub mod multimodal;
pub mod observability;
pub mod policies;
pub mod protocols;
pub mod reasoning_parser;
pub mod routers;
pub mod server;
pub mod service_discovery;
pub mod tokenizer;
pub mod tool_parser;
pub mod version;
pub mod wasm;
pub mod workflow;

// Native inference engine (requires llama.cpp, optional)
#[cfg(feature = "llama-inference")]
pub mod inference;

// Pure Rust GGUF reader (no C++ dependency)
pub mod gguf;

// Native BitNet kernel FFI (requires C++ library, optional)
#[cfg(feature = "native-inference")]
pub mod kernels;

// Native BitNet inference engine
#[cfg(feature = "native-inference")]
pub mod engine;
