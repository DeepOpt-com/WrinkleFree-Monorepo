//! Minimal HTTP server for native BitNet inference.
//!
//! Provides OpenAI-compatible `/v1/chat/completions` endpoint using
//! the native C++ inference engine.
//!
//! Usage:
//!   cargo run --release --features native-inference --bin native_server -- \
//!     --model-path /path/to/model.gguf --port 30000

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio::sync::Mutex;
use tracing::{error, info};

#[cfg(feature = "native-inference")]
use sgl_model_gateway::grpc_client::native_engine::NativeEngineClient;
#[cfg(feature = "native-inference")]
use sgl_model_gateway::protocols::sampling_params::SamplingParams;

#[derive(Clone)]
struct AppState {
    #[cfg(feature = "native-inference")]
    engine: Arc<Mutex<NativeEngineClient>>,
    model_id: String,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    max_tokens: Option<i32>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: i32,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

impl Serialize for ChatMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("ChatMessage", 2)?;
        state.serialize_field("role", &self.role)?;
        state.serialize_field("content", &self.content)?;
        state.end()
    }
}

// Streaming response structures
#[derive(Debug, Serialize)]
struct StreamChoice {
    index: i32,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct StreamResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelInfo>,
}

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model".to_string(),
            owned_by: "local".to_string(),
        }],
    })
}

#[cfg(feature = "native-inference")]
async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Build prompt from messages
    let mut prompt = String::new();
    for msg in &req.messages {
        prompt.push_str(&format!("{}: {}<|eot_id|>", msg.role, msg.content));
    }
    prompt.push_str("assistant:");

    let params = SamplingParams {
        temperature: req.temperature.map(|t| t as f32),
        top_p: req.top_p.map(|t| t as f32),
        max_new_tokens: req.max_tokens.map(|t| t as u32),
        ..Default::default()
    };

    let is_streaming = req.stream.unwrap_or(false);
    let max_tokens = req.max_tokens.unwrap_or(256) as usize;

    if is_streaming {
        // True token-by-token streaming
        let engine = state.engine.clone();
        let model_id = state.model_id.clone();

        let stream = async_stream::stream! {
            let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
            let created = chrono::Utc::now().timestamp();

            // First event with role
            let first = StreamResponse {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: Delta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            yield Ok::<_, Infallible>(Event::default().data(serde_json::to_string(&first).unwrap()));

            // Get engine lock and tokenize
            let engine_guard = engine.lock().await;

            // Encode prompt to tokens
            let input_ids = match engine_guard.encode(&prompt).await {
                Ok(ids) => ids,
                Err(e) => {
                    error!("Tokenization failed: {}", e);
                    let error_chunk = StreamResponse {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(format!("[Error: {}]", e)),
                            },
                            finish_reason: Some("error".to_string()),
                        }],
                    };
                    yield Ok(Event::default().data(serde_json::to_string(&error_chunk).unwrap()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            };

            let num_prompt_tokens = input_ids.len();
            info!("Prefilling {} prompt tokens", num_prompt_tokens);

            // Get EOS token ID
            let eos_token_id = engine_guard.eos_token_id().await.unwrap_or(128009); // BitNet uses 128009 for <|eot_id|>

            // Reset cache for new sequence
            engine_guard.reset_cache().await;

            // Prefill phase
            if let Err(e) = engine_guard.prefill(&input_ids).await {
                error!("Prefill failed: {}", e);
                let error_chunk = StreamResponse {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(format!("[Prefill error: {}]", e)),
                        },
                        finish_reason: Some("error".to_string()),
                    }],
                };
                yield Ok(Event::default().data(serde_json::to_string(&error_chunk).unwrap()));
                yield Ok(Event::default().data("[DONE]"));
                return;
            }

            // Token-by-token decode
            let mut generated_tokens: Vec<i32> = Vec::new();
            let mut prev_text_len = 0;

            for i in 0..max_tokens {
                let position = (num_prompt_tokens + i) as i32;

                // Generate one token
                let token_id = match engine_guard.decode_step(position, &params).await {
                    Ok(id) => id,
                    Err(e) => {
                        error!("Decode step failed at position {}: {}", position, e);
                        break;
                    }
                };

                // Check for EOS
                if token_id == eos_token_id {
                    info!("EOS token generated after {} tokens", i);
                    break;
                }

                generated_tokens.push(token_id);

                // Decode all tokens so far and get the new text
                let full_text = match engine_guard.decode(&generated_tokens).await {
                    Ok(text) => text,
                    Err(e) => {
                        error!("Token decode failed: {}", e);
                        continue;
                    }
                };

                // Get only the new part (handles BPE correctly)
                let new_text = &full_text[prev_text_len..];
                prev_text_len = full_text.len();

                if !new_text.is_empty() {
                    let chunk = StreamResponse {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(new_text.to_string()),
                            },
                            finish_reason: None,
                        }],
                    };
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                }
            }

            info!("Generated {} tokens", generated_tokens.len());

            // Final event
            let final_chunk = StreamResponse {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            yield Ok(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));

            // [DONE] marker
            yield Ok(Event::default().data("[DONE]"));
        };

        Sse::new(stream).into_response()
    } else {
        // Non-streaming response - use batch generation
        let engine = state.engine.lock().await;
        match engine.generate(&prompt, &params).await {
            Ok(result) => {
                let response = ChatCompletionResponse {
                    id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    object: "chat.completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: state.model_id.clone(),
                    choices: vec![Choice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: result.text,
                        },
                        finish_reason: "stop".to_string(),
                    }],
                    usage: Usage {
                        prompt_tokens: result.num_prompt_tokens as i32,
                        completion_tokens: result.num_generated_tokens as i32,
                        total_tokens: (result.num_prompt_tokens + result.num_generated_tokens) as i32,
                    },
                };
                Json(response).into_response()
            }
            Err(e) => {
                error!("Generation failed: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
            }
        }
    }
}

#[cfg(not(feature = "native-inference"))]
async fn chat_completions() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        "Native inference not enabled. Build with --features native-inference",
    )
        .into_response()
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    // Parse args
    let mut model_path = String::from("/home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine/extern/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf");
    let mut port: u16 = 30000;
    let mut host = String::from("0.0.0.0");

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
            _ => i += 1,
        }
    }

    info!("=== Native BitNet Server ===");
    info!("Model: {}", model_path);
    info!("Host:  {}:{}", host, port);

    #[cfg(feature = "native-inference")]
    let state = {
        match NativeEngineClient::new(&model_path).await {
            Ok(engine) => {
                info!("Engine loaded successfully");
                AppState {
                    engine: Arc::new(Mutex::new(engine)),
                    model_id: "bitnet-b1.58-2B-4T".to_string(),
                }
            }
            Err(e) => {
                error!("Failed to create engine: {}", e);
                std::process::exit(1);
            }
        }
    };

    #[cfg(not(feature = "native-inference"))]
    let state = {
        error!("Native inference not enabled");
        std::process::exit(1);
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", host, port).parse().unwrap();
    info!("Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
