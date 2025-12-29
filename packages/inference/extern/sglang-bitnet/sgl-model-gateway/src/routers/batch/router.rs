//! Batch inference router with continuous batching support.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::{
    body::Body,
    extract::State,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response, Sse},
    routing::post,
    Json, Router,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info};

use super::sse::{format_sse_done, format_sse_error, format_sse_event};
use crate::inference::{
    BatchConfig, BatchSamplingParams, BatchScheduler, InferenceRequest, InferenceResponse,
    NativeBatchEngine, SchedulerConfig, SchedulerHandle, SchedulerStats, StreamToken,
};

/// Request ID generator
static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn generate_request_id() -> u64 {
    REQUEST_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Chat completion request (simplified OpenAI-compatible format)
#[derive(Debug, Clone, Deserialize)]
pub struct BatchChatRequest {
    /// Messages (simplified - just concatenate content)
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i32,
    /// Temperature for sampling
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-p sampling
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> i32 {
    256
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat completion response
#[derive(Debug, Clone, Serialize)]
pub struct BatchChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: i32,
    pub message: ChatMessageResponse,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessageResponse {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

/// Scheduler stats response
#[derive(Debug, Clone, Serialize)]
pub struct StatsResponse {
    pub active_sequences: usize,
    pub pending_requests: usize,
    pub prefilling_sequences: usize,
    pub decoding_sequences: usize,
    pub kv_cache_used: usize,
    pub kv_cache_capacity: usize,
    pub kv_cache_utilization: f32,
}

impl From<SchedulerStats> for StatsResponse {
    fn from(stats: SchedulerStats) -> Self {
        let utilization = if stats.kv_cache_capacity > 0 {
            stats.kv_cache_used as f32 / stats.kv_cache_capacity as f32
        } else {
            0.0
        };
        Self {
            active_sequences: stats.active_sequences,
            pending_requests: stats.pending_requests,
            prefilling_sequences: stats.prefilling_sequences,
            decoding_sequences: stats.decoding_sequences,
            kv_cache_used: stats.kv_cache_used,
            kv_cache_capacity: stats.kv_cache_capacity,
            kv_cache_utilization: utilization,
        }
    }
}

/// Batch router state
pub struct BatchRouterState {
    pub scheduler_handle: SchedulerHandle,
    pub engine: Arc<NativeBatchEngine>,
    pub model_name: String,
}

/// Batch router for continuous batching
pub struct BatchRouter {
    engine: Arc<NativeBatchEngine>,
    scheduler_config: SchedulerConfig,
    model_name: String,
}

impl BatchRouter {
    /// Create a new batch router
    pub fn new(
        model_path: &str,
        batch_config: Option<BatchConfig>,
        scheduler_config: Option<SchedulerConfig>,
        model_name: Option<String>,
    ) -> Result<Self, crate::inference::BatchError> {
        let engine = NativeBatchEngine::new(model_path, batch_config)?;
        let scheduler_config = scheduler_config.unwrap_or_default();
        let model_name = model_name.unwrap_or_else(|| "bitnet-b1.58-2B-4T".to_string());

        Ok(Self {
            engine: Arc::new(engine),
            scheduler_config,
            model_name,
        })
    }

    /// Start the scheduler and return an Axum router
    pub fn into_router(self) -> (Router, tokio::task::JoinHandle<()>) {
        let engine = self.engine.clone();
        let config = self.scheduler_config.clone();
        let model_name = self.model_name.clone();

        // Create scheduler
        let (mut scheduler, handle) = BatchScheduler::new(config, engine.clone());

        // Start scheduler task
        let scheduler_handle = tokio::spawn(async move {
            scheduler.run().await;
        });

        // Create router state with engine reference for tokenization
        let state = Arc::new(BatchRouterState {
            scheduler_handle: handle,
            engine,
            model_name,
        });

        // Build Axum router
        let router = Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/v1/batch/stats", post(handle_stats))
            .with_state(state);

        (router, scheduler_handle)
    }
}

/// Format messages into a prompt using BitNet chat template
/// Template: "Role: content<|eot_id|>"
fn format_messages(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| {
            let role = match m.role.to_lowercase().as_str() {
                "system" => "System",
                "user" => "User",
                "assistant" => "Assistant",
                _ => &m.role,
            };
            format!("{}: {}<|eot_id|>", role, m.content)
        })
        .collect::<Vec<_>>()
        .join("")
        + "Assistant:"
}

/// Handle chat completions
async fn handle_chat_completions(
    State(state): State<Arc<BatchRouterState>>,
    Json(request): Json<BatchChatRequest>,
) -> Response {
    let request_id = generate_request_id();

    // Format prompt using chat template
    let prompt = format_messages(&request.messages);

    // Tokenize using the engine's tokenizer (llama.cpp)
    let input_ids = match state.engine.tokenize(&prompt, true) {
        Ok(ids) => ids,
        Err(e) => {
            error!("Tokenization failed: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Tokenization failed: {}", e),
                        "type": "invalid_request_error"
                    }
                })),
            )
                .into_response();
        }
    };

    debug!(
        "Request {}: {} input tokens, stream={}",
        request_id,
        input_ids.len(),
        request.stream
    );

    // Create sampling params
    let params = BatchSamplingParams {
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: 0.0,
        repetition_penalty: 1.0,
        max_tokens: request.max_tokens,
    };

    if request.stream {
        // Streaming response
        handle_streaming_request(state, request_id, input_ids, params, request.max_tokens).await
    } else {
        // Non-streaming response
        handle_non_streaming_request(state, request_id, input_ids, params, request.max_tokens).await
    }
}

async fn handle_streaming_request(
    state: Arc<BatchRouterState>,
    request_id: u64,
    input_ids: Vec<i32>,
    params: BatchSamplingParams,
    max_tokens: i32,
) -> Response {
    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (response_tx, _) = oneshot::channel();

    let inference_request = InferenceRequest {
        request_id,
        input_ids,
        params,
        max_tokens,
        stream: true,
        response_tx: Some(response_tx),
        token_tx: Some(token_tx),
    };

    // Submit to scheduler
    if let Err(e) = state.scheduler_handle.submit(inference_request).await {
        error!("Failed to submit request: {}", e);
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "message": format!("Scheduler unavailable: {}", e),
                    "type": "server_error"
                }
            })),
        )
            .into_response();
    }

    // Create SSE stream
    let model_name = state.model_name.clone();
    let request_id_str = format!("{}", request_id);

    let stream = UnboundedReceiverStream::new(token_rx).map(move |token| {
        let event = format_sse_event(&token, &request_id_str, &model_name);
        Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(event))
    });

    // Add [DONE] at the end
    let stream = stream.chain(futures_util::stream::once(async {
        Ok(axum::response::sse::Event::default().data(format_sse_done()))
    }));

    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

async fn handle_non_streaming_request(
    state: Arc<BatchRouterState>,
    request_id: u64,
    input_ids: Vec<i32>,
    params: BatchSamplingParams,
    max_tokens: i32,
) -> Response {
    let (response_tx, response_rx) = oneshot::channel();

    let inference_request = InferenceRequest {
        request_id,
        input_ids: input_ids.clone(),
        params,
        max_tokens,
        stream: false,
        response_tx: Some(response_tx),
        token_tx: None,
    };

    // Submit to scheduler
    if let Err(e) = state.scheduler_handle.submit(inference_request).await {
        error!("Failed to submit request: {}", e);
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "message": format!("Scheduler unavailable: {}", e),
                    "type": "server_error"
                }
            })),
        )
            .into_response();
    }

    // Wait for response
    match response_rx.await {
        Ok(response) => {
            let chat_response = BatchChatResponse {
                id: format!("chatcmpl-{}", request_id),
                object: "chat.completion".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                model: state.model_name.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessageResponse {
                        role: "assistant".to_string(),
                        content: response.text,
                    },
                    finish_reason: response.finish_reason.to_string(),
                }],
                usage: ChatUsage {
                    prompt_tokens: response.prompt_tokens,
                    completion_tokens: response.completion_tokens,
                    total_tokens: response.prompt_tokens + response.completion_tokens,
                },
            };

            Json(chat_response).into_response()
        }
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": "Request was cancelled",
                    "type": "server_error"
                }
            })),
        )
            .into_response(),
    }
}

/// Handle stats request
async fn handle_stats(State(state): State<Arc<BatchRouterState>>) -> Response {
    // TODO: Get stats from scheduler
    // For now, return placeholder
    let stats = StatsResponse {
        active_sequences: 0,
        pending_requests: 0,
        prefilling_sequences: 0,
        decoding_sequences: 0,
        kv_cache_used: 0,
        kv_cache_capacity: 0,
        kv_cache_utilization: 0.0,
    };

    Json(stats).into_response()
}
