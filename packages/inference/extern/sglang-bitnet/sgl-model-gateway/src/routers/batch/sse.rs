//! SSE (Server-Sent Events) formatting utilities.

use crate::inference::{FinishReason, StreamToken};
use serde_json::json;

/// Format a StreamToken as an SSE event (OpenAI-compatible format)
pub fn format_sse_event(token: &StreamToken, request_id: &str, model: &str) -> String {
    let finish_reason = token.finish_reason.map(|r| match r {
        FinishReason::EOS => "stop",
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
        FinishReason::Cancelled => "cancelled",
    });

    let chunk = json!({
        "id": format!("chatcmpl-{}", request_id),
        "object": "chat.completion.chunk",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "content": if token.is_finished { "" } else { &token.text }
            },
            "finish_reason": finish_reason
        }]
    });

    format!("data: {}\n\n", chunk)
}

/// Format the final [DONE] SSE event
pub fn format_sse_done() -> String {
    "data: [DONE]\n\n".to_string()
}

/// Format an SSE error event
pub fn format_sse_error(error: &str) -> String {
    let error_json = json!({
        "error": {
            "message": error,
            "type": "server_error"
        }
    });
    format!("data: {}\n\n", error_json)
}
