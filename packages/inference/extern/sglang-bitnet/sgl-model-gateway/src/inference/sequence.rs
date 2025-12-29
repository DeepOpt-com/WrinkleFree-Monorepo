//! Sequence state management for continuous batching.
//!
//! Tracks the state of each sequence in the batch scheduler.

use super::batch_ffi::BitNetSeqId;
use super::batch_engine::BatchSamplingParams;
use tokio::sync::{mpsc, oneshot};

/// Token event for streaming responses
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// Generated token ID
    pub token_id: i32,
    /// Decoded token text
    pub text: String,
    /// Whether this is the final token
    pub is_finished: bool,
    /// Reason for finishing (if any)
    pub finish_reason: Option<FinishReason>,
}

/// Reason for sequence completion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// End of sequence token generated
    EOS,
    /// Maximum tokens reached
    Length,
    /// Stopped by stop sequence
    Stop,
    /// Cancelled by user
    Cancelled,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::EOS => write!(f, "stop"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Inference request submitted to the scheduler
pub struct InferenceRequest {
    /// Unique request ID
    pub request_id: u64,
    /// Input token IDs
    pub input_ids: Vec<i32>,
    /// Sampling parameters
    pub params: BatchSamplingParams,
    /// Maximum tokens to generate
    pub max_tokens: i32,
    /// Whether to stream responses
    pub stream: bool,
    /// Channel for final response (non-streaming)
    pub response_tx: Option<oneshot::Sender<InferenceResponse>>,
    /// Channel for streaming tokens
    pub token_tx: Option<mpsc::UnboundedSender<StreamToken>>,
}

/// Inference response (for non-streaming requests)
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    /// Generated token IDs
    pub output_ids: Vec<i32>,
    /// Generated text
    pub text: String,
    /// Finish reason
    pub finish_reason: FinishReason,
    /// Number of prompt tokens
    pub prompt_tokens: i32,
    /// Number of generated tokens
    pub completion_tokens: i32,
}

/// State of an active sequence
#[derive(Debug)]
pub enum SequencePhase {
    /// Waiting for a slot
    Pending,
    /// Processing prompt tokens
    Prefilling {
        /// Remaining prompt tokens to process
        remaining_tokens: Vec<i32>,
    },
    /// Generating tokens
    Decoding,
    /// Generation complete
    Finished(FinishReason),
}

/// State for an active sequence in the scheduler
pub struct SequenceState {
    /// Sequence ID from the engine
    pub seq_id: BitNetSeqId,
    /// Original request ID
    pub request_id: u64,
    /// Current phase
    pub phase: SequencePhase,
    /// Current position in sequence
    pub position: i32,
    /// Original prompt length
    pub prompt_len: i32,
    /// Original prompt tokens (for chunked prefill)
    pub prompt_tokens: Vec<i32>,
    /// Generated tokens so far
    pub generated_tokens: Vec<i32>,
    /// Sampling parameters
    pub params: BatchSamplingParams,
    /// Maximum tokens to generate
    pub max_tokens: i32,
    /// Last generated token
    pub last_token: i32,
    /// Channel for streaming tokens
    pub token_tx: Option<mpsc::UnboundedSender<StreamToken>>,
    /// Channel for final response
    pub response_tx: Option<oneshot::Sender<InferenceResponse>>,
    /// Batch index for current iteration (tracks logits position)
    pub batch_idx: Option<i32>,
}

impl SequenceState {
    /// Create a new sequence from a request
    pub fn new(
        seq_id: BitNetSeqId,
        request: InferenceRequest,
    ) -> Self {
        let prompt_len = request.input_ids.len() as i32;
        let last_token = *request.input_ids.last().unwrap_or(&0);

        Self {
            seq_id,
            request_id: request.request_id,
            phase: SequencePhase::Prefilling {
                remaining_tokens: request.input_ids.clone(),
            },
            position: 0,
            prompt_len,
            prompt_tokens: request.input_ids,
            generated_tokens: Vec::new(),
            params: request.params,
            max_tokens: request.max_tokens,
            last_token,
            token_tx: request.token_tx,
            response_tx: request.response_tx,
            batch_idx: None,
        }
    }

    /// Check if sequence is in prefill phase
    pub fn is_prefilling(&self) -> bool {
        matches!(self.phase, SequencePhase::Prefilling { .. })
    }

    /// Check if sequence is in decode phase
    pub fn is_decoding(&self) -> bool {
        matches!(self.phase, SequencePhase::Decoding)
    }

    /// Check if sequence is finished
    pub fn is_finished(&self) -> bool {
        matches!(self.phase, SequencePhase::Finished(_))
    }

    /// Get next prefill chunk (up to chunk_size tokens)
    pub fn get_prefill_chunk(&mut self, chunk_size: usize) -> Option<Vec<i32>> {
        match &mut self.phase {
            SequencePhase::Prefilling { remaining_tokens } => {
                if remaining_tokens.is_empty() {
                    return None;
                }

                let chunk_size = chunk_size.min(remaining_tokens.len());
                let chunk: Vec<i32> = remaining_tokens.drain(..chunk_size).collect();

                // If all tokens consumed, transition to decoding
                if remaining_tokens.is_empty() {
                    self.phase = SequencePhase::Decoding;
                }

                Some(chunk)
            }
            _ => None,
        }
    }

    /// Record a generated token
    pub fn add_generated_token(&mut self, token_id: i32, is_eos: bool) {
        self.generated_tokens.push(token_id);
        self.last_token = token_id;
        self.position += 1;

        // Check finish conditions
        let at_max = self.generated_tokens.len() >= self.max_tokens as usize;
        if is_eos {
            self.phase = SequencePhase::Finished(FinishReason::EOS);
        } else if at_max {
            self.phase = SequencePhase::Finished(FinishReason::Length);
        }
    }

    /// Send a streaming token event
    pub fn send_token(&self, token_id: i32, text: String, is_finished: bool) {
        if let Some(tx) = &self.token_tx {
            let finish_reason = if is_finished {
                match &self.phase {
                    SequencePhase::Finished(reason) => Some(*reason),
                    _ => None,
                }
            } else {
                None
            };

            let _ = tx.send(StreamToken {
                token_id,
                text,
                is_finished,
                finish_reason,
            });
        }
    }

    /// Complete the sequence and send final response
    pub fn complete(&mut self, text: String) {
        let finish_reason = match &self.phase {
            SequencePhase::Finished(reason) => *reason,
            _ => FinishReason::Length,
        };

        // Send final response
        if let Some(tx) = self.response_tx.take() {
            let _ = tx.send(InferenceResponse {
                output_ids: self.generated_tokens.clone(),
                text,
                finish_reason,
                prompt_tokens: self.prompt_len,
                completion_tokens: self.generated_tokens.len() as i32,
            });
        }

        // Send final streaming token if streaming
        if let Some(tx) = &self.token_tx {
            let _ = tx.send(StreamToken {
                token_id: self.last_token,
                text: String::new(),
                is_finished: true,
                finish_reason: Some(finish_reason),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefill_chunking() {
        let (tx, _rx) = oneshot::channel();
        let request = InferenceRequest {
            request_id: 1,
            input_ids: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            params: BatchSamplingParams::default(),
            max_tokens: 100,
            stream: false,
            response_tx: Some(tx),
            token_tx: None,
        };

        let mut state = SequenceState::new(0, request);
        assert!(state.is_prefilling());

        // First chunk of 3
        let chunk = state.get_prefill_chunk(3);
        assert_eq!(chunk, Some(vec![1, 2, 3]));
        assert!(state.is_prefilling());

        // Second chunk of 3
        let chunk = state.get_prefill_chunk(3);
        assert_eq!(chunk, Some(vec![4, 5, 6]));
        assert!(state.is_prefilling());

        // Third chunk of 3
        let chunk = state.get_prefill_chunk(3);
        assert_eq!(chunk, Some(vec![7, 8, 9]));
        assert!(state.is_prefilling());

        // Final chunk (only 1 token left)
        let chunk = state.get_prefill_chunk(3);
        assert_eq!(chunk, Some(vec![10]));
        assert!(state.is_decoding()); // Transitioned to decode

        // No more chunks
        let chunk = state.get_prefill_chunk(3);
        assert_eq!(chunk, None);
    }

    #[test]
    fn test_token_generation() {
        let (tx, _rx) = oneshot::channel();
        let request = InferenceRequest {
            request_id: 1,
            input_ids: vec![1, 2, 3],
            params: BatchSamplingParams::default(),
            max_tokens: 5,
            stream: false,
            response_tx: Some(tx),
            token_tx: None,
        };

        let mut state = SequenceState::new(0, request);

        // Exhaust prefill
        state.get_prefill_chunk(100);
        assert!(state.is_decoding());

        // Generate tokens
        state.add_generated_token(10, false);
        state.add_generated_token(11, false);
        state.add_generated_token(12, false);
        state.add_generated_token(13, false);
        state.add_generated_token(14, false); // Max reached

        assert!(state.is_finished());
        assert_eq!(state.generated_tokens.len(), 5);
    }
}
