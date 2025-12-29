//! Batch inference router with continuous batching support.
//!
//! Provides endpoints for native batched inference using the C++ BitNet engine.

mod router;
mod sse;

pub use router::BatchRouter;
pub use sse::format_sse_event;
