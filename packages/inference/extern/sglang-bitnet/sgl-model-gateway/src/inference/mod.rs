//! Native inference module for BitNet models.
//!
//! This module provides Rust bindings to the C++ inference engine,
//! enabling direct model inference without going through Python.

mod bitnet_ffi;
mod engine;

pub use engine::NativeEngine;
pub use engine::SamplingParams;
