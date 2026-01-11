# BitNet Native Inference Guide

> **ARCHIVED**: This document described an alternative inference path using sgl-kernel binary format.
> That approach has been removed. The current approach uses GGUF format with the Rust inference engine.

## Current Approach

Use `wf_server` (pure Rust) or `dlm_server` (with llama.cpp for DLM block diffusion):

```bash
# Build
cd packages/inference/rust
cargo build --release --bin wf_server --features native-inference

# Run
./target/release/wf_server --model-path model.gguf --port 30000
```

## Documentation

- [Getting Started](getting-started.md) - Quick start guide
- [DLM Pipeline](dlm-pipeline.md) - Block diffusion inference
- [GGUF Conversion](gguf-conversion.md) - Converting checkpoints
- [CLAUDE.md](../CLAUDE.md) - Full package documentation

## Architecture

```
packages/inference/
├── rust/                    # Rust inference engine
│   └── src/
│       ├── bin/
│       │   ├── wf_server.rs     # Pure Rust BitNet server
│       │   └── dlm_server.rs    # DLM block diffusion server
│       ├── engine/              # Transformer implementation
│       ├── gguf/                # GGUF reader
│       ├── kernels/bitnet/      # SIMD kernels
│       └── inference/           # DLM scheduler
├── cpp/                     # C++ wrappers for llama.cpp FFI
└── extern/
    └── llama.cpp/           # Downloaded on-demand
```

## Performance

| Server | Speed | Notes |
|--------|-------|-------|
| wf_server | ~26 tok/s | Pure Rust, no dependencies |
| dlm_server | ~60 tok/s | DLM block diffusion, requires llama.cpp |
