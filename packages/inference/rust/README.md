# WrinkleFree Inference Engine (Rust)

Native Rust inference engine for BitNet 1.58-bit quantized LLMs.

## Binaries

### wf_server (Pure Rust)
SIMD-optimized BitNet inference with zero C++ dependencies.

```bash
cargo build --release --bin wf_server --features native-inference
./target/release/wf_server --model-path model.gguf --port 30000
```

## Features

| Feature | Description |
|---------|-------------|
| `native-inference` | Pure Rust BitNet SIMD kernels (wf_server) |
| `llama-inference` | llama.cpp-based inference (legacy) |

## Architecture

```
src/
├── bin/
│   └── wf_server.rs       # Pure Rust server
├── engine/                # Native transformer implementation
├── gguf/                  # Pure Rust GGUF reader
├── kernels/bitnet/        # SIMD-optimized ternary kernels
└── inference/             # Scheduler + C++ FFI (legacy)
```
