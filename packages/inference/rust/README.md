# WrinkleFree Inference Engine (Rust)

Native Rust inference engine for BitNet 1.58-bit quantized LLMs.

## Binaries

### wf_server (Pure Rust)
SIMD-optimized BitNet inference with zero C++ dependencies.

```bash
cargo build --release --bin wf_server --features native-inference
./target/release/wf_server --model-path model.gguf --port 30000
```

### dlm_server (DLM Block Diffusion)
Fast-dLLM v2 server for ~2.5x faster inference via parallel block decoding.
Requires llama.cpp (see `scripts/setup_llama_cpp.sh`).

```bash
# Build llama.cpp first
../scripts/setup_llama_cpp.sh

# Build dlm_server
cargo build --release --bin dlm_server --features llama-inference

# Run with library path
export LD_LIBRARY_PATH="../extern/llama.cpp/build/src:../extern/llama.cpp/build/ggml/src"
./target/release/dlm_server --model-path model.gguf --port 30000
```

## Features

| Feature | Description |
|---------|-------------|
| `native-inference` | Pure Rust BitNet SIMD kernels (wf_server) |
| `llama-inference` | llama.cpp-based inference (dlm_server) |

## Architecture

```
src/
├── bin/
│   ├── wf_server.rs       # Pure Rust server
│   └── dlm_server.rs      # DLM block diffusion server
├── engine/                # Native transformer implementation
├── gguf/                  # Pure Rust GGUF reader
├── kernels/bitnet/        # SIMD-optimized ternary kernels
└── inference/             # DLM scheduler + C++ FFI
```
