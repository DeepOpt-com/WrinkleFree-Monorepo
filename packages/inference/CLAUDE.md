# WrinkleFree Inference Engine

Rust inference engine for **BitNet** 1.58-bit quantized LLMs with DLM block diffusion support.

## Quick Reference

| Task | Command |
|------|---------|
| Convert checkpoint to GGUF | `python scripts/convert_checkpoint_to_gguf.py checkpoint/ -o model.gguf` |
| Build wf_server | `cd rust && cargo build --release --bin wf_server --features native-inference` |
| Build dlm_server | `cd rust && cargo build --release --bin dlm_server --features llama-inference` |
| Setup llama.cpp | `./scripts/setup_llama_cpp.sh` |
| Benchmark wf_server | `./rust/target/release/wf_server --model-path model.gguf --benchmark` |
| Test API | `curl http://localhost:30000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hello"}]}'` |

## Binaries

### wf_server (Pure Rust)

Native BitNet inference server with SIMD-optimized ternary kernels. No C++ dependencies.

```bash
cd rust

# Build
cargo build --release --bin wf_server --features native-inference

# Run server
./target/release/wf_server --model-path model.gguf --port 30000

# Run benchmark
./target/release/wf_server --model-path model.gguf --benchmark
```

### dlm_server (DLM Block Diffusion)

Fast-dLLM v2 server for ~2.5x faster inference via parallel block decoding. Requires llama.cpp.

```bash
# Setup llama.cpp first
./scripts/setup_llama_cpp.sh

# Build dlm_server
cd rust
cargo build --release --bin dlm_server --features llama-inference

# Set library path and run
export LD_LIBRARY_PATH="../extern/llama.cpp/build/src:../extern/llama.cpp/build/ggml/src"
./target/release/dlm_server --model-path model.gguf --port 30000
```

## GGUF Conversion

```bash
# Convert checkpoint to GGUF (I2_S recommended for production)
python scripts/convert_checkpoint_to_gguf.py /path/to/checkpoint --outfile model.gguf

# For larger models
python scripts/convert_checkpoint_to_gguf.py /path/to/checkpoint --outfile model.gguf --outtype i2_s
```

**CRITICAL**: Never use TQ2_0 for bf16 DLM checkpoints - it corrupts weights!

| Format | Notes |
|--------|-------|
| **I2_S** | Recommended - fastest, best compatibility |
| F16 | Default, works for all models |
| TQ1_0 | Requires hidden_size % 256 == 0 |
| TQ2_0 | DO NOT USE for bf16 checkpoints |

## API

Both servers expose OpenAI-compatible API:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## Architecture

```
packages/inference/
├── rust/                          # Rust inference engine
│   └── src/
│       ├── bin/
│       │   ├── wf_server.rs       # Pure Rust BitNet server
│       │   └── dlm_server.rs      # DLM block diffusion server
│       ├── engine/                # Transformer forward pass
│       ├── gguf/                  # Pure Rust GGUF reader
│       ├── kernels/bitnet/        # SIMD ternary kernels
│       └── inference/             # DLM scheduler
├── cpp/                           # C++ wrappers for llama.cpp FFI
├── extern/
│   └── llama.cpp/                 # Downloaded on-demand
├── scripts/
│   ├── convert_checkpoint_to_gguf.py
│   └── setup_llama_cpp.sh
├── src/wf_infer/                  # Python utilities
└── tests/
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `native-inference` | Pure Rust BitNet kernels (wf_server) |
| `llama-inference` | llama.cpp-based inference (dlm_server) |

## Building

```bash
cd rust

# Pure Rust (wf_server)
cargo build --release --bin wf_server --features native-inference

# With llama.cpp (dlm_server) - requires setup_llama_cpp.sh first
cargo build --release --bin dlm_server --features llama-inference
```

## Monorepo Integration

| Package | Relationship |
|---------|--------------|
| `training` | Produces checkpoints to convert |
| `deployer` | Cloud deployment orchestration |
| `architecture` | BitLinear layer definitions |
