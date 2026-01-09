# WrinkleFree Inference Engine

Rust inference engine for **BitNet** 1.58-bit quantized LLMs with DLM block diffusion support.

## Quick Reference

| Task | Command |
|------|---------|
| Convert checkpoint to GGUF | `python scripts/convert_checkpoint_to_gguf.py checkpoint/ -o model.gguf` |
| Build wf_server | `cd extern/sglang-bitnet/sgl-model-gateway && cargo build --release --bin wf_server --features native-inference` |
| Build dlm_server | `cd extern/sglang-bitnet/sgl-model-gateway && cargo build --release --bin dlm_server --features llama-inference` |
| Benchmark wf_server | `./wf_server --model-path model.gguf --benchmark --benchmark-iterations 10` |
| Test API | `curl http://localhost:30000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hello"}]}'` |

## Binaries

### wf_server (Pure Rust)

Native BitNet inference server with SIMD-optimized ternary kernels. No C++ dependencies.

```bash
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
# Build llama.cpp first
cd extern/sglang-bitnet/3rdparty/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build -j4

# Build dlm_server
cd ../sgl-model-gateway
cargo build --release --bin dlm_server --features llama-inference

# Set library path and run
export LD_LIBRARY_PATH="extern/sglang-bitnet/3rdparty/llama.cpp/build/src:extern/sglang-bitnet/3rdparty/llama.cpp/build/ggml/src"
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
├── scripts/
│   └── convert_checkpoint_to_gguf.py  # GGUF conversion
├── extern/sglang-bitnet/
│   ├── 3rdparty/llama.cpp/            # llama.cpp (conversion + dlm_server backend)
│   └── sgl-model-gateway/             # Rust inference engine (wf-inference crate)
│       └── src/
│           ├── bin/
│           │   ├── wf_server.rs       # Pure Rust BitNet server
│           │   └── dlm_server.rs      # DLM block diffusion server
│           ├── engine/                # Transformer forward pass
│           ├── gguf/                  # Pure Rust GGUF reader
│           ├── kernels/bitnet/        # SIMD ternary kernels
│           └── inference/             # DLM scheduler
└── docs/                              # Documentation
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `native-inference` | Pure Rust BitNet kernels (wf_server) |
| `llama-inference` | llama.cpp-based inference (dlm_server) |

## Building

```bash
cd extern/sglang-bitnet/sgl-model-gateway

# Pure Rust (wf_server)
cargo build --release --bin wf_server --features native-inference

# With llama.cpp (dlm_server)
cargo build --release --bin dlm_server --features llama-inference
```

## Monorepo Integration

| Package | Relationship |
|---------|--------------|
| `training` | Produces checkpoints to convert |
| `deployer` | Cloud deployment orchestration |
| `architecture` | BitLinear layer definitions |
