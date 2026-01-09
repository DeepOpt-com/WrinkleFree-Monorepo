# WrinkleFree Inference Engine

Rust inference engine for **BitNet** 1.58-bit quantized LLMs with DLM block diffusion support.

## Quick Start

```bash
# 1. Convert checkpoint to GGUF
python scripts/convert_checkpoint_to_gguf.py /path/to/checkpoint --outfile model.gguf

# 2. Build the server
cd extern/sglang-bitnet/sgl-model-gateway
cargo build --release --bin wf_server --features native-inference

# 3. Run inference
./target/release/wf_server --model-path model.gguf --port 30000

# 4. Test the API
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

## Servers

### wf_server - Pure Rust BitNet

Native inference with SIMD-optimized ternary kernels. No C++ dependencies.

```bash
cargo build --release --bin wf_server --features native-inference
./target/release/wf_server --model-path model.gguf --port 30000
```

**Benchmark mode:**
```bash
./target/release/wf_server --model-path model.gguf --benchmark --benchmark-iterations 10
```

### dlm_server - DLM Block Diffusion

Fast-dLLM v2 for ~2.5x faster inference via parallel block decoding.

```bash
# Build llama.cpp first
cd extern/sglang-bitnet/3rdparty/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build -j4
cd ../sgl-model-gateway

# Build and run
cargo build --release --bin dlm_server --features llama-inference
export LD_LIBRARY_PATH="../3rdparty/llama.cpp/build/src:../3rdparty/llama.cpp/build/ggml/src"
./target/release/dlm_server --model-path model.gguf --port 30000
```

## GGUF Conversion

```bash
# Default (F16)
python scripts/convert_checkpoint_to_gguf.py checkpoint/ -o model.gguf

# Quantized (I2_S - recommended for production)
python scripts/convert_checkpoint_to_gguf.py checkpoint/ -o model.gguf --outtype i2_s
```

| Format | Size | Notes |
|--------|------|-------|
| **I2_S** | ~1.1GB (2B) | Fastest, recommended |
| F16 | ~4.5GB (2B) | Default, compatible |
| TQ1_0 | ~2.2GB (2B) | Requires hidden_size % 256 == 0 |

**Warning:** Never use TQ2_0 for bf16 DLM checkpoints - it corrupts weights!

## API

OpenAI-compatible at `/v1/chat/completions`:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Architecture

```
packages/inference/
├── scripts/
│   └── convert_checkpoint_to_gguf.py   # Checkpoint → GGUF
├── extern/sglang-bitnet/
│   ├── 3rdparty/llama.cpp/             # Conversion tools + dlm_server backend
│   └── sgl-model-gateway/              # wf-inference Rust crate
│       └── src/
│           ├── bin/wf_server.rs        # Pure Rust server
│           ├── bin/dlm_server.rs       # DLM server
│           ├── engine/                 # Transformer implementation
│           ├── gguf/                   # GGUF reader
│           ├── kernels/bitnet/         # SIMD kernels
│           └── inference/              # DLM scheduler
└── docs/
```

## Building

```bash
cd extern/sglang-bitnet/sgl-model-gateway

# Pure Rust (no dependencies)
cargo build --release --bin wf_server --features native-inference

# With llama.cpp
cargo build --release --bin dlm_server --features llama-inference
```

## Performance

wf_server on AMD EPYC (32 cores), BitNet 2B I2_S:
- **Prefill**: 106 tok/s
- **Decode**: 7 tok/s

## License

MIT
