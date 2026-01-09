# WrinkleFree Inference Engine

Rust inference engine for **BitNet** 1.58-bit quantized LLMs.

## Quick Reference

| Task | Command |
|------|---------|
| Convert checkpoint to GGUF | `python scripts/convert_checkpoint_to_gguf.py checkpoint/ -o model.gguf` |
| Build wf_server | `cd rust && cargo build --release --bin wf_server --features native-inference` |
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

## GGUF Conversion

```bash
# Convert checkpoint to GGUF (I2_S recommended for production)
python scripts/convert_checkpoint_to_gguf.py /path/to/checkpoint --outfile model.gguf

# For larger models
python scripts/convert_checkpoint_to_gguf.py /path/to/checkpoint --outfile model.gguf --outtype i2_s
```

**CRITICAL**: Never use TQ2_0 for bf16 checkpoints - it corrupts weights!

| Format | Notes |
|--------|-------|
| **I2_S** | Recommended - fastest, best compatibility |
| F16 | Default, works for all models |
| TQ1_0 | Requires hidden_size % 256 == 0 |
| TQ2_0 | DO NOT USE for bf16 checkpoints |

## API

The server exposes an OpenAI-compatible API:

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
│       │   └── wf_server.rs       # Pure Rust BitNet server
│       ├── engine/                # Transformer forward pass
│       ├── gguf/                  # Pure Rust GGUF reader
│       └── kernels/bitnet/        # SIMD ternary kernels
├── scripts/
│   └── convert_checkpoint_to_gguf.py
├── src/wf_infer/                  # Python utilities
└── tests/
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `native-inference` | Pure Rust BitNet kernels (wf_server) |

## Building

```bash
cd rust

# Pure Rust (wf_server)
cargo build --release --bin wf_server --features native-inference
```

## Monorepo Integration

| Package | Relationship |
|---------|--------------|
| `training` | Produces checkpoints to convert |
| `deployer` | Cloud deployment orchestration |
| `architecture` | BitLinear layer definitions |
