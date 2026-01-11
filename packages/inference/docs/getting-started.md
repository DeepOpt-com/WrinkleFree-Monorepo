# Getting Started with BitNet Inference

This guide walks you through serving a 1.58-bit BitNet model with the native Rust inference engine.

## Prerequisites

- Python 3.10+
- Rust toolchain (for building wf_server)
- `gcloud` CLI (for downloading checkpoints from GCS)
- AVX2 or AVX512 CPU support (run `cat /proc/cpuinfo | grep avx` to check)

## Quick Start (5 minutes)

### 1. Build the Rust server

```bash
cd packages/inference/rust

# Build the pure Rust server (no C++ dependencies)
cargo build --release --bin wf_server --features native-inference
```

### 2. Download a model (GGUF format)

**Option A: Pre-converted GGUF from HuggingFace**
```bash
mkdir -p models/bitnet-2b
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
    --local-dir models/bitnet-2b
```

**Option B: Convert from checkpoint**
```bash
# Download checkpoint
mkdir -p models/dlm-bitnet-2b
gcloud storage cp \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.json' \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.safetensors' \
    models/dlm-bitnet-2b/

# Convert to GGUF
python scripts/convert_checkpoint_to_gguf.py models/dlm-bitnet-2b -o models/model.gguf
```

### 3. Start the server

```bash
./rust/target/release/wf_server \
    --model-path models/bitnet-2b/ggml-model-i2_s.gguf \
    --port 30000
```

You should see:
```
WrinkleFree Inference Engine
Loading model: models/bitnet-2b/ggml-model-i2_s.gguf
Server listening on 0.0.0.0:30000
```

### 4. Test the API

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

### 5. Start the chat UI (optional)

In a new terminal:
```bash
cd packages/inference
uv run streamlit run demo/serve_sglang.py --server.port 7860
```

Open http://localhost:7860 in your browser.

## Performance

| Server | Speed | Best For |
|--------|-------|----------|
| **wf_server** | ~26 tok/s | Pure Rust, no dependencies |
| **dlm_server** | ~60 tok/s | DLM block diffusion (requires llama.cpp) |

## API Reference

The server exposes an OpenAI-compatible API:

### Chat Completions

```bash
POST /v1/chat/completions
```

**Request:**
```json
{
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 128,
    "temperature": 0.0
}
```

**Response:**
```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "bitnet",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello! How can I help?"},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 6,
        "total_tokens": 16
    }
}
```

### Health Check

```bash
GET /health
```

Returns: `{"status": "healthy"}`

## Troubleshooting

### "Model not found"

Ensure you have a valid GGUF file:
```bash
ls -la models/*.gguf
```

### Slow inference (< 20 tok/s)

1. Check AVX512 support: `cat /proc/cpuinfo | grep avx512`
2. Use a machine with DDR5 memory (higher bandwidth)
3. On GCP, use `c3d-standard-*` instances (AMD EPYC Genoa)

### Build errors

Ensure you have:
- Rust 1.75+ (`rustup update`)

## Next Steps

- [DLM Pipeline](dlm-pipeline.md) - Block diffusion for faster inference
- [GGUF Conversion](gguf-conversion.md) - Converting checkpoints
- [CLAUDE.md](../CLAUDE.md) - Full package documentation
