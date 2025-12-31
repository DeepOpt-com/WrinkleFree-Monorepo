# Getting Started with BitNet Inference

This guide walks you through serving a 1.58-bit BitNet model with the native sgl-kernel server.

## Prerequisites

- Python 3.10+
- `gcloud` CLI (for downloading checkpoints from GCS)
- AVX2 or AVX512 CPU support (run `cat /proc/cpuinfo | grep avx` to check)

## Quick Start (5 minutes)

### 1. Install dependencies

```bash
cd packages/inference

# One-time setup: builds sgl-kernel with BitNet SIMD kernels
./scripts/setup-cpu.sh
```

### 2. Download a checkpoint

**Option A: From GCS (recommended)**
```bash
mkdir -p models/dlm-bitnet-2b
# Download model files only (excludes optimizer state)
gcloud storage cp \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.json' \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.safetensors' \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.jinja' \
    models/dlm-bitnet-2b/
```

**Option B: From HuggingFace**
```bash
# Downloads to ~/.cache/huggingface/
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoModelForCausalLM.from_pretrained('microsoft/BitNet-b1.58-2B-4T'); \
    AutoTokenizer.from_pretrained('microsoft/BitNet-b1.58-2B-4T')"
```

### 3. Start the server

```bash
# Auto-converts checkpoint to packed .bin format on first run
./scripts/serve_native.sh models/dlm-bitnet-2b
```

You should see:
```
Converting checkpoint to packed format...
Starting native BitNet server...
  Model:     /path/to/models/dlm-bitnet-2b.bin
  Tokenizer: /path/to/models/dlm-bitnet-2b
Native BitNet kernels available: True
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

| Backend | Speed | Best For |
|---------|-------|----------|
| **Native sgl-kernel** | ~29 tok/s | Single user, max throughput |
| BitNet.cpp | ~26 tok/s | Alternative if sgl-kernel fails |
| SGLang | ~16 tok/s | Multi-user (continuous batching) |

The native server achieves 29 tok/s on GCP c3d-standard-32 (AMD EPYC Genoa with AVX512).

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
    "temperature": 0.0,
    "repetition_penalty": 1.2
}
```

**Response:**
```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "bitnet-native",
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

Returns: `{"status": "healthy", "native_kernel": true}`

## Troubleshooting

### "Checkpoint not found"

Download the checkpoint first:
```bash
mkdir -p models/dlm-bitnet-2b
gcloud storage cp -r gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/* models/dlm-bitnet-2b/
```

### "Native BitNet kernels available: False"

Rebuild sgl-kernel:
```bash
./scripts/setup-cpu.sh
```

### Slow inference (< 20 tok/s)

1. Check AVX512 support: `cat /proc/cpuinfo | grep avx512`
2. Use a machine with DDR5 memory (higher bandwidth)
3. On GCP, use `c3d-standard-*` instances (AMD EPYC Genoa)

### "ModuleNotFoundError: sgl_kernel"

Install sgl-kernel:
```bash
.venv/bin/pip install -e extern/sglang-bitnet/sgl-kernel --no-build-isolation
```

## Next Steps

- [Native Inference Deep Dive](bitnet-native-inference.md) - Architecture and optimization details
- [CLAUDE.md](../CLAUDE.md) - Full package documentation
