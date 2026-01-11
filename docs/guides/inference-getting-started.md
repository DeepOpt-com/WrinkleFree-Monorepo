# Inference Getting Started

This guide covers running inference with the Rust `wf_server` - a pure Rust BitNet inference engine with SIMD-optimized ternary kernels.

## Prerequisites

- Rust toolchain (`cargo`)
- A trained checkpoint (local or from GCS)
- Python for GGUF conversion

## Quick Start

```bash
cd packages/inference

# 1. Convert checkpoint to GGUF
python scripts/convert_checkpoint_to_gguf.py /path/to/checkpoint \
    --outfile model.gguf \
    --outtype i2_s

# 2. Build the server
cd rust
cargo build --release --bin wf_server --features native-inference

# 3. Run the server
./target/release/wf_server --model-path ../model.gguf --port 30000

# 4. Test the API (in another terminal)
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

## GGUF Conversion

### Format Selection

| Format | Size (2B model) | Notes |
|--------|-----------------|-------|
| **I2_S** | ~1.1GB | Recommended - fastest, best compatibility |
| F16 | ~4.5GB | Default, works for all models |
| TQ1_0 | ~2.2GB | Requires `hidden_size % 256 == 0` |

**Warning**: Never use TQ2_0 for bf16 checkpoints - it corrupts weights!

### Conversion Examples

```bash
# From local checkpoint (I2_S recommended)
python scripts/convert_checkpoint_to_gguf.py checkpoint/ \
    --outfile model.gguf \
    --outtype i2_s

# From GCS
python scripts/convert_checkpoint_to_gguf.py \
    gs://wrinklefree-checkpoints/checkpoints/experiment/final/ \
    --outfile model.gguf \
    --outtype i2_s

# F16 for debugging (larger but universal)
python scripts/convert_checkpoint_to_gguf.py checkpoint/ \
    --outfile model.gguf \
    --outtype f16
```

## Server Options

```bash
./target/release/wf_server \
    --model-path model.gguf \    # GGUF model file (required)
    --port 30000 \               # Server port (default: 30000)
    --host 0.0.0.0 \             # Bind address (default: 0.0.0.0)
    --threads 4 \                # Inference threads (default: auto)
    --context-len 4096           # Max sequence length
```

## API Reference

The server exposes an OpenAI-compatible API.

### Chat Completions

```bash
POST /v1/chat/completions
```

**Request:**
```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
}
```

**Response:**
```json
{
    "id": "chatcmpl-0",
    "object": "chat.completion",
    "model": "model",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "2+2 equals 4."},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 6,
        "total_tokens": 21
    }
}
```

### Health Check

```bash
GET /health
# Returns: "ok"
```

## Benchmarking

Run benchmarks to measure inference performance:

```bash
./target/release/wf_server \
    --model-path model.gguf \
    --benchmark \
    --benchmark-iterations 20 \
    --benchmark-max-tokens 64 \
    --benchmark-prompt "The capital of France is"
```

**Example output** (AMD EPYC 32 cores, BitNet 2B I2_S):
- Prefill: ~106 tok/s
- Decode: ~7 tok/s

## Python Client

```python
from wf_infer.client import BitNetClient

# Connect to server
client = BitNetClient(host="localhost", port=30000)

# Health check
if client.health_check():
    print("Server is running")

# Generate text
response = client.generate(
    prompt="The capital of France is",
    max_tokens=50,
    temperature=0.7
)
print(response)
```

## Troubleshooting

### GGUF conversion fails

- Ensure checkpoint has `model.safetensors` or `pytorch_model.bin`
- Check that tokenizer files exist (`tokenizer.json` or `tokenizer.model`)

### Server crashes on load

- Check available memory (GGUF is memory-mapped)
- Try F16 format for debugging

### Gibberish output

- Likely used TQ2_0 format - reconvert with I2_S or F16
- Check that model was trained with BitNet layers

## Next Steps

- **Full inference docs**: See [packages/inference/README.md](../../packages/inference/README.md)
- **GGUF format details**: See [packages/inference/docs/gguf-conversion.md](../../packages/inference/docs/gguf-conversion.md)
