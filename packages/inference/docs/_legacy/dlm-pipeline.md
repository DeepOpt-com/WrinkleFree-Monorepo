# DLM Inference Pipeline

End-to-end guide for running DLM (Diffusion Language Model) inference with block diffusion decoding.

## Overview

DLM models use Fast-dLLM v2 block diffusion to generate multiple tokens in parallel. This requires:
- Models trained with the DLM objective (`objectives.dlm.enabled=true`)
- The `dlm_server` binary (not `wf_server`)
- GGUF model files in I2_S format
- llama.cpp (setup via `scripts/setup_llama_cpp.sh`)

**Performance Note**: The ~2.5x speedup over autoregressive decoding is theoretical. Actual speedup depends on model, hardware, and workload. Benchmark for your use case.

## Pipeline Steps

### Step 1: Train with DLM Objective

Train using `unified.yaml` config which enables DLM:

```bash
uv run --package wf-train python scripts/train_lightning.py \
    model=smollm2_135m training=base
```

Key training config parameters:
```yaml
objectives:
  dlm:
    enabled: true
    weight: 0.5
    mask_prob: 0.15
    mask_token_id: 0  # MUST match inference
    use_complementary_masks: true
```

### Step 2: Download Checkpoint

```bash
mkdir -p models/dlm-checkpoint
gcloud storage cp \
    'gs://wrinklefree-checkpoints/dlm/my-model/*.json' \
    'gs://wrinklefree-checkpoints/dlm/my-model/*.safetensors' \
    models/dlm-checkpoint/
```

### Step 3: Convert to GGUF

```bash
# Convert checkpoint to GGUF (I2_S recommended)
python scripts/convert_checkpoint_to_gguf.py \
    models/dlm-checkpoint \
    --outfile models/dlm-model.gguf \
    --outtype i2_s

# Verify size (should be ~1.1GB for 2B model with I2_S)
ls -lh models/dlm-model.gguf
```

**DO NOT USE**: TQ2_0 for bf16 checkpoints - it produces garbage output.

### Step 4: Setup llama.cpp

```bash
# Download and build llama.cpp (required for dlm_server)
./scripts/setup_llama_cpp.sh
```

### Step 5: Build and Start DLM Server

```bash
# Build dlm_server
cd rust
cargo build --release --features llama-inference --bin dlm_server

# Set library path
export LD_LIBRARY_PATH="../extern/llama.cpp/build/src:../extern/llama.cpp/build/ggml/src"

# Run server with adaptive mode (RECOMMENDED - speed + quality)
./target/release/dlm_server \
    -m ../models/dlm-model.gguf \
    --port 30000 \
    --decode-mode adaptive \
    --threshold 0.9

# Or greedy mode (fastest, lower quality)
./target/release/dlm_server \
    -m ../models/dlm-model.gguf \
    --port 30000 \
    --decode-mode greedy
```

**Decode Mode Options** (per Fast-dLLM v2 paper):

| Mode | Threshold | Throughput | Quality | Use Case |
|------|-----------|------------|---------|----------|
| `greedy` | N/A | ~61 tok/s | Baseline | Maximum speed |
| `iterative` | 0.7 | ~54 tok/s | Good | Balanced |
| `iterative` | 0.9 | ~21 tok/s | **Best** | Per-paper (slow) |
| **`adaptive`** | 0.9 | **~61 tok/s** | **Best** | **RECOMMENDED** |

**Adaptive mode** uses progressive thresholds (0.5 → 0.7 → 0.9) to quickly unmask easy tokens in early iterations, then applies full refinement only to difficult positions. This achieves ~3x speedup over fixed iterative while maintaining quality.

The server auto-detects the mask token from the model vocabulary.

### Step 6: Test Inference

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 50
    }'
```

Expected: Coherent response like "2 + 2 equals 4."

## Configuration Sync

These parameters **must match** between training and inference:

| Training Config | Inference Config | Default | Notes |
|-----------------|------------------|---------|-------|
| `objectives.dlm.mask_token_id` | Auto-detected or `--mask-token-id` | 0 | Critical for correctness |
| Model vocab_size | GGUF metadata | - | Must match exactly |

Parameters that can differ:
| Training | Inference | Notes |
|----------|-----------|-------|
| `mask_prob=0.15` | N/A | Only affects training |
| `use_complementary_masks=true` | N/A | Only affects training |
| N/A | `block_size=32` | Must match training block size |
| N/A | `--decode-mode greedy` | Fast single-pass (~61 tok/s) |
| N/A | `--decode-mode iterative` | Per-paper correctness |
| N/A | `--threshold 0.9` | Higher = better quality, slower |

## Using the Python Client

```python
from wf_infer.client import BitNetClient

client = BitNetClient.from_url("http://localhost:30000")

# Non-streaming
response = client.chat_openai([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
])
print(response)

# Streaming
for token in client.chat_openai_stream(messages):
    print(token, end="", flush=True)
```

## Troubleshooting

See [dlm-troubleshooting.md](dlm-troubleshooting.md) for common issues.

## Related Docs

- [GGUF Conversion Guide](gguf-conversion.md)
- [Getting Started](getting-started.md)
- [Package CLAUDE.md](../CLAUDE.md)
