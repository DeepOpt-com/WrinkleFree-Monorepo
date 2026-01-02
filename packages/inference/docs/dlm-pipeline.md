# DLM Inference Pipeline

End-to-end guide for running DLM (Diffusion Language Model) inference with block diffusion decoding.

## Overview

DLM models use Fast-dLLM v2 block diffusion to generate multiple tokens in parallel. This requires:
- Models trained with the DLM objective (`objectives.dlm.enabled=true`)
- The `dlm_server` binary (not `native_server`)
- GGUF model files in TQ1_0 or I2_S format

**Performance Note**: The ~2.5x speedup over autoregressive decoding is theoretical. Actual speedup depends on model, hardware, and workload. Benchmark for your use case.

## Pipeline Steps

### Step 1: Train with DLM Objective

Train using `unified.yaml` config which enables DLM:

```bash
uv run --package wrinklefree python scripts/train_lightning.py \
    model=smollm2_135m training=unified
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

**CRITICAL**: Use the Microsoft BitNet converter, not standard llama.cpp!

```bash
# Fix architecture name (capital N -> lowercase n)
sed -i 's/BitNetForCausalLM/BitnetForCausalLM/g' models/dlm-checkpoint/config.json

# Convert with Microsoft BitNet converter
uv run python extern/reference/BitNet.cpp/utils/convert-hf-to-gguf-bitnet.py \
    models/dlm-checkpoint \
    --outfile models/dlm-model.gguf \
    --outtype i2_s

# Verify size (should be ~1.1GB for 2B model with I2_S)
ls -lh models/dlm-model.gguf
```

**DO NOT USE**: TQ2_0 for bf16 checkpoints - it produces garbage output.

### Step 4: Start DLM Server

```bash
# Build if needed (see CLAUDE.md for full build instructions)
cd extern/sglang-bitnet/sgl-model-gateway
cargo build --release

# Run server
./target/release/dlm_server \
    --model models/dlm-model.gguf \
    --port 30000 \
    --block-size 32 \
    --threshold 0.95
```

The server auto-detects the mask token from the model vocabulary.

### Step 5: Test Inference

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
| N/A | `block_size=32` | Tune for speed/quality |
| N/A | `threshold=0.95` | Lower = faster, less quality |

## Using the Python Client

```python
from wrinklefree_inference.client import BitNetClient

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
