# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree Inference Engine serves **DLM-BitNet** models (Diffusion Language Models with 1.58-bit quantization):

- **Primary Server**: Rust `dlm_server` with Fast-dLLM v2 block diffusion
- **Model Format**: GGUF (converted from training checkpoints)
- **Frontend**: Streamlit chat UI with SSE streaming
- **Architecture Support**: BitNet with optional SubLN (Sub-Layer Normalization)

**Key Rule**: Use `dlm_server` for DLM checkpoints, `native_server` for regular BitNet.

**Note**: All paths in this document are relative to `packages/inference/`.

## Quick Reference

| Task | Command |
|------|---------|
| Convert checkpoint to GGUF | `python scripts/convert_checkpoint_to_gguf.py checkpoint/ -o model.gguf` |
| Start DLM server | `./scripts/launch_rust_gateway.sh --native -m model.gguf` |
| Start chat UI | `uv run streamlit run demo/serve_sglang.py --server.port 7860` |
| Test API | `curl http://localhost:30000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'` |

## GGUF Conversion (CRITICAL)

### Converter Selection Guide

| Checkpoint Type | Converter | Output Format | Notes |
|-----------------|-----------|---------------|-------|
| **WrinkleFree DLM** (bf16) | `scripts/convert_checkpoint_to_gguf.py` | `i2_s` (default) | Auto-fixes arch name |
| **WrinkleFree SubLN** | Export first, then convert | `f16` or `tq1_0` | Requires tensor rename |
| Microsoft BitNet | `extern/BitNet/utils/convert-hf-to-gguf-bitnet.py` | `i2_s` | Packed 2-bit weights |
| Standard HuggingFace | `extern/sglang-bitnet/3rdparty/llama.cpp/convert_hf_to_gguf.py` | Any | Generic converter |

### Standard DLM Checkpoint Conversion

```bash
# 1. Convert to GGUF (auto-fixes BitNetForCausalLM → BitnetForCausalLM)
python scripts/convert_checkpoint_to_gguf.py \
  /path/to/checkpoint \
  --outfile models/model.gguf \
  --outtype i2_s \
  --validate

# 2. Test with llama-cli
./extern/sglang-bitnet/3rdparty/llama.cpp/build/bin/llama-cli \
  -m models/model.gguf -p "Hello" -n 50
```

### SubLN Checkpoint Conversion (WrinkleFree Training)

SubLN models require tensor renaming before GGUF conversion:

```bash
# 1. Export SubLN checkpoint (renames .0/.1 → attn_sub_norm/ffn_sub_norm)
uv run --package wrinklefree python packages/training/scripts/export_subln_checkpoint.py \
  --checkpoint /path/to/lightning_checkpoint.pt \
  --output-dir models/exported-subln \
  --config-path /path/to/config.json

# 2. Convert to GGUF (use F16 or TQ1_0, NOT TQ2_0 for bf16!)
python extern/sglang-bitnet/3rdparty/llama.cpp/convert_hf_to_gguf.py \
  models/exported-subln \
  --outfile models/model-subln.gguf \
  --outtype f16

# 3. Test loading (output may be nonsense for DLM - needs block diffusion)
./extern/sglang-bitnet/3rdparty/llama.cpp/build/bin/llama-cli \
  -m models/model-subln.gguf -p "Hello" -n 20
```

### Output Format Guide

| Format | Size (2B) | Speed | Use Case |
|--------|-----------|-------|----------|
| **i2_s** | ~1.1GB | Fast | **Default, recommended for vanilla llama.cpp** |
| **tq1_0** | ~680MB | Faster | Good for bf16 DLM checkpoints |
| tq2_0 | ~780MB | Fastest | **AVOID for bf16 checkpoints - produces garbage!** |
| f16 | ~4.5GB | Slow | Reference/debugging only |
| tl1/tl2 | ~1.1GB | Fastest | Requires kernel config, CPU-specific |

**CRITICAL**: Never use TQ2_0 for bf16 DLM checkpoints - it corrupts already-ternary weights!

### Common Conversion Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "BitnetForCausalLM not found" | Arch name mismatch | Use `convert_checkpoint_to_gguf.py` (auto-fixes) |
| Shape mismatch errors | Packed weights not unpacked | Use Microsoft BitNet converter |
| Gibberish output | Wrong quantization type | Use `i2_s` or `tq1_0`, not `tq2_0` |
| Missing tokenizer | Incomplete checkpoint | Copy `tokenizer.json` + `tokenizer_config.json` |
| SubLN tensor errors | Tensors not renamed | Run `export_subln_checkpoint.py` first |

## Model Types

### DLM (Diffusion Language Model)

DLM models use block diffusion decoding - they generate multiple tokens per forward pass by iteratively unmasking.

**Requirements**:
- Trained with DLM objective (`objectives.dlm.enabled=true`)
- `mask_token_id` must match training (default: 0)
- **Must use `dlm_server`** - autoregressive decoding produces nonsense

**Decoding Modes** (dlm_server):
| Mode | Threshold | Speed | Quality |
|------|-----------|-------|---------|
| `greedy` | N/A | ~61 tok/s | Baseline |
| `adaptive` | 0.9 | ~61 tok/s | **Best (recommended)** |
| `iterative` | 0.9 | ~21 tok/s | Best (slow) |

### Regular BitNet

Standard autoregressive BitNet models (e.g., `microsoft/BitNet-b1.58-2B-4T`).

**Requirements**:
- Use `native_server` or `batch_server`
- Works with standard llama-cli

### SubLN (Sub-Layer Normalization)

SubLN adds normalization before output projections for training stability.

**Tensor naming**:
```
WrinkleFree format:              llama.cpp format:
.self_attn.o_proj.0.weight  →   .self_attn.attn_sub_norm.weight
.self_attn.o_proj.1.weight  →   .self_attn.o_proj.weight
.mlp.down_proj.0.weight     →   .mlp.ffn_sub_norm.weight
.mlp.down_proj.1.weight     →   .mlp.down_proj.weight
```

Use `export_subln_checkpoint.py` to rename before GGUF conversion.

## Architecture

```
packages/inference/
├── scripts/
│   ├── convert_checkpoint_to_gguf.py  # PRIMARY converter (auto-fixes, validates)
│   ├── convert_dlm_to_gguf.py         # TL1/TL2 specific conversion
│   ├── convert_to_sglkernel.py        # For sgl-kernel packed format
│   ├── launch_rust_gateway.sh         # Start Rust DLM server
│   └── serve.sh                       # Full stack launcher
├── demo/
│   └── serve_sglang.py                # Streamlit chat UI
├── extern/
│   └── sglang-bitnet/
│       ├── 3rdparty/llama.cpp/        # Stripped to BitNet essentials
│       │   ├── convert_hf_to_gguf.py  # Generic HF→GGUF converter
│       │   ├── gguf-py/               # GGUF Python library
│       │   ├── src/                   # Core llama.cpp
│       │   ├── ggml/                  # GGML tensor library
│       │   ├── examples/main/         # llama-cli
│       │   └── examples/quantize/     # llama-quantize
│       └── sgl-model-gateway/         # Rust HTTP server
└── src/wrinklefree_inference/         # Python utilities
```

**llama.cpp is stripped** to BitNet/DLM essentials only:
- Core: `src/`, `ggml/`, `common/`, `include/`
- Tools: `examples/main/` (llama-cli), `examples/quantize/`
- Conversion: `convert_hf_to_gguf.py`, `gguf-py/`
- Non-essential code moved to `_deprecated/`

## Server Selection

| Server | Decoding | Use Case | API |
|--------|----------|----------|-----|
| **`dlm_server`** | Block diffusion | DLM checkpoints | OpenAI-compatible |
| `native_server` | Autoregressive | Regular BitNet | OpenAI-compatible |
| `batch_server` | Batched AR | Multi-user production | OpenAI-compatible |
| SGLang | AR + batching | Python debugging | OpenAI-compatible |

**For DLM checkpoints, always use `dlm_server`** - it implements Fast-dLLM v2.

## Build llama.cpp

```bash
cd extern/sglang-bitnet/3rdparty/llama.cpp

# Configure (with AVX512 if available)
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_NATIVE=ON

# Build (throttled to prevent system freeze)
cmake --build build -j4

# Verify
ls build/bin/llama-cli build/bin/llama-quantize
ls build/src/libllama.so build/ggml/src/libggml.so
```

## Testing

```bash
# Test GGUF conversion
python scripts/convert_checkpoint_to_gguf.py --help

# Test model loading
./extern/sglang-bitnet/3rdparty/llama.cpp/build/bin/llama-cli \
  -m models/model.gguf -p "Hello" -n 10

# Run unit tests
uv run pytest tests/ -v

# Integration tests (requires running server)
INFERENCE_URL=http://localhost:30000 uv run pytest tests/ -v -m integration
```

## Deployment

### Local Development

```bash
# 1. Build llama.cpp
cd extern/sglang-bitnet/3rdparty/llama.cpp && cmake -B build && cmake --build build -j4 && cd -

# 2. Convert model
python scripts/convert_checkpoint_to_gguf.py checkpoint/ -o models/model.gguf

# 3. Start server
./scripts/launch_rust_gateway.sh --native -m models/model.gguf

# 4. Start UI (another terminal)
uv run streamlit run demo/serve_sglang.py --server.port 7860
```

### Vultr High Frequency (Production)

```bash
# Deploy with -march=native optimization
./scripts/deploy_vultr.sh <instance-ip>

# Test
curl http://<ip>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## Monorepo Integration

| Package | Relationship |
|---------|--------------|
| `training` | Produces checkpoints to convert and serve |
| `deployer` | Cloud deployment orchestration (SkyPilot) |
| `eval` | Uses inference for benchmarks |
| `architecture` | BitLinear/SubLN layer definitions |

## Related Files

| File | Purpose |
|------|---------|
| `packages/training/scripts/export_subln_checkpoint.py` | Export SubLN checkpoints for GGUF |
| `extern/BitNet/utils/convert-hf-to-gguf-bitnet.py` | Microsoft BitNet converter |
| `extern/sglang-bitnet/sgl-model-gateway/` | Rust HTTP server (DLM + native) |
| `models/` | Local GGUF models (gitignored) |

## Build Throttling

All scripts are throttled to prevent system freeze:
- **Builds**: Limited to 4 parallel jobs (`-j4`)
- **Servers**: Pinned to 8 cores via `taskset -c 0-7`

For manual builds, use the safe wrapper:
```bash
./scripts/build-safe.sh cmake --build build
```
