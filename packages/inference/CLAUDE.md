# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree Inference Engine serves **BitNet** models (1.58-bit quantized LLMs):

- **Model Format**: GGUF (converted from training checkpoints)
- **Architecture Support**: BitNet with SubLN (Sub-Layer Normalization)
- **Inference Options**:
  - **llama-cli**: Standard autoregressive decoding (works for all models)
  - **dlm_server**: Fast-dLLM v2 block diffusion (speed optimization for DLM-trained models)

**Note**: Models trained with `training=base` support BOTH autoregressive and block diffusion inference. The unified config trains with CE loss (autoregressive) + DLM loss (diffusion), so llama-cli works correctly. Use dlm_server for potential speed gains via parallel token prediction.

**Note**: All paths in this document are relative to `packages/inference/`.

## Quick Reference

| Task | Command |
|------|---------|
| Convert checkpoint to GGUF | `python scripts/convert_checkpoint_to_gguf.py checkpoint/ -o model.gguf` |
| Build dlm_server | `cd extern/sglang-bitnet/sgl-model-gateway && cargo build --release --bin dlm_server --features=native-inference` |
| Start DLM server | See "Running dlm_server" section below |
| Test API | `curl http://localhost:30000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hello"}]}'` |

## GGUF Conversion

### Basic Conversion

```bash
# Convert to GGUF (auto-fixes architecture name, F16 default)
python scripts/convert_checkpoint_to_gguf.py \
  /path/to/checkpoint \
  --outfile models/model.gguf

# For 2B+ models, use TQ1_0 for smaller size (auto-fallback to F16 if incompatible)
python scripts/convert_checkpoint_to_gguf.py \
  /path/to/checkpoint \
  --outfile models/model.gguf \
  --outtype tq1_0
```

### Output Format Guide

| Format | Size (135M) | Size (2B) | Notes |
|--------|-------------|-----------|-------|
| **f16** | ~260MB | ~4.5GB | **Default, works for ALL models** |
| **i2_s** | ~55MB | ~1.1GB | **Fastest - ternary quantized, AVX-512 optimized** |
| tq1_0 | N/A | ~2.2GB | Requires hidden_size % 256 == 0 (2B+, not 135M) |
| bf16 | ~260MB | ~4.5GB | Same as F16, alternative |
| f32 | ~520MB | ~9GB | For debugging only |

**CRITICAL**:
- **Use I2_S for production** - it's the fastest and most memory efficient
- Never use TQ2_0 for bf16 DLM checkpoints - it corrupts weights!
- After F16 conversion, quantize to I2_S: `llama-quantize model-f16.gguf model-i2s.gguf I2_S`

### Common Conversion Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "BitnetForCausalLM not found" | Arch name mismatch | Script auto-fixes this |
| TQ1_0 shape error | Model too small | Use F16 (default) |
| Missing tokenizer | Incomplete checkpoint | Copy `tokenizer.json` + `tokenizer_config.json` |

## Running dlm_server

DLM models require block diffusion decoding. The `dlm_server` implements Fast-dLLM v2.

### Prerequisites

```bash
# 1. Build llama.cpp (required for dlm_server)
cd extern/sglang-bitnet/3rdparty/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build -j4

# 2. Build dlm_server
cd ../sgl-model-gateway
cargo build --release --bin dlm_server --features=native-inference
```

### Starting the Server

```bash
# Set library path and run
export LD_LIBRARY_PATH="extern/sglang-bitnet/3rdparty/llama.cpp/build/src:extern/sglang-bitnet/3rdparty/llama.cpp/build/ggml/src:$LD_LIBRARY_PATH"

./extern/sglang-bitnet/sgl-model-gateway/target/release/dlm_server \
  --model-path models/model.gguf \
  --port 30000 \
  --decode-mode adaptive \
  --threshold 0.7 \
  --block-size 32 \
  --mask-token-id 0
```

### Server Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path` | (required) | Path to GGUF model |
| `--port` | 30000 | Server port |
| `--decode-mode` | adaptive | `greedy`, `iterative`, or `adaptive` |
| `--threshold` | 0.7 | Confidence threshold (0-1) |
| `--block-size` | 32 | Tokens per diffusion block |
| `--mask-token-id` | auto | Override mask token ID (usually 0 for DLM) |

### Decoding Modes

| Mode | Speed | Quality | Notes |
|------|-------|---------|-------|
| `greedy` | Fast | Baseline | Single pass per block |
| `adaptive` | Fast | **Best** | Refinement based on confidence |
| `iterative` | Slow | Best | Multiple refinement passes |

### Testing the API

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":50}'
```

## SubLN Architecture

SubLN adds normalization before output projections. WrinkleFree checkpoints use this by default.

**Note**: The GGUF converter handles SubLN tensor naming automatically. No manual export required.

## LRC (Low-Rank Correction)

LRC recovers accuracy lost during quantization by adding a low-rank correction term:

```
output = W_quant @ Q_a(X) + U @ V^T @ X
```

Where:
- `W_quant`: Quantized ternary weights (±1, 0)
- `Q_a(X)`: Quantized activations (int8)
- `U, V`: Low-rank correction matrices (rank typically 32-128)

### LRC Tensor Types

14 LRC tensors per layer (U/V pairs for each projection):

| Projection | Tensors |
|------------|---------|
| Attention Q | `attn_q.lrc_U`, `attn_q.lrc_V` |
| Attention K | `attn_k.lrc_U`, `attn_k.lrc_V` |
| Attention V | `attn_v.lrc_U`, `attn_v.lrc_V` |
| Attention O | `attn_output.lrc_U`, `attn_output.lrc_V` |
| FFN Gate | `ffn_gate.lrc_U`, `ffn_gate.lrc_V` |
| FFN Up | `ffn_up.lrc_U`, `ffn_up.lrc_V` |
| FFN Down | `ffn_down.lrc_U`, `ffn_down.lrc_V` |

### Pipeline Integration

LRC is fully supported in the inference pipeline:

1. **Training** (`packages/architecture`): `BitLinearLRC` layer stores U/V matrices
2. **Conversion** (`convert_hf_to_gguf.py`): Maps `q_proj.lrc_U` → `blk.N.attn_q.lrc_U`
3. **Inference** (`llama.cpp`): `llm_build_lrc_mm()` applies correction during forward pass

**Note**: LRC tensors are automatically detected and converted. No special flags needed.

### LRC Model Sizes

LRC adds ~5-10% size overhead (depending on rank):

| Model | Base I2_S | With LRC (rank=64) |
|-------|-----------|-------------------|
| 135M | ~55MB | ~60MB |
| 2B | ~1.1GB | ~1.2GB |

## Architecture

```
packages/inference/
├── scripts/
│   ├── convert_checkpoint_to_gguf.py  # PRIMARY converter (auto-fixes, validates)
│   ├── launch_rust_gateway.sh         # Start Rust DLM server
│   ├── launch_sglang_bitnet.sh        # Start SGLang server
│   ├── serve.sh                       # Full stack launcher (multiple backends)
│   ├── benchmark_*.py                 # Performance benchmarking scripts
│   └── _legacy/                       # Archived conversion scripts
├── demo/
│   └── serve_sglang.py                # Streamlit chat UI
├── extern/
│   └── sglang-bitnet/                 # Stripped to essentials (~38MB)
│       ├── 3rdparty/
│       │   ├── llama.cpp/             # Core llama.cpp (convert, quantize, cli)
│       │   └── amd/                   # AMD GPU profiling
│       ├── sgl-kernel/                # Native BitNet SIMD kernels
│       ├── sgl-model-gateway/         # Rust HTTP server (dlm_server)
│       └── python/sglang/             # SGLang backend
└── src/wf_infer/                      # Python utilities
```

**sglang-bitnet is stripped** to BitNet/DLM essentials only:
- `3rdparty/llama.cpp/`: Core (`src/`, `ggml/`, `common/`), tools (`examples/main/`, `examples/quantize/`), conversion (`convert_hf_to_gguf.py`, `gguf-py/`)
- `3rdparty/amd/`: AMD GPU profiling utilities
- `sgl-kernel/`: Native BitNet SIMD kernels (AVX2/AVX512)
- `sgl-model-gateway/`: Rust HTTP server (dlm_server)
- `python/sglang/`: SGLang backend for CPU inference
- Removed: benchmark/, examples/, docs/, test/, scripts/, docker/, assets/

## Server Options

**For unified-trained models (CE + DLM):**
- **llama-cli**: Works correctly for autoregressive generation
- **dlm_server**: Faster inference via block diffusion (recommended for production)

**dlm_server** implements Fast-dLLM v2 and can achieve speedups by predicting multiple tokens in parallel. However, llama-cli is a valid fallback for testing and debugging.

## Build llama.cpp

```bash
cd extern/sglang-bitnet/3rdparty/llama.cpp

# Configure with native CPU optimizations (AVX-512 on supported CPUs)
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_C_FLAGS="-march=native" \
  -DCMAKE_CXX_FLAGS="-march=native"

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

# Run unit tests
uv run pytest tests/ -v

# Integration tests (requires running dlm_server)
INFERENCE_URL=http://localhost:30000 uv run pytest tests/ -v -m integration
```

## Local Development Workflow

```bash
# 1. Build llama.cpp
cd extern/sglang-bitnet/3rdparty/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build -j4
cd -

# 2. Build dlm_server
cd extern/sglang-bitnet/sgl-model-gateway
cargo build --release --bin dlm_server --features=native-inference
cd -

# 3. Convert model
python scripts/convert_checkpoint_to_gguf.py checkpoint/ -o models/model.gguf

# 4. Start dlm_server
export LD_LIBRARY_PATH="extern/sglang-bitnet/3rdparty/llama.cpp/build/src:extern/sglang-bitnet/3rdparty/llama.cpp/build/ggml/src"
./extern/sglang-bitnet/sgl-model-gateway/target/release/dlm_server \
  --model-path models/model.gguf --port 30000 --mask-token-id 0

# 5. Test
curl http://localhost:30000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## Monorepo Integration

| Package | Relationship |
|---------|--------------|
| `training` | Produces checkpoints to convert and serve |
| `deployer` | Cloud deployment orchestration (SkyPilot) |
| `eval` | Uses inference for benchmarks |
| `architecture` | BitLinear/BitLinearLRC/SubLN layer definitions |

## Related Files

| File | Purpose |
|------|---------|
| `scripts/convert_checkpoint_to_gguf.py` | Main GGUF converter (wrapper) |
| `extern/sglang-bitnet/3rdparty/llama.cpp/convert_hf_to_gguf.py` | Underlying GGUF converter |
| `extern/sglang-bitnet/sgl-model-gateway/src/bin/dlm_server.rs` | DLM server (Fast-dLLM v2) |
| `models/` | Local GGUF models (gitignored) |

## Build Throttling

All scripts are throttled to prevent system freeze:
- **Builds**: Limited to 4 parallel jobs (`-j4`)
- **Servers**: Pinned to 8 cores via `taskset -c 0-7`

For manual builds, use the safe wrapper:
```bash
./scripts/build-safe.sh cmake --build build
```
