# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree Inference Engine serves **DLM (Diffusion Language Model)** checkpoints with 1.58-bit quantization:
- **Primary Backend**: Rust `dlm_server` with Fast-dLLM v2 block diffusion (NO PYTHON)
- **Model Format**: GGUF (converted from DLM safetensors checkpoints)
- **Frontend**: Streamlit chat UI with SSE streaming
- **Deployment**: Vultr High Frequency (~$0.29/hr), GCP C3D

**IMPORTANT**: Use `dlm_server` (NOT `native_server`) for DLM checkpoints!

## Monorepo Integration

This package is part of the WrinkleFree monorepo.

**Related packages**:
| Package | Relationship |
|---------|--------------|
| `training` | Produces models to serve |
| `deployer` | Cloud deployment orchestration |
| `eval` | Uses inference for benchmarks |

**External submodules**:
- `extern/sglang-bitnet/` - SGLang fork with BitNet support
- `extern/BitNet/` (at monorepo root) - Microsoft BitNet.cpp

**Running from monorepo root**:
```bash
# Start server
uv run --package wrinklefree-inference python packages/inference/scripts/launch_sglang_bitnet.sh

# Run Streamlit UI
uv run --package wrinklefree-inference streamlit run packages/inference/demo/serve_sglang.py
```

## Build & Server Throttling (IMPORTANT)

All launch scripts have **hard-coded CPU limits** to prevent system freeze:
- **Builds**: Limited to 4 parallel jobs (`-j4`)
- **Servers**: Pinned to 8 cores via `taskset -c 0-7`

For manual builds, use the safe wrapper:
```bash
./scripts/build-safe.sh cargo build --release
./scripts/build-safe.sh cmake --build build
```

## CPU-Only Quick Start (Recommended)

**One-time setup** - run the setup script:
```bash
./scripts/setup-cpu.sh
```

**Start serving**:
```bash
# Full stack (server + chat UI)
./scripts/serve.sh --backend sglang

# Or individual components:
./scripts/launch_sglang_bitnet.sh  # Server on port 30000
uv run streamlit run demo/serve_sglang.py --server.port 7860  # Chat UI
```

**Test API**:
```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## Native sgl-kernel Server (RECOMMENDED - 29+ tok/s)

The native server uses sgl-kernel's optimized SIMD kernels (AVX2/AVX512) for maximum CPU throughput.

**Key optimizations:**
- `bitnet_gemv` for single-token decode (8x faster than gemm)
- Greedy decoding by default (eliminates sampling overhead)
- Repetition penalty to reduce output loops
- KV cache for efficient autoregressive generation

### Quick Start (One Command)

```bash
# Download checkpoint from GCS (one-time, excludes optimizer state)
mkdir -p models/dlm-bitnet-2b
gcloud storage cp \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.json' \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.safetensors' \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.jinja' \
    models/dlm-bitnet-2b/

# Start server (auto-converts to .bin if needed)
./scripts/serve_native.sh models/dlm-bitnet-2b

# Start Streamlit UI (in another terminal)
uv run streamlit run demo/serve_sglang.py --server.port 7860
```

### Full Workflow

**Step 1: Prerequisites**
```bash
# Install sgl-kernel with BitNet support (one-time)
./scripts/setup-cpu.sh
```

**Step 2: Download checkpoint from GCS**
```bash
mkdir -p models/dlm-bitnet-2b
gcloud storage cp \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.json' \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.safetensors' \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-2800/*.jinja' \
    models/dlm-bitnet-2b/
```

**Step 3: Start server** (auto-converts checkpoint to packed .bin format)
```bash
./scripts/serve_native.sh models/dlm-bitnet-2b
```

**Step 4: Test**
```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

**Performance:** ~29 tok/s on GCP c3d-standard-32 (AMD EPYC Genoa with AVX512)

**Files:**
| File | Purpose |
|------|---------|
| `scripts/serve_native.sh` | One-command wrapper (auto-converts + serves) |
| `scripts/serve_bitnet_native.py` | Native server with TL2 kernels |
| `scripts/convert_to_sglkernel.py` | Converts bf16 checkpoints to packed format |
| `demo/serve_sglang.py` | Streamlit chat UI |

## BitNet.cpp Quick Start (Alternative - 1.6x slower)

**Prerequisites**: `clang` compiler required (`sudo apt install clang`)

```bash
# Build BitNet.cpp (one-time) - uses setup_env.py to generate kernel headers
cd extern/BitNet
git submodule update --init --recursive
python setup_env.py --hf-repo microsoft/BitNet-b1.58-2B-4T -q i2_s
cd ../..

# Start server (option 1: script)
./scripts/launch_bitnet_cpp.sh

# Start server (option 2: direct) - model is downloaded to models/ by setup_env.py
./extern/BitNet/build/bin/llama-server -m extern/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --host 0.0.0.0 --port 8080

# Start Streamlit chat UI (with BitNet.cpp backend)
BITNET_BACKEND=bitnet_cpp uv run streamlit run demo/serve_sglang.py --server.port 7860

# Test
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "bitnet", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## DLM Deployment (PRIMARY)

**The DLM server uses Fast-dLLM v2 for ~2.5x faster inference via parallel block decoding.**

### IMPORTANT: DLM vs Regular BitNet

| Model Type | Example | Server | Notes |
|------------|---------|--------|-------|
| **DLM checkpoint** | `gs://wrinklefree-checkpoints/dlm/...` | `dlm_server` | Trained with mask tokens |
| Regular BitNet | `microsoft/BitNet-b1.58-2B-4T` | `native_server` | No DLM support |

**The Microsoft BitNet model is NOT a DLM model!** You must use a WrinkleFree DLM checkpoint.

### Quick Start (DLM Checkpoint)

```bash
# 1. Download DLM checkpoint from GCS
mkdir -p models/dlm-bitnet-2b
gsutil -m cp -r 'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-3600/*' \
  models/dlm-bitnet-2b/

# 2. Fix architecture name for llama.cpp (capital N → lowercase n)
sed -i 's/BitNetForCausalLM/BitnetForCausalLM/g' models/dlm-bitnet-2b/config.json

# 3. Convert to GGUF format using Microsoft BitNet converter (handles packed 2-bit weights)
uv run python extern/reference/BitNet.cpp/utils/convert-hf-to-gguf-bitnet.py \
  models/dlm-bitnet-2b --outfile models/dlm-bitnet-2b.gguf --outtype i2_s

# 4. Upload to server and restart (example for Vultr)
rsync -avz models/dlm-bitnet-2b.gguf root@<server-ip>:/opt/wrinklefree/models/
ssh root@<server-ip> "pm2 restart dlm-server"

# 5. Test
curl http://<ip>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}'
```

**Expected output**: Coherent response (e.g., "2 + 2 equals 4")

### Server Binaries (All Rust, No Python)

| Binary | Decoding Method | Use Case | Notes |
|--------|-----------------|----------|-------|
| **`dlm_server`** | Fast-dLLM v2 block diffusion | DLM checkpoints | ~2.5x faster, parallel decode |
| `native_server` | Autoregressive (token-by-token) | Regular BitNet | Standard decoding |
| `batch_server` | Batched autoregressive | Multi-user | Continuous batching |

**For DLM checkpoints, always use `dlm_server`** - it implements the block diffusion algorithm that makes DLM fast.

### Convert DLM Checkpoint to GGUF (CRITICAL)

**WARNING: DLM checkpoints have PACKED 2-bit weights that require special handling!**

DLM checkpoints store weights in a **packed 2-bit format** (4 values per byte) with separate `weight_scale` tensors. Using the standard llama.cpp converter will produce **gibberish output** or shape mismatch errors.

**Use Microsoft BitNet's converter** at `extern/reference/BitNet.cpp/utils/convert-hf-to-gguf-bitnet.py`:

```bash
# IMPORTANT: Fix architecture name first (our training uses BitNetForCausalLM, llama.cpp expects BitnetForCausalLM)
sed -i 's/BitNetForCausalLM/BitnetForCausalLM/g' models/my-checkpoint/config.json

# Convert to GGUF using Microsoft BitNet converter (handles packed weights)
uv run python extern/reference/BitNet.cpp/utils/convert-hf-to-gguf-bitnet.py \
  models/my-checkpoint --outfile models/my-model.gguf --outtype i2_s

# Verify conversion - should be ~1.1GB for I2_S format
ls -lh models/my-model.gguf
```

**DO NOT USE:**
- `extern/sglang-bitnet/3rdparty/llama.cpp/convert_hf_to_gguf.py` - doesn't unpack 2-bit weights
- `llama-quantize` for post-hoc I2_S/TQ2_0 quantization - corrupts already-ternary weights

**How the fix works** (in `convert-hf-to-gguf-bitnet.py`):
1. Build a `scale_map` from `weight_scale` tensors
2. Unpack 2-bit packed weights: `(data >> shift) & 3 - 1` to get ternary values
3. Reshape from `[N/4, ...]` to `[N, ...]`

**Troubleshooting**:
- **Gibberish output** ("GGGGG..." or nonsense): Used wrong converter or post-hoc quantization
- **Shape mismatch** ("expected 2560, 2560, got 660, 2560, 1, 1"): Packed weights not unpacked
- "tensor out of bounds" error: Model file corrupted or incomplete conversion
- "tokenizer not found": Missing tokenizer.json in checkpoint directory
- "BitnetForCausalLM not found": Need to run sed command to fix architecture name

## Rust Server Build (No Python)

### Manual Build (with -march=native for AVX512)

```bash
# 1. Build llama.cpp with native SIMD
cd extern/sglang-bitnet/3rdparty/llama.cpp
cmake -B build \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_FLAGS="-march=native -mtune=native -O3" \
  -DCMAKE_CXX_FLAGS="-march=native -mtune=native -O3" \
  -DLLAMA_NATIVE=ON -G Ninja
ninja -C build -j$(nproc)
cd ../../../..

# 2. Build Rust gateway with native inference
cd extern/sglang-bitnet/sgl-model-gateway
NATIVE_SIMD=1 RUSTFLAGS="-C target-cpu=native" \
  cargo build --release --features native-inference
cd ../../..

# 3. Run
./scripts/launch_rust_gateway.sh --native
```

**Architecture**: HTTP (Axum/Rust) → C++ SIMD Kernels (no Python)

**Performance**: Scales with CPU frequency - 3.0+ GHz CPUs recommended

## SGLang Quick Start (Supports Continuous Batching)

SGLang is the recommended backend for multi-user scenarios due to **continuous batching** support:
- 1.72x speedup with 4 concurrent requests
- 5.12x speedup with 8 concurrent requests
- 71.3 tok/s total throughput with 16 concurrent requests

```bash
# Install dependencies (use setup script)
./scripts/setup-cpu.sh

# Start SGLang server
./scripts/launch_sglang_bitnet.sh

# Start Streamlit chat UI
uv run streamlit run demo/serve_sglang.py --server.port 7860

# Test via curl
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

### Test Continuous Batching
```bash
# Run batching tests (requires running server)
INFERENCE_URL=http://localhost:30000 uv run pytest tests/test_continuous_batching.py -v
```

## Architecture

```
demo/
└── serve_sglang.py                        # Streamlit chat frontend

scripts/
├── launch_rust_gateway.sh                 # Rust gateway (PRIMARY - no Python)
├── deploy_vultr.sh                        # Deploy to Vultr High Frequency
├── launch_sglang_bitnet.sh                # SGLang Python server (batching)
├── launch_bitnet_cpp.sh                   # BitNet.cpp server (fallback)
├── convert_to_sglkernel.py                # Convert bf16 checkpoints to packed format
├── convert_dlm_to_gguf.py                 # Convert DLM checkpoints to GGUF
├── build-safe.sh                          # Safe build wrapper (4 jobs, 8 cores)
└── reference/                             # Python server implementations (for debugging)
    ├── serve_bitnet_native.py             # Python sgl-kernel server
    └── serve_sglkernel_native.py          # Alternative Python server

deploy/
├── setup_remote_rust.sh                   # Remote setup (builds Rust + llama.cpp)
├── ecosystem_rust.config.js               # pm2 config for Rust server
├── start_rust_server.sh                   # pm2 start script
└── start_streamlit.sh                     # Streamlit start script

src/wrinklefree_inference/
├── sglang_backend/                        # SGLang integration utilities
├── kernels/                               # Kernel wrappers
├── kv_cache/                              # KV cache utilities
├── client/                                # API client
└── moe/                                   # MoE support

extern/
├── sglang-bitnet/                         # SGLang with native BitNet kernels (self-contained)
│   ├── 3rdparty/llama.cpp/                # llama.cpp fork with BitNet support
│   ├── include/                           # BitNet kernel headers
│   ├── src/                               # BitNet kernel sources
│   ├── sgl-model-gateway/                 # Rust HTTP gateway
│   │   └── src/inference/                 # Native C++ FFI bindings
│   ├── sgl-kernel/                        # Native SIMD kernels (AVX2/AVX512)
│   │   └── csrc/inference/                # C++ inference engine
│   └── python/sglang/srt/models/bitnet.py # BitNet model (Python)
└── BitNet/                                # Microsoft BitNet.cpp (reference only, not used by Rust gateway)

legacy/                                    # Archived code (see legacy/README.md)
```

## Key Files

| File | Purpose |
|------|---------|
| `extern/sglang-bitnet/sgl-model-gateway/` | **Primary server** - Rust with C++ inference |
| `scripts/launch_rust_gateway.sh` | Launch Rust server locally |
| `scripts/deploy_vultr.sh` | Deploy to Vultr High Frequency |
| `deploy/setup_remote_rust.sh` | Remote build script with -march=native |
| `demo/serve_sglang.py` | Streamlit chat UI frontend |
| `extern/sglang-bitnet/3rdparty/llama.cpp/` | C++ SIMD inference engine |

## Common Tasks

### Deploy to Vultr High Frequency (Recommended)

Vultr High Frequency instances offer 3+ GHz CPUs at competitive prices.

```bash
# 1. Set API key (one-time)
export VULTR_API_KEY=your_key_here  # Or add to ~/.config/.env.global

# 2. Create instance via CLI
vultr-cli instance create \
  --plan vhf-8c-32gb \
  --region lax \
  --os 2284 \
  --label "bitnet-rust" \
  --ssh-keys "your-ssh-key-id"

# 3. Deploy (builds Rust server with -march=native)
./scripts/deploy_vultr.sh <instance-ip>

# 4. Test
curl http://<ip>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

**Pricing** (High Frequency):
| Plan | vCPU | RAM | Price/hr |
|------|------|-----|----------|
| vhf-4c-16gb | 4 | 16GB | ~$0.14 |
| vhf-8c-32gb | 8 | 32GB | ~$0.29 |

**Cleanup** (stop billing):
```bash
vultr-cli instance delete <instance-id>
```

### Deploy to other clouds
```bash
cd ../deployer
# GCP C3D (production)
sky launch skypilot/inference/gcp_c3d.yaml -y --cluster ie-c3d
# RunPod (development)
sky launch skypilot/inference/runpod_cpu.yaml -y --cluster ie-runpod
```

### Run KV cache validation
```bash
uv run python scripts/validate_kv_cache.py --url http://localhost:30000
```

### Benchmark kernels
```bash
uv run python scripts/benchmark_kernels.py
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Integration tests (requires running server)
INFERENCE_URL=http://localhost:30000 uv run pytest tests/ -v -m integration
```

## sglang-bitnet Setup

**IMPORTANT**: We use a custom fork of SGLang at `extern/sglang-bitnet/`, NOT the upstream sglang package.
This fork includes native SIMD kernels (AVX2/AVX512) for BitNet inference. Do not install sglang from PyPI.

**Run on Desktop**: Heavy builds and server runs should be done on Desktop (ssh Desktop), not locally.

### Full CPU-only Setup (one-time)

```bash
# 1. Initialize submodule
git submodule update --init extern/sglang-bitnet

# 2. Install CPU-only PyTorch
.venv/bin/pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install vllm CPU stub (provides fallback ops without CUDA)
.venv/bin/pip install -e extern/vllm-cpu-stub

# 4. Install sglang from our fork
.venv/bin/pip install -e extern/sglang-bitnet/python

# 5. Build sgl-kernel with BitNet kernels
.venv/bin/pip install scikit-build-core cmake ninja
.venv/bin/pip install -e extern/sglang-bitnet/sgl-kernel --no-build-isolation

# 6. Copy .so to source dir (required for editable install)
cp .venv/lib/python3.12/site-packages/sgl_kernel/common_ops.*.so \
   extern/sglang-bitnet/sgl-kernel/python/sgl_kernel/
```

### Verify BitNet Kernels

```python
from sgl_kernel.quantization import bitnet_check_kernel_available
print(bitnet_check_kernel_available())  # Should be True
```

### Start Server

```bash
.venv/bin/python -m sglang.launch_server \
    --model-path microsoft/bitnet-b1.58-2B-4T \
    --port 30000 --device cpu
```

## Chat Template

BitNet-b1.58-2B-4T uses a simple chat template (from HuggingFace tokenizer_config.json):

```
Role: content<|eot_id|>
```

Example: `System: You are helpful<|eot_id|>User: Hello<|eot_id|>Assistant:`

SGLang automatically applies this template when using the OpenAI-compatible `/v1/chat/completions` endpoint.

## Model Conversion (IMPORTANT for Custom Checkpoints)

When serving custom DLM/BitNet checkpoints (not official microsoft/BitNet models), you may need to convert them:

### Converting bf16 Checkpoints to Packed Format

Use `convert_to_sglkernel.py` to convert bf16/fp16 training checkpoints to optimized 2-bit packed format:

```bash
# Convert checkpoint (reduces size by ~2.5x)
python scripts/convert_to_sglkernel.py models/my-checkpoint models/my-checkpoint.bin

# With quantized lm_head for 2x faster inference (adds ~82MB)
python scripts/convert_to_sglkernel.py models/my-checkpoint models/my-checkpoint.bin --pack-lm-head
```

**Input**: HuggingFace checkpoint directory with `config.json` and `model.safetensors`
**Output**: Binary file with packed 2-bit weights for sgl-kernel

### Weight Formats

| Format | Description | Size | Speed |
|--------|-------------|------|-------|
| bf16 (online) | Training weights, quantized on-the-fly | 4.5GB | ~2 tok/s |
| packed (offline) | Pre-quantized for inference | 1.8GB | ~26 tok/s |
| GGUF | llama.cpp format for BitNet.cpp | 1.1GB | ~26 tok/s |

### Testing Custom Checkpoints

```bash
# Validate conversion
python scripts/test_sglkernel_inference.py --checkpoint models/my-checkpoint

# Test server inference
python scripts/test_sglkernel_inference.py --server-url http://localhost:30000
```

## Notes

### Backend Selection Guide

| Backend | Best For | Decoding | Speed | Python? |
|---------|----------|----------|-------|---------|
| **`dlm_server`** | DLM checkpoints | Block diffusion | **~2.5x faster** | No |
| `native_server` | Regular BitNet | Autoregressive | Baseline | No |
| `batch_server` | Multi-user | Batched AR | Scales with users | No |
| SGLang | Legacy/batching | AR | ~16 tok/s | Yes |

**Recommendations:**
- **`dlm_server`** for DLM checkpoints (the whole point of this project!)
- **`batch_server`** for multi-user production with regular BitNet
- Python servers moved to `scripts/reference/` for debugging only

**Notes:**
- All Rust servers: OpenAI-compatible API (`/v1/chat/completions`)
- All Rust servers: GGUF model format (convert with `convert_dlm_to_gguf.py`)
- Custom SGLang fork: `extern/sglang-bitnet/` (NOT upstream PyPI)
