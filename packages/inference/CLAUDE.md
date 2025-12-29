# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree Inference Engine is a serving layer for 1.58-bit quantized LLMs:
- **Primary Backend**: BitNet.cpp (26+ tok/s) or SGLang-BitNet (16 tok/s)
- **Frontend**: Streamlit chat UI with SSE streaming
- **Deployment**: Via WrinkleFree-Deployer (GCP C3D, H3, RunPod)

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

## BitNet.cpp Quick Start (Recommended - 1.6x faster)

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

## Rust Gateway with Native Inference (NEW - Fastest)

Bypasses Python completely for maximum performance.

**Prerequisites**: Build llama.cpp inside sglang-bitnet (gcc/g++ required):

```bash
# Build llama.cpp (one-time) - self-contained in sglang-bitnet
cd extern/sglang-bitnet/3rdparty/llama.cpp
cmake -B build -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build --config Release -j4
cd ../../../..

# Build Rust gateway
cd extern/sglang-bitnet/sgl-model-gateway
cargo build --release --features native-inference -j4
cd ../../..
```

```bash
# Run Rust gateway with native C++ inference
./scripts/launch_rust_gateway.sh --native

# Or with gRPC to Python (fallback)
./scripts/launch_rust_gateway.sh --grpc

# Test
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

**Architecture**: HTTP (Axum/Rust) → C++ SIMD Kernels (no Python)

**Performance**: ~26 tok/s (matches BitNet.cpp, eliminates 49ms Python overhead)

## SGLang Quick Start (Alternative - Slower)

```bash
# Install dependencies
uv sync

# Build sgl-kernel (one-time)
cd extern/sglang-bitnet/sgl-kernel
uv pip install -e . --no-build-isolation
cd ../../..

# Start SGLang server
./scripts/launch_sglang_bitnet.sh

# Start Streamlit chat UI
uv run streamlit run demo/serve_sglang.py --server.port 7860

# Test via curl
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## Architecture

```
demo/
└── serve_sglang.py                        # Streamlit chat frontend

scripts/
├── build-safe.sh                          # Safe build wrapper (4 jobs, 8 cores)
├── launch_rust_gateway.sh                # Rust gateway (native inference)
├── launch_sglang_bitnet.sh               # SGLang Python server
├── launch_bitnet_cpp.sh                  # BitNet.cpp server
├── benchmark_kernels.py                   # Kernel performance testing
├── validate_kv_cache.py                   # KV cache validation
└── test_repacking.py                      # Weight repacking tests

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
| `demo/serve_sglang.py` | Primary Streamlit chat UI |
| `scripts/launch_sglang_bitnet.sh` | SGLang server launch script |
| `extern/sglang-bitnet/python/sglang/srt/models/bitnet.py` | BitNet model with weight packing |
| `extern/sglang-bitnet/sgl-kernel/` | Native SIMD kernels |

## Common Tasks

### Deploy to cloud
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

## Notes

- **Rust Gateway is recommended** for production (26+ tok/s, self-contained in sglang-bitnet)
- **Custom SGLang fork**: We use `extern/sglang-bitnet/` (custom fork with BitNet kernels), NOT upstream sglang
- **Self-contained build**: The Rust gateway builds llama.cpp from `sglang-bitnet/3rdparty/llama.cpp`, no external BitNet.cpp dependency
- **Chat template**: Both BitNet.cpp and SGLang support OpenAI-compatible chat API
- HuggingFace models are automatically packed on-the-fly during loading (sglang only)
- Both servers use OpenAI-compatible API (`/v1/chat/completions`)
- Legacy code (old BitNet.cpp integration, CLI, benchmarks) is in `legacy/`
- `extern/BitNet/` is kept as reference only - not used by the Rust gateway
