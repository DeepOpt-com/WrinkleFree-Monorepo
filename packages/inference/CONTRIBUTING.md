# Contributing to Inference (wrinklefree-inference)

> Contributor guide for navigating and understanding the inference package codebase.

## Quick Orientation

### What This Package Does
GGUF conversion and DLM (Diffusion Language Model) serving for 1.58-bit BitNet models using Rust's Fast-dLLM v2 block diffusion.

### Dependencies

| Depends On | What For |
|------------|----------|
| llama.cpp (submodule) | GGUF format, tensor operations |
| Rust/Cargo | dlm_server binary |
| gguf-py | Python GGUF manipulation |

| Used By | What For |
|---------|----------|
| `eval` | Model evaluation via inference API |
| `mobile` | Shares C++ code for mobile inference |
| `deployer` | Deploys inference servers |

---

## Codebase Architecture

### Directory Structure

```
packages/inference/
├── scripts/
│   ├── convert_checkpoint_to_gguf.py  # PRIMARY converter
│   ├── convert_to_sglkernel.py        # sgl-kernel packed format
│   ├── launch_rust_gateway.sh         # Start dlm_server
│   └── serve.sh                       # Full stack launcher
│
├── src/wrinklefree_inference/
│   ├── client/
│   │   └── bitnet_client.py           # Python client for inference API
│   ├── cache/
│   │   ├── loader.py                  # Model cache management
│   │   ├── gcs_client.py              # GCS model download
│   │   └── bitnet_converter.py        # Conversion utilities
│   ├── kv_cache/
│   │   └── kv_cache.py                # KV cache management
│   ├── kernels/
│   │   └── bitnet_patch.py            # Kernel patches
│   └── sglang_backend/
│       └── bitnet_quantization.py     # SGLang integration
│
├── demo/
│   └── serve_sglang.py                # Streamlit chat UI
│
└── extern/
    └── sglang-bitnet/
        ├── 3rdparty/llama.cpp/        # Core inference (submodule)
        │   ├── build/                 # Compiled binaries
        │   ├── convert_hf_to_gguf.py  # Generic converter
        │   ├── gguf-py/               # GGUF Python library
        │   ├── src/                   # llama.cpp core
        │   └── ggml/                  # GGML tensor library
        └── sgl-model-gateway/         # Rust HTTP server
            └── src/bin/dlm_server.rs  # DLM server entry point
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| GGUF Converter | `scripts/convert_checkpoint_to_gguf.py` | Training checkpoint → GGUF |
| dlm_server | `extern/sglang-bitnet/sgl-model-gateway/` | Rust HTTP server for DLM inference |
| BitNetClient | `src/wrinklefree_inference/client/bitnet_client.py` | Python API client |
| llama.cpp | `extern/sglang-bitnet/3rdparty/llama.cpp/` | GGUF loading, tensor ops |

---

## Code Flow

### Checkpoint to Serving Pipeline

```
Training checkpoint (PyTorch)
│
├─► scripts/convert_checkpoint_to_gguf.py
│   │
│   ├─► Fix architecture name (BitnetForCausalLM → LlamaForCausalLM)
│   │
│   ├─► Call llama.cpp/convert_hf_to_gguf.py
│   │
│   └─► Output: model.gguf (F16)
│
├─► (Optional) llama-quantize model.gguf model-i2s.gguf I2_S
│
└─► dlm_server --model-path model-i2s.gguf
    │
    ├─► Load GGUF via llama.cpp
    │
    └─► HTTP API at :30000
        └─► /v1/chat/completions (OpenAI-compatible)
```

### DLM Inference Flow (dlm_server)

```
HTTP Request
│
├─► Tokenize input
│
├─► Block Diffusion (Fast-dLLM v2):
│   │
│   ├─► Initialize block with mask tokens
│   │
│   ├─► For each iteration:
│   │   ├─► Forward pass (predict logits)
│   │   ├─► Sample tokens for low-confidence positions
│   │   └─► Check convergence (threshold)
│   │
│   └─► Return decoded block
│
├─► Repeat for each block
│
└─► HTTP Response (streaming supported)
```

---

## Entry Points

| Task | Start Here |
|------|------------|
| Modify GGUF conversion | `scripts/convert_checkpoint_to_gguf.py` |
| Change server behavior | `extern/sglang-bitnet/sgl-model-gateway/src/bin/dlm_server.rs` |
| Add Python API features | `src/wrinklefree_inference/client/bitnet_client.py` |
| Modify llama.cpp | `extern/sglang-bitnet/3rdparty/llama.cpp/` (submodule) |
| Change demo UI | `demo/serve_sglang.py` |

---

## Patterns & Conventions

### GGUF Conversion Pattern

The converter script wraps llama.cpp's converter with fixes:
```python
# scripts/convert_checkpoint_to_gguf.py

# 1. Fix architecture name in config.json
fix_architecture_name(checkpoint_dir)  # BitnetForCausalLM → LlamaForCausalLM

# 2. Call llama.cpp converter
subprocess.run([
    sys.executable,
    "extern/sglang-bitnet/3rdparty/llama.cpp/convert_hf_to_gguf.py",
    checkpoint_dir,
    "--outfile", output_path,
    "--outtype", output_type,
])
```

### DLM Server API Pattern

OpenAI-compatible API:
```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### LD_LIBRARY_PATH Pattern

dlm_server needs llama.cpp shared libraries:
```bash
export LD_LIBRARY_PATH="extern/sglang-bitnet/3rdparty/llama.cpp/build/src:extern/sglang-bitnet/3rdparty/llama.cpp/build/ggml/src:$LD_LIBRARY_PATH"

./extern/sglang-bitnet/sgl-model-gateway/target/release/dlm_server \
  --model-path model.gguf
```

---

## Testing

### Running Tests

```bash
# Unit tests
uv run --package wrinklefree-inference pytest packages/inference/tests/ -v

# Integration tests (requires running dlm_server)
INFERENCE_URL=http://localhost:30000 \
  uv run --package wrinklefree-inference pytest packages/inference/tests/ -v -m integration
```

### Manual Testing

```bash
# Test GGUF conversion
python scripts/convert_checkpoint_to_gguf.py path/to/checkpoint -o test.gguf --dry-run

# Test API
curl http://localhost:30000/health
curl http://localhost:30000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Test"}]}'
```

---

## Common Tasks

### Adding a New GGUF Output Type

1. Edit `scripts/convert_checkpoint_to_gguf.py`
2. Add to output type validation
3. Add any preprocessing needed for the type
4. Test with a small model

### Modifying the DLM Server

1. Edit `extern/sglang-bitnet/sgl-model-gateway/src/bin/dlm_server.rs`
2. Rebuild: `cargo build --release --bin dlm_server --features=native-inference`
3. Test with a model

### Building llama.cpp from Scratch

```bash
cd extern/sglang-bitnet/3rdparty/llama.cpp

# Clean previous build
rm -rf build

# Configure
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=ON

# Build
cmake --build build -j4

# Verify
ls build/bin/llama-cli build/bin/llama-quantize
ls build/src/libllama.so build/ggml/src/libggml.so
```

---

## Gotchas & Tips

- **TQ2_0 Corruption**: NEVER use TQ2_0 format for bf16 DLM checkpoints. It corrupts ternary weight distribution. Use I2_S instead.

- **LD_LIBRARY_PATH**: dlm_server requires llama.cpp shared libraries. Always set `LD_LIBRARY_PATH` before running.

- **Autoregressive Decoders**: DO NOT use llama-cli or native_server with DLM models. They produce garbage output. DLM requires block diffusion decoding.

- **Mask Token ID**: DLM training and inference must use the same mask_token_id (typically 0). Mismatch causes bad output.

- **Submodule Updates**: llama.cpp is a git submodule. After pulling, run `git submodule update --init --recursive`.

- **Build Throttling**: Use `-j4` instead of `-j$(nproc)` when building llama.cpp to prevent system freeze.

- **Architecture Name**: WrinkleFree checkpoints use "BitnetForCausalLM" but llama.cpp expects "LlamaForCausalLM". The converter fixes this automatically.

- **I2_S Recommended**: For production, convert to F16 first, then quantize to I2_S. This gives best speed and size.
