# BitNet Native Inference Guide

This document describes how to run BitNet 1.58-bit models using native SIMD kernels.

## TL;DR - Quick Start (Python Server)

The **recommended** approach is the Python native server which achieves **29 tok/s**:

```bash
# Step 1: Convert checkpoint (one-time)
python scripts/convert_to_sglkernel.py models/my-checkpoint models/my-checkpoint.bin

# Step 2: Start server
python scripts/serve_bitnet_native.py \
    --model models/my-checkpoint.bin \
    --tokenizer models/my-checkpoint

# Step 3: Test
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## Background: Why Native Inference?

The standard BitNet.cpp and llama.cpp paths have **known issues with 2B models**:
- I2_S quantization not properly supported in llama.cpp (#12997)
- TL2 format produces garbage output for 2B architecture
- Kernel dimensions are hardcoded for 3B model

**Two native inference paths are available:**

### 1. Python Server (RECOMMENDED) - `serve_bitnet_native.py`
- **Performance**: ~29 tok/s
- **Key optimizations**:
  - `bitnet_gemv` for single-token decode (8x faster than gemm)
  - Greedy decoding by default
  - Repetition penalty
- **Simplest setup**: Just Python + sgl-kernel

### 2. C++ Engine - `bitnet_engine.cpp`
- **Performance**: ~40+ tok/s (theoretical)
- **More complex**: Requires Rust + C++ build
- Uses sgl-kernel SIMD kernels via FFI

## Python Server Setup (Recommended)

### 1. Convert Checkpoint

```bash
# From packages/inference directory
cd packages/inference

# Convert DLM checkpoint to sgl-kernel format
python scripts/convert_to_sglkernel.py \
    /path/to/checkpoint \
    /path/to/output.bin
```

The converter handles both:
- **Online mode**: bf16 weights → quantize + pack
- **Offline mode**: pre-packed weights → repack for sgl-kernel format

### 2. Build Native Inference Engine

```bash
cd extern/sglang-bitnet/sgl-kernel/csrc/inference

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### 3. Build Rust Server

```bash
cd extern/sglang-bitnet/sgl-model-gateway

cargo build --release --features native-inference
```

### 4. Run Server

```bash
./target/release/native_server \
    --model-path /path/to/model.bin \
    --port 30000
```

### 5. Test

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## Weight Format

### sgl-kernel Binary Format (.bin)

```
[8 bytes]  Magic: "SGLBITNT"
[4 bytes]  Version: 1
[4 bytes]  Config JSON length
[N bytes]  Config JSON
[4 bytes]  Number of tensors
For each tensor:
    [4 bytes]  Name length
    [N bytes]  Name (UTF-8)
    [4 bytes]  Dtype (0=uint8, 1=float32, 2=float16, 3=bfloat16)
    [4 bytes]  Number of dimensions
    [dims x 4] Shape
    [4 bytes]  Scale present flag
    [4 bytes]  Scale value (float32, if present)
    [8 bytes]  Data size in bytes
    [N bytes]  Raw tensor data
```

### Weight Packing (I2_I8)

Linear layer weights are packed as:
- **Shape**: [out_features, in_features/4]
- **Encoding**: 4 ternary values per byte along K (input) dimension
- **Values**: 00=-1, 01=0, 10=+1
- **Block size**: K must be multiple of 128 (QK_I2_S)

## Architecture

```
                    ┌─────────────────────────┐
                    │    Rust HTTP Server     │
                    │   (native_server.rs)    │
                    └───────────┬─────────────┘
                                │ FFI
                    ┌───────────▼─────────────┐
                    │   BitNet Engine (C++)   │
                    │   (bitnet_engine.cpp)   │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
┌─────────▼─────────┐ ┌────────▼────────┐ ┌─────────▼─────────┐
│  sgl-kernel SIMD  │ │   KV Cache      │ │  Model Loader     │
│  (bitnet_gemv.cpp)│ │  (kv_cache.cpp) │ │(sglkernel_loader.h)│
└───────────────────┘ └─────────────────┘ └───────────────────┘
```

## Performance

Expected throughput on c3d-standard-30 (30 vCPUs, DDR5):
- **Target**: > 40 TPS
- **Typical**: 26-47 TPS depending on prompt length

Key factors:
- Memory bandwidth (DDR5 helps significantly)
- AVX512 support (check with `cat /proc/cpuinfo | grep avx512`)
- Thread count (set via `--threads` flag)

## Known Issues

### 1. BitNet.cpp TL2/I2_S Broken for 2B

**Symptoms**: Garbage output from `llama-server` or `llama-cli`

**Root cause**: BitNet.cpp's kernel dimensions are hardcoded for 3B model (3200x8640) but DLM 2B uses 2560x6912.

**Solution**: Use the native inference path instead of llama.cpp.

### 2. Weight Scale Application

**Issue**: Online vs offline quantization requires different handling during GGUF conversion.

**Solution**: Our converter auto-detects the mode from `config.json`:
- `quantization_mode: "online"` → quantize + pack
- `quantization_mode: "offline"` → unpack + repack

### 3. Python Overhead

**Issue**: Python SGLang path works but has ~49ms overhead per request.

**Solution**: Native C++ inference eliminates Python entirely.

## Comparison: Inference Backends

| Backend | Format | Works for 2B? | TPS | Notes |
|---------|--------|---------------|-----|-------|
| **Python Native** | **.bin** | **Yes** | **29** | **Recommended - simplest** |
| C++ Native | .bin | Yes | 40+ | More complex setup |
| BitNet.cpp | GGUF TL2 | No | N/A | Kernel dim mismatch |
| llama.cpp | GGUF I2_S | No | N/A | Not supported |
| Python SGLang | safetensors | Yes | ~16 | Higher overhead |

## Files

| File | Purpose |
|------|---------|
| `scripts/serve_bitnet_native.py` | **Python native server (29 tok/s)** |
| `scripts/convert_to_sglkernel.py` | Convert safetensors → .bin |
| `scripts/test_sglkernel_inference.py` | Validation test script |
| `extern/sglang-bitnet/sgl-kernel/` | SIMD kernels (bitnet_gemv, bitnet_gemm) |
| `extern/sglang-bitnet/sgl-kernel/csrc/inference/bitnet_engine.cpp` | C++ inference engine (alternative) |

## Troubleshooting

### "Tensor not found" error
Check tensor naming convention. DLM models use:
- `model.layers.N.self_attn.q_proj.weight`
- `model.layers.N.mlp.gate_proj.weight`

### Slow inference
1. Check AVX512 support: `cat /proc/cpuinfo | grep avx512`
2. Verify thread count: `--threads 30` for 30 vCPU machine
3. Check memory bandwidth (DDR5 > DDR4)

### NaN in output
Fixed in commit cbc55cd. Ensure you have the latest code.

### Build errors
Ensure you have:
- GCC 11+ or Clang 14+
- CMake 3.18+
- Rust 1.75+
