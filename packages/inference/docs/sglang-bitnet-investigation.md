# sglang-bitnet Investigation

## Summary

Investigation into using sgl-kernel's optimized `bitnet_gemv` SIMD kernels for faster DLM inference (target: 29 tok/s vs current 7-8 tok/s with llama.cpp).

**Date:** 2024-12-31
**GCP Instance:** dlm-c3d-8 (c3d-standard-8, AMD EPYC 9B14 with AVX512)

## Current Working Setup

| Component | Value |
|-----------|-------|
| Server | Rust `native_server` (wraps llama.cpp) |
| Model | `dlm-bitnet-2b-tq2.gguf` |
| Format | TQ2_0 (ternary quantization) |
| Port | 30000 |
| Performance | ~7-8 tok/s |
| Output Quality | Coherent |

```bash
# Start server
cd ~/sglang-bitnet/sgl-model-gateway
export LD_LIBRARY_PATH=~/sglang-bitnet/3rdparty/llama.cpp/build/src:~/sglang-bitnet/3rdparty/llama.cpp/build/ggml/src:$LD_LIBRARY_PATH
./target/release/native_server --model-path ~/BitNet-test/models/dlm-bitnet-2b-tq2.gguf --port 30000

# Test
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## sgl-kernel Architecture

The sgl-kernel approach promises ~29 tok/s through:

1. **`bitnet_gemv`** - SIMD-optimized GEMV for single-token decode (8x faster than GEMM)
2. **`bitnet_gemm`** - Cache-optimized batched GEMM for prefill
3. **Block-interleaved packing** - 128-element SIMD blocks for AVX512

### Key Files

| File | Purpose |
|------|---------|
| `sgl-kernel/csrc/bitnet/bitnet_gemv.cpp` | Core SIMD kernels |
| `sgl-kernel/csrc/cpu/bitnet_fused_layer.cpp` | Fused layer operations |
| `sgl-kernel/csrc/cpu/torch_extension_cpu.cpp` | PyTorch op registration |
| `sgl-kernel/python/sgl_kernel/quantization/bitnet.py` | Python bindings |
| `scripts/convert_to_sglkernel.py` | Checkpoint converter |
| `scripts/reference/serve_bitnet_native.py` | Python inference server |

### Model Format

sgl-kernel uses a custom `.bin` format (not GGUF):

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

Weight packing (SIMD block-interleaved):
- Block size: 128 elements (QK_I2_S)
- 32 packed bytes per 128-element block
- `byte[j].bits[6:7]` = weight[j+0]
- `byte[j].bits[4:5]` = weight[j+32]
- `byte[j].bits[2:3]` = weight[j+64]
- `byte[j].bits[0:1]` = weight[j+96]

## Checkpoint Conversion

Successfully converted DLM checkpoint to sgl-kernel format:

```bash
# Convert (4.49GB -> 1.71GB)
python scripts/convert_to_sglkernel.py ~/models/dlm-bitnet-2b ~/models/dlm-bitnet-2b.bin

# Output:
# INFO: Packed 210 linear layers
# INFO: Input:  4.49 GB
# INFO: Output: 1.71 GB
```

## Build Issues

### Environment

- **Python:** 3.10.12
- **PyTorch:** 2.9.1+cpu
- **OS:** Ubuntu (GCP c3d-standard-8)

### Failed Build Attempts

#### 1. Full sgl-kernel build

```
error: 'brgemm' is not a member of 'at::native::cpublas'
error: 'brgemm_release' is not a member of 'at::native::cpublas'
```

Files affected: `gemm.cpp`, `gemm_fp8.cpp`, `bmm.cpp`

#### 2. Minimal build (excluded brgemm files)

```
error: 'struct at::vec::CPU_CAPABILITY::Vectorized<float>' has no member named 'exp_u20'
```

File affected: `norm.cpp`

#### 3. BitNet-only build

Build succeeded but `torch_extension_cpu.cpp` (which registers the ops) has dependencies on all the above files.

### Root Cause

PyTorch 2.9.1 is missing internal APIs that sgl-kernel depends on:
- `at::native::cpublas::brgemm` / `brgemm_release` - Batch GEMM functions
- `Vectorized<float>::exp_u20` - Fast exponential approximation

These APIs likely exist in a different PyTorch version or were renamed/removed.

## Solutions

### Option 1: Find Compatible PyTorch Version

Check which PyTorch version has the required APIs:
```bash
# Search PyTorch source for brgemm
git log --all -p --source -- '**/cpublas*' | grep brgemm
```

### Option 2: Patch sgl-kernel

Remove/stub the brgemm-dependent code paths:
- `gemm.cpp` - Replace brgemm calls with regular gemm
- `bmm.cpp` - Use torch::bmm fallback
- `norm.cpp` - Replace exp_u20 with exp

### Option 3: Cross-compile

Build on a machine with compatible PyTorch and copy the `.so`:
```bash
# On compatible machine
pip install -e sgl-kernel --no-build-isolation
scp sgl_kernel/common_ops.*.so remote:~/.local/lib/python3.10/site-packages/sgl_kernel/
```

### Option 4: Use Python Fallback

The Python fallback works but is ~100x slower (unpacks weights for every forward pass).

## Performance Comparison

| Approach | Speed | Notes |
|----------|-------|-------|
| Rust native_server (llama.cpp TQ2) | ~7-8 tok/s | Working |
| sgl-kernel native (target) | ~29 tok/s | Build blocked |
| sgl-kernel Python fallback | ~0.1 tok/s | Too slow |

## Files on GCP Instance

```
~/models/dlm-bitnet-2b/           # HuggingFace checkpoint
~/models/dlm-bitnet-2b.bin        # sgl-kernel packed format
~/BitNet-test/models/dlm-bitnet-2b-tq2.gguf  # GGUF for llama.cpp
~/sglang-bitnet/                  # sglang-bitnet repo
~/sgl-kernel/                     # sgl-kernel (modified)
~/scripts/                        # Conversion and server scripts
```

## Next Steps

1. Identify PyTorch version with brgemm support
2. Or patch sgl-kernel to remove brgemm dependencies
3. Test with native kernels to verify 29 tok/s target
