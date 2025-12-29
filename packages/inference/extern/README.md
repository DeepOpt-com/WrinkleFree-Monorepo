# External Dependencies

## Installation Order (CPU-Only)

For CPU-only BitNet inference, install in this order:

```bash
# 1. Install vllm-cpu-stub first (satisfies vllm imports without CUDA)
cd vllm-cpu-stub && uv pip install -e .

# 2. Build sgl-kernel with CPU SIMD support
cd sglang-bitnet/sgl-kernel && uv pip install -e . --no-build-isolation

# 3. Install sglang with CPU config
cd sglang-bitnet && uv pip install -e python/
```

## sglang-bitnet (Primary)

Fork of SGLang with BitNet kernel support. **This is the primary serving backend.**

- Native SIMD kernels (AVX2/AVX512) for 1.58-bit inference
- OpenAI-compatible API
- Streaming support
- CPU-optimized

## vllm-cpu-stub

CPU-only stub for vllm that provides PyTorch fallback implementations.

SGLang has conditional imports from vllm. On CPU-only systems, vllm's CUDA dependencies
cause import failures. This stub provides pure PyTorch implementations that satisfy
those imports without requiring CUDA.

**Includes:**
- `vllm.model_executor.layers.activation` - SiluAndMul, GeluAndMul
- `vllm.model_executor.layers.layernorm` - RMSNorm, GemmaRMSNorm
- `vllm._custom_ops` - CPU implementations of custom ops
- `vllm.distributed.parallel_state` - No-op distributed state

## BitNet (Reference Only)

Microsoft's BitNet.cpp implementation. **Reference implementation only - do not serve directly.**

- Based on llama.cpp
- GGUF model format
- Useful for model conversion and benchmarking

The SGLang-BitNet backend is preferred for production serving because:
1. Native Python integration
2. Better memory management
3. OpenAI-compatible API
4. Streaming support
