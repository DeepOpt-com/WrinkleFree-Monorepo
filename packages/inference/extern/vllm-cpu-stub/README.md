# vllm-cpu-stub

CPU-only stub package for vllm, providing PyTorch fallback implementations.

## Purpose

SGLang has conditional imports from vllm for certain operations. When running on CPU-only
systems (like BitNet inference), vllm's CUDA dependencies cause import failures. This stub
provides pure PyTorch implementations that satisfy those imports.

## What's Included

- `vllm.model_executor.layers.activation` - SiluAndMul, GeluAndMul, etc.
- `vllm.model_executor.layers.layernorm` - RMSNorm, GemmaRMSNorm
- `vllm._custom_ops` - CPU implementations of custom ops (rms_norm, rotary_embedding, etc.)
- `vllm.distributed.parallel_state` - No-op distributed state for single-device
- `vllm.logger` - Standard Python logger
- `vllm._C` - Stub that raises helpful errors

## Installation

```bash
cd extern/vllm-cpu-stub
uv pip install -e .
```

## Usage with SGLang-BitNet

1. Install this stub BEFORE installing sglang:
   ```bash
   cd extern/vllm-cpu-stub && uv pip install -e .
   cd extern/sglang-bitnet && uv pip install -e python/ -c python/pyproject_cpu.toml
   ```

2. The stub satisfies vllm imports without CUDA dependencies
3. Actual inference uses sgl-kernel's native SIMD kernels (AVX2/AVX512)

## Limitations

- Not a full vllm implementation - only provides components needed by SGLang
- Performance is not optimized - use sgl-kernel for actual inference
- FP8/Marlin quantization ops are stubs - not functional
