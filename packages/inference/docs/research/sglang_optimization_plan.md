# SGLang BitNet Performance Optimization Plan

> **HISTORICAL NOTE**: This document contains research notes from Dec 2025 optimization work.
> Scripts referenced here may have been archived or removed. The current recommended inference
> path uses `dlm_server` with GGUF models. See [CLAUDE.md](../../CLAUDE.md).

## Goal
Close the 1.35x throughput gap: sglang 19.2 tok/s → BitNet.cpp 26 tok/s on AMD CPU.

## Profiling Results (Dec 28, 2025)

### Real-Time Measurements (`scripts/profile_sglang_realtime.py`)
```
Kernel time (measured):       3.3ms → 300 tok/s theoretical
Actual latency (measured):   52.2ms → 19.2 tok/s
Overhead:                    48.9ms (94% of time)

BitNet.cpp target:           38.5ms → 26 tok/s
Gap to close:                13.7ms (26% slower than BitNet.cpp)
```

### Key Comparison
| Backend | Throughput | Latency | vs BitNet.cpp |
|---------|------------|---------|---------------|
| HuggingFace (unpacked) | 5.1 tok/s | 197ms | 5.1x slower |
| SGLang + SIMD kernels | 19.2 tok/s | 52.2ms | 1.35x slower |
| BitNet.cpp | 26.0 tok/s | 38.5ms | Baseline |

**Key insight**: SGLang with SIMD kernels is **3.8x faster** than HuggingFace due to weight packing and SIMD kernels. The remaining gap to BitNet.cpp is Python framework overhead.

### Overhead Breakdown
| Component | Time/token | Notes |
|-----------|-----------|-------|
| Tensor allocation | ~3ms | 30 layers × 2 KV tensors |
| Batch info construction | ~2ms | ForwardBatch setup per layer |
| Python function calls | ~4ms | 50 calls/layer × 30 layers |
| Type conversions | ~12ms | 4 conversions/layer × 30 layers |
| KV cache management | ~6ms | Python indexing (30 layers) |
| Tensor reshapes | ~2ms | 6 reshapes/layer × 30 layers |
| **Subtotal (measured)** | ~29ms | |
| **Unexplained** | ~30ms | HTTP, scheduler, sampling, GIL |

### Key Insight
> [LMDeploy's TurboMind](https://research.aimultiple.com/inference-engines/) achieves **29% higher throughput** via pure C++ engine that eliminates Python overhead entirely.

## Research Summary (Dec 2025)

### Web Search Findings
- [SGLang CPU Backend Optimization (2025 H2)](https://github.com/sgl-project/sglang/issues/8281) - **Closed as completed Oct 2025**
- `torch.compile` graph mode gives ~10% improvement for decoding
- [LMDeploy benchmark](https://research.aimultiple.com/inference-engines/): Pure C++ (TurboMind) matches SGLang+kernels
- sgl-model-gateway is Rust-based but only handles **routing**, not inference

## Aggressive Optimization: 4 Phases

### Phase 1: Enable torch.compile for BitNet (Quick Win)
**Effort**: 1-2 hours | **Expected gain**: 10-15%

Add fake op registrations for BitNet kernels to enable torch.compile.

**File to modify**: `extern/sglang-bitnet/python/sglang/srt/model_executor/cpu_graph_runner.py`

```python
# Add to register_fake_ops():
@torch.library.register_fake("sgl_kernel::bitnet_gemv_cpu")
def _(packed_weights, activations, scale):
    out_features = packed_weights.shape[0]
    return torch.empty(out_features, dtype=torch.float32, device=packed_weights.device)

@torch.library.register_fake("sgl_kernel::bitnet_gemm_cpu")
def _(packed_weights, activations, scale):
    out_features = packed_weights.shape[0]
    batch_size = activations.shape[0]
    return torch.empty(batch_size, out_features, dtype=torch.float32, device=packed_weights.device)

@torch.library.register_fake("sgl_kernel::bitnet_quantize_activations_cpu")
def _(activations):
    return torch.empty_like(activations, dtype=torch.int8), torch.empty(1, dtype=torch.float32)
```

**File to modify**: `scripts/launch_sglang_bitnet.sh`
- Add `--enable-torch-compile` flag **by default**

**Test on Desktop**:
```bash
ssh Desktop "cd /home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine && \
  .venv/bin/python -m sglang.launch_server \
    --model-path microsoft/bitnet-b1.58-2B-4T \
    --port 30000 --device cpu --enable-torch-compile"
```

**Validation**:
```bash
# 1. Smoke test
curl http://Desktop:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "bitnet", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

# 2. Benchmark
python scripts/benchmark_compare.py --sglang-only --quick
```

**Correctness Check (CRITICAL)**:
Compare output with baseline (non-compiled) to ensure torch.compile doesn't change model outputs.

```bash
# Create test script: scripts/test_torch_compile_correctness.py
# 1. Run without torch.compile, save outputs
# 2. Run with torch.compile, save outputs
# 3. Assert outputs are identical (or within tolerance for floating point)

ssh Desktop "cd /home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine && \
  .venv/bin/python -c \"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('microsoft/bitnet-b1.58-2B-4T')
model = AutoModelForCausalLM.from_pretrained('microsoft/bitnet-b1.58-2B-4T', torch_dtype=torch.bfloat16)

# Fixed seed for reproducibility
torch.manual_seed(42)
input_ids = tokenizer.encode('Hello, how are you?', return_tensors='pt')

# Baseline output (greedy)
with torch.no_grad():
    baseline = model.generate(input_ids, max_new_tokens=20, do_sample=False)

print('Baseline:', tokenizer.decode(baseline[0]))
# Save this output to compare against torch.compile version
\""
```

### Note: Continuous Batching
SGLang already supports continuous batching. However, for **CPU single-request latency**:
- Continuous batching helps **throughput** (multiple concurrent users), not per-request latency
- Our focus is single-request latency (16 → 27 tok/s)

---

### Phase 2: C++ Decode Loop (Medium Effort)
**Effort**: 1-2 days | **Expected gain**: 40-50% (→ 22-24 tok/s)

Replace the Python decode loop with C++ using pybind11.

**Target**: Eliminate these overheads:
| Component | Current | Target | Savings |
|-----------|---------|--------|---------|
| Python function calls | 4ms | <0.1ms | 3.9ms |
| Type conversions | 12ms | 2ms | 10ms |
| KV cache management | 6ms | 1ms | 5ms |
| Tensor reshapes | 2ms | 0.2ms | 1.8ms |
| **Total** | 24ms | 3.3ms | **20.7ms** |

**Implementation**:

1. **Create C++ decode wrapper** (`sgl-kernel/csrc/cpu/bitnet_decode.cpp`):
```cpp
class BitNetDecoder {
    // Pre-allocated tensors
    at::Tensor input_ids;
    at::Tensor k_cache, v_cache;
    at::Tensor hidden_states;

public:
    BitNetDecoder(const std::string& model_path, int max_seq_len);

    // Single function call replaces entire Python loop
    at::Tensor decode_step(at::Tensor input_id, int position);
};
```

2. **Python binding** (`sgl-kernel/python/sgl_kernel/decode.py`):
```python
from sgl_kernel import bitnet_decode_step

def fast_decode(model, input_ids, kv_cache):
    """C++ decode loop - replaces Python forward pass."""
    return bitnet_decode_step(model.handle, input_ids, kv_cache.handle)
```

3. **Integration point** (`sglang/srt/model_executor/model_runner.py`):
```python
# Replace:
logits = self.model.forward(input_ids, positions, forward_batch)
# With:
logits = fast_decode(self.model, input_ids, self.kv_cache)
```

**Benchmarks to run after each change**:
```bash
# 1. Baseline (save output for correctness check)
ssh Desktop ".venv/bin/python scripts/benchmark_decode.py --mode baseline --save-output baseline.json"

# 2. After C++ decode loop
ssh Desktop ".venv/bin/python scripts/benchmark_decode.py --mode cpp --compare baseline.json"

# Expected:
# - Throughput: 22-24 tok/s (up from 16)
# - Correctness: Exact match with baseline
```

**Correctness validation**:
```bash
# After EVERY change, compare outputs:
ssh Desktop ".venv/bin/python -c \"
import torch
from test_utils import run_inference

baseline = run_inference(mode='python')
cpp = run_inference(mode='cpp')
assert torch.allclose(baseline.logits, cpp.logits, atol=1e-5), 'Output mismatch!'
print('✓ Correctness verified')
\""
```

---

### Phase 3: Use BitNet.cpp Backend (Fastest Path)
**Effort**: 2-4 hours | **Expected gain**: 70% (→ 27 tok/s)

If C++ decode loop is too complex, **use BitNet.cpp directly**.

**Why**: BitNet.cpp already achieves 27 tok/s with pure C++. No Python overhead.

**Implementation**:

1. **Update Streamlit frontend** to use BitNet.cpp API:
```python
# demo/serve_sglang.py
BACKEND = os.getenv("BITNET_BACKEND", "bitnet_cpp")  # or "sglang"

if BACKEND == "bitnet_cpp":
    API_URL = "http://localhost:8080/v1/chat/completions"
else:
    API_URL = "http://localhost:30000/v1/chat/completions"
```

2. **Update launch script**:
```bash
# scripts/launch_bitnet_cpp.sh
cd extern/BitNet
./build/bin/llama-server \
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    --host 0.0.0.0 --port 8080
```

3. **Benchmark comparison**:
```bash
# sglang baseline
ssh Desktop "curl -s -X POST http://localhost:30000/v1/chat/completions ..."
# Throughput: 16 tok/s

# BitNet.cpp
ssh Desktop "curl -s -X POST http://localhost:8080/v1/chat/completions ..."
# Throughput: 27 tok/s
```

**Correctness check** (outputs should be semantically similar, not identical due to different sampling):
```bash
ssh Desktop ".venv/bin/python scripts/compare_backends.py --prompt 'Hello'"
```

---

### Phase 4: Full C++ Inference Engine (Major Effort)
**Effort**: 1-2 weeks | **Expected gain**: Match BitNet.cpp

If we need sglang features with BitNet.cpp performance, build a hybrid:

1. **C++ inference core** (based on llama.cpp architecture)
2. **Python HTTP wrapper** (thin layer for API compatibility)
3. **Rust tokenizer** (from sgl-model-gateway)

This is the "LMDeploy TurboMind" approach.

**Files to create**:
| File | Purpose |
|------|---------|
| `sgl-kernel/csrc/inference/bitnet_engine.cpp` | Full inference engine in C++ |
| `sgl-kernel/csrc/inference/kv_cache.cpp` | C++ KV cache manager |
| `sgl-kernel/csrc/inference/scheduler.cpp` | C++ batch scheduler |
| `sgl-kernel/python/sgl_kernel/engine.py` | Python wrapper |

**Only pursue this if**:
- Need sglang features (batching, prefix caching, etc.)
- BitNet.cpp doesn't meet feature requirements
- Have dedicated engineering resources

## Files to Modify

| File | Action | Phase |
|------|--------|-------|
| `extern/sglang-bitnet/python/sglang/srt/model_executor/cpu_graph_runner.py` | Add BitNet fake ops | 1 |
| `scripts/launch_sglang_bitnet.sh` | Enable `--enable-torch-compile` by default | 1 |
| `sgl-kernel/csrc/cpu/bitnet_decode.cpp` | C++ decode loop | 2 |
| `sgl-kernel/python/sgl_kernel/decode.py` | Python bindings | 2 |
| `demo/serve_sglang.py` | Add BitNet.cpp backend option | 3 |
| `scripts/launch_bitnet_cpp.sh` | BitNet.cpp server script | 3 |

## Success Criteria

| Phase | Target | Throughput | Improvement | Status |
|-------|--------|------------|-------------|--------|
| Baseline | Current sglang | 19.2 tok/s | - | Measured (Dec 28) |
| Phase 1 | torch.compile | 21 tok/s | +10% | OOM/slow graph capture |
| Phase 2 | C++ decode loop | 24 tok/s | +25% | Not implemented |
| Phase 3 | BitNet.cpp | **26.0 tok/s** | **+35%** | **ACHIEVED** |

### Results (Dec 28, 2025)

#### Updated Measurements
- **SGLang baseline**: 19.2 tok/s (52.2ms/token) - improved from earlier 16 tok/s measurement
- **BitNet.cpp**: 26.0 tok/s (38.5ms/token)
- **Gap**: 13.7ms (26% slower than BitNet.cpp)

#### Phase 1: torch.compile
- Graph capture causes OOM or takes >10 minutes
- Memory drops from 2.8GB to <1GB during capture
- Fixed `.item()` graph breaks in quantization code
- Added BitNet fake ops for torch.compile compatibility
- **Status**: Deferred - impractical for CPU due to memory constraints

#### Phase 3: BitNet.cpp Backend (RECOMMENDED)
- 26.0 tok/s achieved (1.35x faster than SGLang)
- Added `scripts/launch_bitnet_cpp.sh` launch script
- `--cache-reuse 64` enabled by default
- Updated Streamlit frontend with `BITNET_BACKEND` env var
- **Conclusion**: Use BitNet.cpp for maximum performance

#### Phase 4: C++ Fused Operations (Attempted)
- Created `bitnet_mlp_forward_cpu` and `bitnet_qkv_forward_cpu` fused ops
- **Finding**: No speedup - actually 0.63x slower than unfused Python
- Existing SIMD kernels already highly optimized
- Python dispatch overhead is minimal per-op; bottleneck is framework-level
- The 48.9ms overhead is from HTTP/scheduler/event loop, not individual ops

#### Additional Profiling Insights
- HuggingFace pure Python: 5.1 tok/s (197ms/token) - very slow due to weight unpacking
- SGLang SIMD kernels: 3.8x faster than HuggingFace
- Remaining gap (13.7ms) is framework overhead, not kernel performance

### Recommended Production Setup
```bash
# Best performance: BitNet.cpp with cache-reuse (26+ tok/s)
./scripts/launch_bitnet_cpp.sh  # Uses --cache-reuse 64 by default

# If SGLang features needed (batching, prefix caching): 16 tok/s
./scripts/launch_sglang_bitnet.sh

# Streamlit UI (works with both backends)
BITNET_BACKEND=bitnet_cpp uv run streamlit run demo/serve_sglang.py
```

## Benchmark Commands (Run on Desktop)

```bash
# Phase 1: torch.compile
ssh Desktop "cd /home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine && \
  .venv/bin/python -m sglang.launch_server --model-path microsoft/bitnet-b1.58-2B-4T \
  --port 30000 --device cpu --enable-torch-compile &"
sleep 60  # Wait for warmup
curl http://Desktop:30000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Count 1 to 50"}],"max_tokens":100}' | jq

# Phase 2: C++ decode (after implementation)
ssh Desktop ".venv/bin/python scripts/benchmark_decode.py --mode cpp"

# Phase 3: BitNet.cpp
ssh Desktop "cd /home/lev/code/WrinkleFree/packages/inference/extern/BitNet && \
  ./build/bin/llama-server -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --port 8080 &"
sleep 5
curl http://Desktop:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"bitnet","messages":[{"role":"user","content":"Count 1 to 50"}],"max_tokens":100}' | jq
```

## Correctness Validation (After EVERY Phase)

```bash
# Save baseline output (once)
ssh Desktop ".venv/bin/python scripts/correctness_check.py --save-baseline"

# After each phase, compare:
ssh Desktop ".venv/bin/python scripts/correctness_check.py --compare"
# Must output: "✓ Outputs match baseline (atol=1e-5)"
```

## Previous C++ KV Cache Work (Already Done)
Git log shows iterations 1-4 were already implemented:
```
bd1b700ba feat: Add C++ optimized KV cache gather with multi-dtype support
7f6e65e28 feat: Add KV Cache PyTorch bindings and Python wrapper
274621f12 feat: Add AVX-512 optimized gather/scatter (Iteration 2)
1e7143bd9 feat: Add C++ KV Cache Manager skeleton (Iteration 1)
```
This work is complete but Gemini's analysis suggests it targets the wrong bottleneck (<2% of latency).

---

## Original C++ KV Cache Plan (Archived)

The original 10-iteration C++ KV cache plan was already partially implemented (iterations 1-4).
However, Gemini's analysis revealed it targets the wrong bottleneck (<2% of latency).
See git history for implementation details.

---

## Final Conclusion (Dec 28, 2025)

### What We Learned

1. **SGLang SIMD kernels are effective**: 3.8x faster than HuggingFace baseline (19.2 vs 5.1 tok/s)

2. **The remaining gap is framework overhead**: 48.9ms per token comes from:
   - HTTP server/request handling
   - Python scheduler and event loop
   - Tensor allocations for indices/metadata
   - Type conversions and dispatch overhead

3. **Op-level fusion doesn't help**: C++ fused MLP/QKV ops were slower (0.63x) because:
   - Existing SIMD kernels are already highly optimized
   - Adding vector allocation overhead for fused intermediate buffers
   - The bottleneck is at the framework level, not per-op dispatch

4. **torch.compile is impractical on CPU**: Graph capture consumes too much memory and takes too long

### Recommendations

1. **For maximum performance**: Use BitNet.cpp (26 tok/s)
   - Pure C++ inference loop eliminates all Python overhead
   - `--cache-reuse 64` for repeated prompt patterns

2. **For SGLang features**: Accept 19.2 tok/s with current implementation
   - Batching, prefix caching, continuous batching still valuable for multi-user scenarios
   - 26% slower than BitNet.cpp is acceptable trade-off for features

3. **Future optimization paths** (if needed):
   - Implement C++ scheduler bypass for single-request decode
   - Profile and optimize SGLang's async event loop
   - Consider Rust-based HTTP layer (like sgl-model-gateway)

## Sources
- [SGLang CPU Backend Optimization Roadmap](https://github.com/sgl-project/sglang/issues/8281)
- [torch.compile for LLM inference](https://huggingface.co/docs/transformers/llm_optims)
- [Mini-SGLang: Efficient Inference Engine](https://lmsys.org/blog/2025-12-17-minisgl/)
- [Accelerating LLM Inference with TorchAO and SGLang](https://pytorch.org/blog/accelerating-llm-inference/)
