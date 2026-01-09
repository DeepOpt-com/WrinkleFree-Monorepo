# SGLang BitNet Performance Optimization Plan

> **ARCHIVED**: This document contains historical research notes from Dec 2025.
> The sglang-bitnet submodule has been **removed** from the codebase. All file paths
> and scripts referencing `extern/sglang-bitnet/` no longer exist.
>
> **Current inference path**: Use `wf_server` or `dlm_server` from `rust/`.
> See [CLAUDE.md](../../CLAUDE.md) for build and run instructions.

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

**File to modify (REMOVED)**: `extern/sglang-bitnet/python/sglang/srt/model_executor/cpu_graph_runner.py`

> **Note**: This file no longer exists. The sglang-bitnet submodule was removed.
> Code below is preserved for historical reference only.

```python
# Historical - sglang-bitnet removed:
# @torch.library.register_fake("sgl_kernel::bitnet_gemv_cpu")
# def _(packed_weights, activations, scale):
#     out_features = packed_weights.shape[0]
#     return torch.empty(out_features, dtype=torch.float32, device=packed_weights.device)
```

**File to modify (REMOVED)**: `scripts/launch_sglang_bitnet.sh`
- This script was deleted as part of cleanup

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

**Implementation** (Historical - not implemented):

1. **Create C++ decode wrapper** (`sgl-kernel/csrc/cpu/bitnet_decode.cpp` - never created):
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

2. **Python binding** (never created - sglang removed):
```python
# Historical - not implemented
# from sgl_kernel import bitnet_decode_step
```

3. **Integration point** (never implemented - sglang removed):
```python
# Historical reference only - this approach was abandoned
# in favor of the pure Rust wf_server implementation
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

### Phase 3: Use BitNet.cpp Backend (Fastest Path) - SUPERSEDED
**Effort**: 2-4 hours | **Expected gain**: 70% (→ 27 tok/s)

> **Note**: This approach was superseded by the pure Rust `wf_server` implementation,
> which achieves similar performance without requiring the BitNet.cpp submodule.

**Current recommended path**:
```bash
# Build and run wf_server (Pure Rust, ~26 tok/s)
cd rust && cargo build --release --bin wf_server --features native-inference
./target/release/wf_server --model-path ../models/model.gguf --port 30000

# Or dlm_server for DLM models (~60 tok/s)
cargo build --release --bin dlm_server --features llama-inference
./target/release/dlm_server --model-path ../models/dlm-model.gguf --port 30000
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

**Files created (in Rust instead of C++)**:
| File | Purpose |
|------|---------|
| `rust/src/bin/wf_server.rs` | Full inference engine in Rust |
| `rust/src/engine/` | Rust transformer implementation |
| `rust/src/inference/` | DLM scheduler |
| `cpp/llama_engine.cpp` | C++ wrapper for llama.cpp FFI (dlm_server only) |

**Only pursue this if**:
- Need sglang features (batching, prefix caching, etc.)
- BitNet.cpp doesn't meet feature requirements
- Have dedicated engineering resources

## Files to Modify (Historical - see notes)

> **Note**: The sglang-bitnet submodule and related scripts have been removed.
> The Rust inference engine (`rust/`) is the current implementation.

| File | Status | Notes |
|------|--------|-------|
| `extern/sglang-bitnet/...` | REMOVED | Submodule deleted |
| `scripts/launch_sglang_bitnet.sh` | DELETED | Replaced by `scripts/launch_rust_gateway.sh` |
| `scripts/launch_bitnet_cpp.sh` | DELETED | Use wf_server/dlm_server instead |
| `rust/src/bin/wf_server.rs` | CURRENT | Pure Rust inference server |
| `rust/src/bin/dlm_server.rs` | CURRENT | DLM block diffusion server |

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

### Recommended Production Setup (Current)
```bash
# Best performance: wf_server (Pure Rust, ~26 tok/s)
cd rust && cargo build --release --bin wf_server --features native-inference
./target/release/wf_server --model-path ../models/model.gguf --port 30000

# DLM block diffusion (~60 tok/s)
cargo build --release --bin dlm_server --features llama-inference
export LD_LIBRARY_PATH="../extern/llama.cpp/build/src:../extern/llama.cpp/build/ggml/src"
./target/release/dlm_server --model-path ../models/dlm-model.gguf --port 30000
```

## Benchmark Commands (Historical)

> **Note**: These commands reference removed submodules. Current benchmark commands:
>
> ```bash
> # Current: Test wf_server
> cd rust && ./target/release/wf_server --model-path ../models/model.gguf --benchmark
>
> # Current: Test API
> curl http://localhost:30000/v1/chat/completions \
>   -H "Content-Type: application/json" \
>   -d '{"messages":[{"role":"user","content":"Count 1 to 50"}],"max_tokens":100}'
> ```

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

## Final Conclusion (Dec 28, 2025) - UPDATED Jan 2026

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

### Recommendations (Updated Jan 2026)

> **Note**: The sglang-bitnet submodule has been removed. The recommendations below
> have been superseded by the pure Rust implementation.

1. **For maximum performance**: Use `wf_server` (Pure Rust, ~26 tok/s)
   - Eliminates all Python overhead
   - Single binary, no dependencies

2. **For DLM models**: Use `dlm_server` (~60 tok/s)
   - Block diffusion decoding for ~2.5x speedup
   - Requires llama.cpp (downloaded via `scripts/setup_llama_cpp.sh`)

3. **Future optimization paths** (completed):
   - ~~Implement C++ scheduler bypass~~ → Done in Rust
   - ~~Consider Rust-based HTTP layer~~ → wf_server is pure Rust

## Sources
- [SGLang CPU Backend Optimization Roadmap](https://github.com/sgl-project/sglang/issues/8281)
- [torch.compile for LLM inference](https://huggingface.co/docs/transformers/llm_optims)
- [Mini-SGLang: Efficient Inference Engine](https://lmsys.org/blog/2025-12-17-minisgl/)
- [Accelerating LLM Inference with TorchAO and SGLang](https://pytorch.org/blog/accelerating-llm-inference/)
