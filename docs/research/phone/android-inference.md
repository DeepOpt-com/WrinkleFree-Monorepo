# Research: SGLang-BitNet on Android

## Executive Summary

**Feasibility: Moderate to High** - Our sglang-bitnet implementation is already well-positioned for Android porting because it has ARM NEON kernels. The main work is build system integration and removing Python dependencies.

**Estimated Effort: 2-4 weeks** for a functional prototype

---

## Current Architecture Analysis

### What We Have (Key Files)

| Component | File | Android-Ready? |
|-----------|------|----------------|
| BitNet GEMV kernel | `packages/inference/extern/sglang-bitnet/sgl-kernel/csrc/bitnet/bitnet_gemv.cpp` | **Yes** - ARM NEON implemented |
| Inference engine | `packages/inference/extern/sglang-bitnet/sgl-kernel/csrc/inference/bitnet_engine.cpp` | Yes - pure C++ |
| Weight packing | `packages/inference/src/wrinklefree_inference/kernels/bitnet_patch.py` | Needs C++ port |
| Model loading | llama.cpp integration | Yes - has Android docs |

### ARM NEON Support Already Exists

`bitnet_gemv.cpp` (lines 687-795) already implements:
- ARM NEON 128-bit vector operations
- Optional DotProd extension (ARM8.4+) for faster int8 multiply-accumulate
- Runtime CPU feature detection

```cpp
// Already in our codebase (bitnet_gemv.cpp:687-795)
#if defined(__ARM_NEON)
#if defined(__ARM_FEATURE_DOTPROD)
    // Fast path: uses vdotq_s32 for 4 int8 multiplications per cycle
#else
    // Fallback: vmlal_s8 with manual lane splitting
#endif
#endif
```

---

## Difficulty Assessment by Component

### Easy (Days)
- **Build system**: Add Android NDK CMake toolchain - llama.cpp has [documented this](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md)
- **Inference engine**: `bitnet_engine.cpp` is already pure C++ with no Python deps
- **NEON kernels**: Already implemented, just need testing on real Android

### Medium (1-2 Weeks)
- **JNI bindings**: Create Java/Kotlin interface to C++ engine
- **Weight format**: Port `bitnet_patch.py` repacking logic to C++ (or pre-convert models)
- **Tokenizer**: Either port HuggingFace BPE to C++ or use llama.cpp's tokenizer

### Hard (Unknown)
- **Model loading**: Currently uses HuggingFace Safetensors - need GGUF conversion or custom loader
- **Memory management**: Android memory constraints may require streaming/mmap

---

## Performance Expectations

Based on industry benchmarks and our architecture:

| Device Class | Expected tok/s | Notes |
|--------------|----------------|-------|
| Flagship 2024+ (Snapdragon 8 Gen 3) | 15-25 | DotProd extension helps |
| Mid-range (Snapdragon 7xx series) | 8-15 | Basic NEON |
| Older devices (pre-2020) | 4-8 | May struggle |

**Comparison**: Microsoft's bitnet.cpp achieves 1.37x-5.07x speedups on ARM vs baseline llama.cpp.

**Our advantage**: 1.58-bit models are ~10x smaller than FP16, so the 2B model is only ~500MB - easily fits in RAM on most phones.

---

## Three Approaches

### Option A: Minimal Port (Recommended)

**Effort**: 2-3 weeks

1. Strip Python from inference path entirely
2. Build `bitnet_engine.cpp` + `bitnet_gemv.cpp` with Android NDK
3. Create thin JNI wrapper
4. Pre-convert models to GGUF format on desktop

**Pros**: Fastest path to working demo, reuses our existing optimized kernels
**Cons**: Limited features, manual model conversion

### Option B: Use Microsoft's bitnet.cpp

**Effort**: 1-2 weeks (if they add Android support)

Microsoft's official [BitNet framework](https://github.com/microsoft/BitNet) has ARM support and is "expanding to support mobile devices including iPhone and Android" per their roadmap.

**Pros**: Official implementation, ongoing maintenance
**Cons**: Not yet available for Android, may not support our model format

### Option C: Integrate with Existing Framework

Use [Cactus](https://www.infoq.com/news/2025/12/cactus-on-device-inference/), [llama.cpp Android](https://github.com/JackZeng0208/llama.cpp-android-tutorial), or Google's [MediaPipe LLM](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android).

**Effort**: 3-4 weeks (need to add 1.58-bit quantization support)

**Pros**: Mature frameworks with UI, batching, etc.
**Cons**: None support 1.58-bit natively - would need to contribute custom kernels

---

## Key Technical Challenges

### 1. No Scalar Fallback

Our code explicitly fails on unsupported SIMD:
```cpp
throw std::runtime_error(
    "bitnet_vec_dot_i2_i8: No SIMD support available! "
    "This CPU does not support AVX2, AVX512, or NEON."
);
```

All ARM64 devices (Android API 21+) support NEON, so this is fine.

### 2. Weight Format Conversion

Current flow: HuggingFace Safetensors -> Python repacking -> kernel format

For Android: Need to either:
- Pre-convert on desktop, ship converted weights
- Port `bitnet_patch.py` logic (~257 lines) to C++

### 3. Tokenizer

Currently depends on `transformers` library. Options:
- Port tokenizer to C++ (SentencePiece has native C++ library)
- Use llama.cpp's built-in tokenizer
- Ship pre-tokenized vocabulary with the app

### 4. Memory Constraints

Most Android devices have 4-8GB RAM, but apps typically get ~1-2GB.
- Our 2B model at 1.58-bit: ~500MB weights
- KV cache: ~50-100MB depending on context length
- Total: ~600-700MB - **should work**

---

## Android Build Quick Start

Based on [llama.cpp Android docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md):

```bash
# Set up Android NDK
export ANDROID_NDK=/path/to/ndk

# Configure CMake
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_C_FLAGS="-march=armv8.4a+dotprod" \
  -DCMAKE_CXX_FLAGS="-march=armv8.4a+dotprod" \
  -DGGML_OPENMP=OFF \
  -B build-android

# Build
cmake --build build-android --config Release -j$(nproc)
```

---

## Recommended Next Steps

If we want to pursue this:

1. **Validate NEON kernels**: Build just `bitnet_gemv.cpp` for Android and run unit tests
2. **Profile on real device**: Get baseline tok/s on target hardware
3. **Decide on model format**: GGUF (llama.cpp) vs custom Safetensors loader
4. **Build minimal demo**: CLI inference on Android via ADB before adding UI

---

## Sources

- [llama.cpp Android Build Docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md)
- [Microsoft BitNet Repository](https://github.com/microsoft/BitNet)
- [BitNet b1.58 2B4T on HuggingFace](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [Cactus v1: Cross-Platform LLM Inference](https://www.infoq.com/news/2025/12/cactus-on-device-inference/)
- [Google AI Edge LLM Inference](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android)
- [llama.cpp Android Tutorial](https://github.com/JackZeng0208/llama.cpp-android-tutorial)
- [PowerInfer-2: Fast LLM on Smartphone](https://arxiv.org/html/2406.06282v3)
