# Contributing to Mobile (wrinklefree-mobile)

> Contributor guide for navigating and understanding the mobile package codebase.

## Quick Orientation

### What This Package Does
Android inference for DLM-BitNet models using TL1 (Tuned Lookup Table v1) ARM NEON optimized kernels, with shared C++ code from the inference package.

### Dependencies

| Depends On | What For |
|------------|----------|
| `inference` | Shared C++ code (sgl-kernel, llama.cpp) |
| Android NDK | Native compilation |
| Kotlin/Compose | Android UI |

---

## Codebase Architecture

### Directory Structure

```
packages/mobile/
├── android/                        # Android Studio project
│   ├── app/                        # Demo chat app
│   │   └── src/main/kotlin/        # Jetpack Compose UI
│   │       └── ChatScreen.kt       # Main chat interface
│   ├── llmcore/                    # JNI library module
│   │   ├── src/main/kotlin/        # Kotlin wrapper
│   │   │   └── LLMEngine.kt        # Kotlin API for native code
│   │   └── src/main/cpp/           # JNI bindings
│   │       ├── llmcore_jni.cpp     # JNI entry points
│   │       └── CMakeLists.txt      # Links to shared inference code
│   ├── build.gradle.kts            # Project build config
│   └── settings.gradle.kts         # Module settings
│
├── models/                         # GGUF model files (gitignored)
│
└── scripts/
    ├── build-android.sh            # Build for ARM64
    └── test-android.sh             # Push and test on device
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Chat UI | `app/src/main/kotlin/` | Jetpack Compose demo app |
| LLMEngine | `llmcore/.../LLMEngine.kt` | Kotlin wrapper for native code |
| JNI bindings | `llmcore/.../llmcore_jni.cpp` | C++ ↔ Kotlin bridge |
| CMakeLists | `llmcore/.../CMakeLists.txt` | Links to inference package C++ |

### Shared Code (No Duplication)

```
packages/mobile/android/llmcore/CMakeLists.txt
│
└─► Links to (NOT copies):
    ├─► ../inference/extern/sglang-bitnet/sgl-kernel/csrc/
    │   └─► Inference engine + ARM TL1 kernels
    │
    └─► ../inference/extern/sglang-bitnet/3rdparty/llama.cpp/
        └─► GGUF loading, tensor operations
```

---

## Code Flow

### Mobile Inference Flow

```
ChatScreen.kt (UI)
│
├─► ChatViewModel.kt
│   └─► LLMEngine.generate(prompt)
│
├─► LLMEngine.kt (Kotlin)
│   └─► native generate() call
│
├─► llmcore_jni.cpp (JNI)
│   └─► Call C++ inference engine
│
└─► ../inference/extern/sglang-bitnet/ (C++)
    └─► GGUF loading + TL1 kernel inference
```

### Build Flow

```
scripts/build-android.sh
│
├─► Configure NDK toolchain for ARM64
│
├─► CMake with -DBITNET_ARM_TL1=ON
│   └─► Enables ARM NEON TL1 kernels
│
├─► Build llama.cpp shared libraries
│   ├─► libllama.so
│   └─► libggml.so
│
└─► Build llama-cli for testing
    └─► Output: build-android/bin/llama-cli
```

---

## Entry Points

| Task | Start Here |
|------|------------|
| Modify Android UI | `android/app/src/main/kotlin/` |
| Change Kotlin API | `android/llmcore/.../LLMEngine.kt` |
| Modify JNI bindings | `android/llmcore/.../llmcore_jni.cpp` |
| Change build config | `android/llmcore/.../CMakeLists.txt` |
| Add build scripts | `scripts/` |

---

## Patterns & Conventions

### JNI Naming Convention

```cpp
// llmcore_jni.cpp
extern "C" JNIEXPORT jstring JNICALL
Java_com_wrinklefree_llmcore_LLMEngine_generate(
    JNIEnv* env,
    jobject /* this */,
    jstring prompt
) {
    // Implementation
}
```

### Library Loading Pattern

```kotlin
// LLMEngine.kt
companion object {
    init {
        System.loadLibrary("llmcore")
    }
}

external fun generate(prompt: String): String
```

### CMake Link Pattern

```cmake
# CMakeLists.txt
# Link to shared inference code (NO duplication)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../inference/extern/sglang-bitnet/3rdparty/llama.cpp
    llama.cpp
)

target_link_libraries(llmcore llama ggml)
```

---

## Testing

### On-Device Testing

```bash
# Build for ARM64
./scripts/build-android.sh

# Push to connected device
adb push build-android/bin/llama-cli /data/local/tmp/bitnet/
adb push build-android/3rdparty/llama.cpp/src/libllama.so /data/local/tmp/bitnet/
adb push build-android/3rdparty/llama.cpp/ggml/src/libggml.so /data/local/tmp/bitnet/
adb push models/ggml-model-i2_s.gguf /data/local/tmp/bitnet/

# Run test
adb shell "cd /data/local/tmp/bitnet && \
  export LD_LIBRARY_PATH=. && \
  ./llama-cli -m ggml-model-i2_s.gguf -p 'Hello' -n 50 --threads 4"
```

### Android Studio

1. Open `packages/mobile/android/` in Android Studio
2. Build and run `app` module on device/emulator

---

## Common Tasks

### Building for Android

```bash
cd packages/mobile

# Ensure Android NDK is installed
# Set ANDROID_NDK_HOME if needed

# Build
./scripts/build-android.sh

# Output:
# - build-android/bin/llama-cli
# - build-android/3rdparty/llama.cpp/src/libllama.so
# - build-android/3rdparty/llama.cpp/ggml/src/libggml.so
```

### Getting a Model

```bash
# Download official BitNet model (recommended for testing)
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf \
  ggml-model-i2_s.gguf \
  --local-dir models/
```

### Custom Model Conversion

```bash
# Convert DLM checkpoint to TL1 GGUF
cd ../inference
python scripts/convert_dlm_to_gguf.py \
  <checkpoint> \
  -o ../mobile/models/model-tl1.gguf \
  --quant tl1
```

---

## Gotchas & Tips

- **TL1 vs TL2**: Mobile uses TL1 (ARM NEON). Server uses TL2 (x86 AVX512). Never mix them.

- **Dimension-Specific Kernels**: TL1 kernels are generated for specific model dimensions. The build uses preset 3B kernels. For custom models, regenerate via BitNet's `setup_env.py`.

- **Shared Libraries**: The JNI module links to `libllama.so` and `libggml.so`. Both must be pushed to the device.

- **LD_LIBRARY_PATH**: When testing via adb, set `LD_LIBRARY_PATH=.` in the same directory as the .so files.

- **Thread Count**: Use `--threads 4` for most devices. More threads can hurt performance on some SoCs.

- **No Code Duplication**: CMakeLists references inference package C++ directly. Don't copy files.

- **Performance Expectations**:
  - Snapdragon 8 Gen 3: 35-60 tok/s
  - Snapdragon 8 Gen 1/2: 25-40 tok/s
  - Snapdragon 7xx: 18-30 tok/s

- **I2_S Format**: Use I2_S GGUF format for mobile. It's the fastest and most compatible.
