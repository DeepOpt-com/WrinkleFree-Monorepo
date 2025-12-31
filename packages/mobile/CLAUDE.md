# Mobile Package

Android inference for DLM-BitNet models using shared C++ code from `packages/inference`.

## Architecture

```
mobile/
├── android/                    # Android Studio project
│   ├── app/                    # Demo chat app (Jetpack Compose)
│   └── llmcore/                # JNI library module
│       ├── src/main/cpp/       # JNI bindings
│       └── CMakeLists.txt      # Links to shared inference code
├── models/                     # GGUF model files (gitignored)
└── scripts/                    # Build and conversion scripts
```

## Quantization Types

| Type | Platform | CMake Flag | Description |
|------|----------|------------|-------------|
| **TL1** | ARM (Android) | `-DBITNET_ARM_TL1=ON` | Tuned Lookup Table v1 for NEON |
| TL2 | x86_64 (Server) | `-DBITNET_X86_TL2=ON` | Tuned Lookup Table v2 for AVX512 |

**For Android, always use TL1 quantization.**

## Key Commands

```bash
# Build native library for Android (with TL1 kernels)
./scripts/build-android.sh

# Test via ADB
adb push models/dlm-bitnet-2b-tl1.gguf /data/local/tmp/
adb shell "cd /data/local/tmp && ./llama-cli -m dlm-bitnet-2b-tl1.gguf -p 'Hello' -n 50"
```

## Shared Code (No Duplication)

This package does NOT copy C++ code. Instead, CMakeLists.txt references:
- `../inference/extern/sglang-bitnet/sgl-kernel/csrc/` - Inference engine + ARM kernels
- `../inference/extern/sglang-bitnet/3rdparty/llama.cpp/` - GGUF model loading

## Model Conversion

Convert DLM checkpoints to GGUF with TL1 quantization for ARM:
```bash
cd ../inference
python scripts/convert_dlm_to_gguf.py <checkpoint> -o ../mobile/models/model-tl1.gguf --quant tl1
```

## Performance

With DLM block diffusion (Fast-dLLM v2):
- Snapdragon 8 Gen 3: 35-60 tok/s
- Snapdragon 8 Gen 1/2: 25-40 tok/s
- Snapdragon 7xx: 18-30 tok/s
