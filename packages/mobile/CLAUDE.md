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
# Build BitNet for Android (ARM64)
./scripts/build-android.sh

# Download official BitNet 2B model (1.2GB)
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf ggml-model-i2_s.gguf --local-dir models/

# Test on connected Android device
./scripts/test-android.sh

# Or manually via ADB:
adb push build-android/bin/llama-cli /data/local/tmp/bitnet/
adb push build-android/3rdparty/llama.cpp/src/libllama.so /data/local/tmp/bitnet/
adb push build-android/3rdparty/llama.cpp/ggml/src/libggml.so /data/local/tmp/bitnet/
adb push models/ggml-model-i2_s.gguf /data/local/tmp/bitnet/
adb shell "cd /data/local/tmp/bitnet && export LD_LIBRARY_PATH=. && ./llama-cli -m ggml-model-i2_s.gguf -p 'Hello' -n 50 --threads 4"
```

## Shared Code (No Duplication)

This package does NOT copy C++ code. Instead, CMakeLists.txt references:
- `../inference/extern/sglang-bitnet/sgl-kernel/csrc/` - Inference engine + ARM kernels
- `../inference/extern/sglang-bitnet/3rdparty/llama.cpp/` - GGUF model loading

## Models

### Official BitNet Model (Recommended for testing)
```bash
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf ggml-model-i2_s.gguf --local-dir models/
```

### Custom DLM Model Conversion
To convert DLM checkpoints to GGUF (requires kernel generation for TL1):
```bash
cd ../inference
python scripts/convert_dlm_to_gguf.py <checkpoint> -o ../mobile/models/model-tl1.gguf --quant tl1
```

Note: TL1 kernels are model-dimension specific. The build currently uses preset 3B kernels.
For custom models, generate kernels via BitNet's `setup_env.py`.

## Performance

With DLM block diffusion (Fast-dLLM v2):
- Snapdragon 8 Gen 3: 35-60 tok/s
- Snapdragon 8 Gen 1/2: 25-40 tok/s
- Snapdragon 7xx: 18-30 tok/s
