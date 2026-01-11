# WrinkleFree Mobile

Android inference for BitNet 1.58-bit models.

## Overview

This package provides Android JNI bindings for BitNet.cpp inference, enabling on-device LLM inference with minimal memory footprint.

## Structure

```
mobile/
├── android/          # Android Studio project
│   ├── app/          # Main Android app
│   └── jni/          # JNI bindings to BitNet.cpp
├── scripts/          # Build scripts
└── CLAUDE.md         # AI assistant guidance
```

## Building

See `scripts/` for build automation. Requires:
- Android NDK
- CMake
- BitNet.cpp (from extern/BitNet)

## Related Packages

- `inference`: Full inference engine (desktop/server)
- `training`: Produces models to deploy

## Notes

- This is a pure Android package (no Python src/)
- Uses shared C++ code from extern/BitNet
- Optimized for ARM NEON instructions
