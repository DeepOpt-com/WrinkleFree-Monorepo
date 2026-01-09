# External Dependencies

This directory contains external dependencies downloaded on-demand.

## llama.cpp

Required for:
- GGUF model conversion (`scripts/convert_checkpoint_to_gguf.py`)
- DLM block diffusion server (`dlm_server` with `llama-inference` feature)

### Setup

```bash
./scripts/setup_llama_cpp.sh
```

This clones and builds llama.cpp with shared libraries.

### Manual Setup

```bash
git clone --depth 1 --branch merge-dev \
    https://github.com/Eddie-Wang1120/llama.cpp.git extern/llama.cpp
cd extern/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build -j$(nproc)
```

## Note

This directory is not committed to git. Dependencies are downloaded as needed.
