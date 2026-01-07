# WrinkleFree Inference Engine

> Part of [WrinkleFree Monorepo](https://github.com/DeepOpt-com/WrinkleFree-Monorepo) - Serving layer for 1.58-bit quantized LLMs.

Serving layer for 1.58-bit quantized LLMs using **SGLang-BitNet** with native SIMD kernels (AVX2/AVX512).

## Quick Start

```bash
# Clone the monorepo
git clone --recurse-submodules git@github.com:DeepOpt-com/WrinkleFree-Monorepo.git
cd WrinkleFree-Monorepo/packages/inference

# Install dependencies
uv sync --all-packages

# Build sgl-kernel (one-time setup)
cd extern/sglang-bitnet/sgl-kernel
uv pip install -e . --no-build-isolation
cd ../../..

# Start SGLang server
./scripts/launch_sglang_bitnet.sh

# Start Streamlit chat UI (in another terminal)
uv run streamlit run demo/serve_sglang.py --server.port 7860
```

Access the chat interface at `http://localhost:7860`

## Features

- **SGLang-BitNet backend** - Optimized serving with native SIMD kernels
- **Streamlit chat UI** - Interactive chat interface with streaming
- **OpenAI-compatible API** - Standard `/v1/chat/completions` endpoint
- **HuggingFace integration** - Direct model loading (no conversion needed)

## Architecture

```
demo/
└── serve_sglang.py          # Streamlit chat frontend

scripts/
└── launch_sglang_bitnet.sh  # Server launch script

extern/
└── sglang-bitnet/           # SGLang with BitNet support
    ├── python/sglang/       # SGLang Python package
    ├── sgl-kernel/          # Native SIMD kernels
    └── sgl-model-gateway/   # Rust DLM server (Fast-dLLM v2)

src/wf_infer/
├── sglang_backend/          # SGLang integration utilities
├── kernels/                 # Kernel wrappers
├── kv_cache/                # KV cache utilities
├── client/                  # API client
└── cache/                   # Model caching utilities
```

## API Usage

The server exposes an OpenAI-compatible API:

```bash
# Chat completion
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'

# List models
curl http://localhost:30000/v1/models
```

## Configuration

Environment variables for `launch_sglang_bitnet.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_MODEL` | `microsoft/bitnet-b1.58-2B-4T` | Model to load |
| `SGLANG_PORT` | `30000` | Server port |
| `SGLANG_HOST` | `0.0.0.0` | Server host |

## Kernel Performance

Native SIMD kernels provide significant speedups over PyTorch:

```bash
# Benchmark kernels
uv run python scripts/benchmark_kernels.py
```

- GEMV (batch=1): ~10x faster
- Large dims: ~47x faster

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v
```

## Cloud Deployment

See [WrinkleFree-Deployer](../deployer) for SkyPilot configurations.

## License

MIT
