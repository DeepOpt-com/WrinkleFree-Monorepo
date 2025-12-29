# Docker Images

Container images for WrinkleFree training and inference.

## Available Images

| Image | Purpose | Base Image | Target |
|-------|---------|------------|--------|
| `Dockerfile.train` | Training jobs | `nvidia/cuda:12.4.0-devel-ubuntu22.04` | GCR |
| `Dockerfile.bitnet` | 1.58-bit inference (CPU) | `ubuntu:22.04` | Local/Registry |
| `Dockerfile.vllm` | General LLM serving | `ubuntu:22.04` | Local/Registry |

## Training Image

Pre-built image with all ML dependencies for fast SkyPilot job startup (~30s vs ~10min).

### Build & Push

```bash
# Build and push to GCR (requires gcloud auth)
./scripts/build-image.sh

# Build only (no push)
./scripts/build-image.sh --no-push

# Custom tag
./scripts/build-image.sh v1.0.0
```

### Image Location

```
gcr.io/wrinklefree-481904/wf-train:latest
gcr.io/wrinklefree-481904/wf-train:YYYYMMDD-<git-hash>
```

### Usage in SkyPilot

```yaml
# skypilot/train.yaml
resources:
  image_id: docker:gcr.io/wrinklefree-481904/wf-train:latest
```

### Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Fast HuggingFace downloads |
| `VIRTUAL_ENV` | `/app/.venv` | Pre-built Python 3.11 venv |

### Pre-installed Dependencies

PyTorch 2.5+, Transformers 4.40+, Hydra, bitsandbytes, Accelerate, WandB, google-cloud-storage, torchao, muon-optimizer

## Inference Images

Containers for serving 1.58-bit quantized models.

### Build

```bash
cd WrinkleFree-Deployer

# BitNet (1.58-bit optimized for CPU)
docker build -f docker/Dockerfile.bitnet -t wrinklefree-bitnet .

# vLLM (GPU serving)
docker build -f docker/Dockerfile.vllm -t wrinklefree-vllm .
```

### Run

#### BitNet

```bash
docker run -d \
  --name wrinklefree-inference \
  -p 8080:8080 \
  -v /path/to/models:/models \
  -e MODEL_PATH=/models/model.gguf \
  -e NUM_THREADS=32 \
  -e CONTEXT_SIZE=4096 \
  wrinklefree-bitnet
```

#### vLLM

```bash
docker run -d \
  --name wrinklefree-inference \
  -p 8080:8080 \
  -v /path/to/models:/models \
  -e MODEL_PATH=/models/model \
  -e MAX_MODEL_LEN=4096 \
  wrinklefree-vllm
```

### Environment Variables

#### BitNet

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/model.gguf` | Path to GGUF model |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8080` | Server port |
| `NUM_THREADS` | `0` | CPU threads (0 = auto) |
| `CONTEXT_SIZE` | `4096` | Max context length |

#### vLLM

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/model` | Path to model |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8080` | Server port |
| `MAX_MODEL_LEN` | `4096` | Max context length |
| `TENSOR_PARALLEL_SIZE` | `1` | GPU parallelism |

### API Endpoints

Both backends expose OpenAI-compatible endpoints:

```bash
# Health check
curl http://localhost:8080/health

# Completions
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "max_tokens": 50
  }'

# Chat completions
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### Performance Tuning

#### Memory Locking (Recommended)

Prevent model from being swapped to disk:

```bash
docker run --ulimit memlock=-1:-1 \
  -e MLOCK=true \
  wrinklefree-bitnet
```

#### CPU Affinity

For NUMA-aware scheduling on multi-socket systems:

```bash
docker run --cpuset-cpus="0-47" \
  -e NUM_THREADS=48 \
  wrinklefree-bitnet
```

#### AVX512 Verification

Verify AVX512 is available in container:

```bash
docker run --rm wrinklefree-bitnet \
  grep -o 'avx512[^ ]*' /proc/cpuinfo | sort -u
```

### Compose Example

```yaml
# docker-compose.yml
version: '3.8'

services:
  inference:
    build:
      context: .
      dockerfile: docker/Dockerfile.bitnet
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models:ro
    environment:
      - MODEL_PATH=/models/model.gguf
      - NUM_THREADS=32
      - CONTEXT_SIZE=4096
    deploy:
      resources:
        limits:
          memory: 300G
        reservations:
          memory: 256G
    ulimits:
      memlock:
        soft: -1
        hard: -1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
```

### Multi-Architecture Builds

For ARM64 (Graviton) support:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.bitnet \
  -t wrinklefree-bitnet:multiarch \
  --push .
```
