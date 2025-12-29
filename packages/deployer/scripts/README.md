# Scripts

Automation scripts for WrinkleFree deployment and testing.

## Available Scripts

| Script | Purpose |
|--------|---------|
| `build-image.sh` | Build and push Docker training image to GCR |
| `benchmark_throughput.py` | Benchmark inference server throughput |
| `run_tests.py` | Test runner with structured JSON output |
| `wait_healthy.sh` | Wait for inference server health check |
| `download_test_model.sh` | Download SmolLM2-135M test model |
| `run_smoke_test.sh` | Run smoke tests |
| `setup-nebius.sh` | Set up Nebius cloud credentials |
| `setup-env.sh` | Environment setup |

## Usage

### build-image.sh

Build and push the WrinkleFree training Docker image to GCR.

```bash
# Build and push with auto-generated tag (YYYYMMDD-<git-hash>)
./scripts/build-image.sh

# Build only, don't push
./scripts/build-image.sh --no-push

# Build with custom tag
./scripts/build-image.sh v1.0.0
```

### benchmark_throughput.py

Measure tokens/second at different batch sizes to calculate cost per million tokens.

```bash
# Against local server
uv run python scripts/benchmark_throughput.py --endpoint http://localhost:8080

# Against deployed service
uv run python scripts/benchmark_throughput.py --endpoint https://my-service.sky.serve

# Custom parameters
uv run python scripts/benchmark_throughput.py \
    --endpoint http://localhost:8080 \
    --batch-sizes 1,4,8,16 \
    --duration 30 \
    --max-tokens 100

# JSON output (AI-agent friendly)
uv run python scripts/benchmark_throughput.py --output json
```

### run_tests.py

Test runner with structured JSON output for AI agent consumption.

```bash
# Smoke tests
uv run python scripts/run_tests.py --suite smoke --output json

# All tests
uv run python scripts/run_tests.py --suite all --output json

# Load tests
uv run python scripts/run_tests.py --suite load --duration 60 --output json

# Cloud smoke tests
uv run python scripts/run_tests.py --suite cloud --cloud aws --output json
```

Exit codes:
- `0` - All tests passed
- `1` - Test failures
- `2` - Infrastructure error (service not reachable)
- `3` - Configuration error

### wait_healthy.sh

Wait for an inference server to become healthy.

```bash
# Default timeout (120s)
./scripts/wait_healthy.sh http://localhost:8080

# Custom timeout
./scripts/wait_healthy.sh http://localhost:8080 300
```

### download_test_model.sh

Download SmolLM2-135M for testing (small, fast-loading model).

```bash
# Default location (./models/test)
./scripts/download_test_model.sh

# Custom location
./scripts/download_test_model.sh ./custom/path
```

### setup-env.sh

Set up environment variables and credentials.

```bash
source scripts/setup-env.sh
```

### setup-nebius.sh

Configure Nebius cloud credentials for SkyPilot.

```bash
./scripts/setup-nebius.sh
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run with verbose output
uv run python scripts/benchmark_throughput.py --endpoint http://localhost:8080 --output json
```
