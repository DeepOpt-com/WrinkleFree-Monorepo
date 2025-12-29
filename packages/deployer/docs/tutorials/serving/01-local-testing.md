# Tutorial: Local Testing

Test the entire inference stack on your local machine before deploying to cloud.

**Time:** ~15 minutes
**Cost:** Free
**Requirements:** Docker installed

## What You'll Learn

- How to run the inference server locally
- How to test the API endpoints
- How to run the test suite
- How to debug issues

## Step 1: Verify Docker is Installed

```bash
docker --version
# Should output: Docker version 24.x.x or higher

docker compose version
# Should output: Docker Compose version v2.x.x
```

**If not installed:**
- macOS: `brew install docker` or [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Linux: See [Docker Engine installation](https://docs.docker.com/engine/install/)

## Step 2: Get a Test Model

We'll use SmolLM2-135M - a tiny model perfect for testing.

```bash
# Create models directory
mkdir -p models/test

# Option A: If you have a model from training
cp ../training/outputs/model.gguf models/test/smollm2-135m.gguf

# Option B: Download a test model
./scripts/download_test_model.sh
```

**What's happening:**
- We're getting a small (~100MB) model
- It's quantized to 1.58-bit for fast CPU inference
- Perfect for testing - loads in seconds

## Step 3: Start the Local Server

```bash
# Start the inference server
docker compose -f docker-compose.test.yml up -d

# This runs in the background (-d = detached)
```

**What's happening:**
```
docker-compose.test.yml
    │
    ├── Builds the inference container
    │   └── Uses Dockerfile.bitnet
    │
    ├── Mounts your model
    │   └── models/test → /models
    │
    └── Starts the server
        └── Listening on port 8080
```

## Step 4: Wait for the Server to be Ready

```bash
# Watch the logs
docker compose -f docker-compose.test.yml logs -f

# Or use our helper script
./scripts/wait_healthy.sh http://localhost:8080

# Expected output:
# Waiting for http://localhost:8080 to be healthy...
# Server is healthy after 45s
```

**What to look for in logs:**
```
inference_1  | Loading model...
inference_1  | Model loaded in 12.3s
inference_1  | Server listening on 0.0.0.0:8080  ← Ready!
```

## Step 5: Test the API

### Health Check

```bash
curl http://localhost:8080/health

# Expected response:
# {"status": "healthy"}
```

### Simple Completion

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The meaning of life is",
    "max_tokens": 20,
    "temperature": 0.7
  }'

# Expected response:
# {
#   "id": "completion-xxx",
#   "choices": [{
#     "text": " to find happiness and fulfillment...",
#     "index": 0,
#     "finish_reason": "length"
#   }]
# }
```

### Chat Format (if supported)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50
  }'
```

## Step 6: Run the Test Suite

```bash
# Install test dependencies
uv sync

# Run smoke tests
uv run pytest tests/test_smoke.py -v

# Expected output:
# tests/test_smoke.py::test_health_endpoint PASSED
# tests/test_smoke.py::test_basic_completion PASSED
# ...
# 6 passed in 12.34s
```

### Run with JSON Output (for automation)

```bash
uv run python scripts/run_tests.py --suite smoke --output json

# Output:
# {
#   "suite": "smoke",
#   "total": 6,
#   "passed": 6,
#   "failed": 0,
#   ...
# }
```

## Step 7: Run Load Tests (Optional)

See how the server handles multiple requests:

```bash
# Start Locust load testing
docker compose -f docker-compose.test.yml --profile load up -d

# Open http://localhost:8089 in your browser
# Set:
#   - Number of users: 5
#   - Spawn rate: 1
#   - Host: http://inference:8080
# Click "Start swarming"
```

Or run headless:

```bash
uv run python scripts/run_tests.py --suite load --duration 30 --output json
```

## Step 8: Clean Up

```bash
# Stop and remove containers
docker compose -f docker-compose.test.yml down

# Verify everything is stopped
docker ps
```

## Troubleshooting

### Container won't start

```bash
# Check what's wrong
docker compose -f docker-compose.test.yml logs

# Common issues:
# - Model file not found: Check models/test/ directory
# - Port 8080 in use: Stop other services or change port
```

### Health check keeps failing

```bash
# Check container status
docker compose -f docker-compose.test.yml ps

# Check resource usage
docker stats

# Common issues:
# - Not enough memory: Give Docker more memory in settings
# - Model too large: Use a smaller test model
```

### Slow responses

```bash
# Check CPU usage
docker stats

# Try reducing threads
# Edit docker-compose.test.yml:
environment:
  - NUM_THREADS=2  # Lower number
```

## What's Next?

Now that you've verified everything works locally:

- **Ready for cloud?** → [First Cloud Deployment](02-first-cloud-deployment.md)
- **Want to understand more?** → [Concepts Guide](../concepts.md)
- **Having issues?** → [Troubleshooting](../troubleshooting.md)

## Summary

You've learned how to:

1. ✅ Start a local inference server with Docker
2. ✅ Test the API endpoints
3. ✅ Run the automated test suite
4. ✅ Run load tests
5. ✅ Clean up when done

**Key commands to remember:**
```bash
# Start server
docker compose -f docker-compose.test.yml up -d

# Check health
curl http://localhost:8080/health

# Run tests
uv run pytest tests/test_smoke.py -v

# Stop server
docker compose -f docker-compose.test.yml down
```
