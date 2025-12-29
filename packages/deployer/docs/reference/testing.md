# Testing Strategy

Fast, cheap feedback loops for validating the hybrid inference architecture.

## Test Pyramid

```
                    ┌─────────────────────┐
                    │   Production Load   │  ← Hetzner + AWS (expensive, rare)
                    │      Testing        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Staging Tests     │  ← Hetzner Cloud VM (cheap)
                    │   (Single Node)     │
                    └──────────┬──────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │           Integration Tests             │  ← Local Docker
          │   (API, Health, Basic Load)             │
          └────────────────────┬────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────┐
    │                    Unit Tests                        │  ← pytest (instant)
    │   (Config validation, endpoint mocking)              │
    └──────────────────────────────────────────────────────┘
```

## Quick Start: Local Testing

### 1. Start Local Stack

```bash
# Uses SmolLM2-135M (small, fast to load)
docker compose -f docker-compose.test.yml up -d

# Wait for model to load (~60s)
./scripts/wait_healthy.sh http://localhost:8080
```

### 2. Run Tests

```bash
# All tests with JSON output (AI-agent friendly)
uv run pytest tests/ --json-report --json-report-file=results.json

# Smoke test only
uv run pytest tests/test_smoke.py -v

# Load test (60s burst)
uv run locust -f tests/locustfile.py --headless -u 10 -r 2 -t 60s --json
```

### 3. Interpret Results

```bash
# Run tests with JSON output (returns exit code 0/1/2/3)
uv run python scripts/run_tests.py --suite smoke --output json

# Exit codes:
# 0 - All tests passed
# 1 - Test failures
# 2 - Infrastructure error (service not reachable)
# 3 - Configuration error
```

## Test Model

We use **SmolLM2-135M** (1.58-bit quantized) for testing:
- **Size**: ~100MB GGUF
- **Load time**: ~5s on CPU
- **Memory**: ~500MB
- **Inference**: Fast enough for iteration

Download:
```bash
# From HuggingFace (will be automated in setup)
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct-GGUF \
    --local-dir ./models/test
```

## Test Categories

### 1. Smoke Tests (`tests/test_smoke.py`)

Basic functionality validation:
- Health endpoint returns 200
- `/v1/completions` accepts requests
- Response contains expected fields
- Latency under threshold

```python
def test_health_endpoint():
    """Health check returns 200."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200

def test_completion_basic():
    """Basic completion works."""
    response = requests.post(
        f"{BASE_URL}/v1/completions",
        json={"prompt": "Hello", "max_tokens": 10}
    )
    assert response.status_code == 200
    assert "choices" in response.json()
```

### 2. Integration Tests (`tests/test_integration.py`)

End-to-end validation:
- Multiple concurrent requests
- Streaming responses
- Error handling (invalid input)
- Memory stability under load

### 3. Load Tests (`tests/locustfile.py`)

Performance validation:
- Throughput (requests/sec)
- Latency percentiles (p50, p95, p99)
- Error rate under load
- Scaling behavior

### 4. Contract Tests (`tests/test_contract.py`)

API compatibility:
- OpenAI API spec compliance
- Response schema validation
- Header requirements

## AI Agent Integration

All tests output structured JSON for programmatic consumption:

### Test Results Schema

```json
{
  "summary": {
    "total": 15,
    "passed": 14,
    "failed": 1,
    "duration_seconds": 45.2
  },
  "tests": [
    {
      "name": "test_health_endpoint",
      "status": "passed",
      "duration": 0.05
    },
    {
      "name": "test_completion_latency",
      "status": "failed",
      "duration": 2.1,
      "error": "AssertionError: p95 latency 350ms > threshold 200ms",
      "metrics": {
        "p50_ms": 120,
        "p95_ms": 350,
        "p99_ms": 500
      }
    }
  ],
  "load_test": {
    "requests_per_second": 12.5,
    "p50_latency_ms": 80,
    "p95_latency_ms": 150,
    "error_rate": 0.001
  }
}
```

### Running from AI Agents

```bash
# Run tests and get JSON results
uv run python scripts/run_tests.py --output json

# Check if deployment is healthy
uv run python scripts/run_tests.py --suite smoke --output json

# Run load test and get metrics
uv run python scripts/run_tests.py --suite load --duration 60 --output json
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tests passed |
| 1 | Test failures |
| 2 | Infrastructure error (service not reachable) |
| 3 | Configuration error |

## Cloud Smoke Tests (Spot Instance Validation)

Validate that the burst layer works correctly before production deployment.

### Running Cloud Tests

```bash
# Test AWS spot instances (~$0.10-0.30 per run)
uv run python scripts/run_tests.py --suite cloud --cloud aws --output json

# Test GCP spot instances
uv run python scripts/run_tests.py --suite cloud --cloud gcp --output json

# Test Azure spot instances
uv run python scripts/run_tests.py --suite cloud --cloud azure --output json
```

### What Gets Tested

1. **Cloud credentials** - Verifies SkyPilot can access the cloud
2. **Spot launch** - Provisions a small spot instance
3. **Health check** - Waits for instance to become healthy
4. **Inference request** - Sends a test completion request
5. **Termination** - Gracefully tears down the instance

### Prerequisites

```bash
# Install and configure SkyPilot
pip install skypilot[aws,gcp,azure]

# Configure credentials
sky check

# Verify cloud is enabled (should show green checkmark)
sky check | grep -E "AWS|GCP|Azure"
```

### Cost per Test Run

| Cloud | Instance Type | Spot Price | Est. Cost |
|-------|--------------|------------|-----------|
| AWS | r7a.large | ~$0.03/hr | ~$0.10 |
| GCP | n2d-highmem-2 | ~$0.02/hr | ~$0.08 |
| Azure | Standard_E2s_v5 | ~$0.04/hr | ~$0.12 |

*Costs are estimates for ~10 minute test runs*

### AI Agent Usage

```bash
# Run cloud smoke test and get JSON results
uv run python scripts/run_tests.py --suite cloud --cloud aws --output json

# Parse results
cat /tmp/cloud_smoke_results.json | jq '.passed, .failed'
```

### Debugging Failed Cloud Tests

```bash
# Keep instance running for debugging
uv run pytest tests/test_cloud_smoke.py --cloud aws --skip-teardown -v

# SSH into the instance
sky ssh wf-smoke-aws-<timestamp>

# Check logs
sky logs wf-smoke-aws-<timestamp>

# Manual teardown when done
sky down wf-smoke-aws-<timestamp> -y
```

## Cost Optimization

### Local Development (Free)
- Docker Compose with small model
- CPU inference only
- 8GB RAM sufficient

### Staging (€0.03/hour)
- Hetzner Cloud CPX31 (4 vCPU, 8GB RAM)
- ~€0.72/day if always on
- Spin up on demand for testing

### Cloud Smoke (~$0.10-0.30/run)
- Validates spot instance provisioning
- Run before production deployments
- Tests AWS/GCP/Azure individually

### Pre-Production (~€5/test run)
- Single Hetzner AX42 dedicated
- Full model, realistic load
- Run weekly or before releases

## Continuous Integration

```yaml
# .github/workflows/test.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Start test stack
        run: docker compose -f docker-compose.test.yml up -d

      - name: Wait for healthy
        run: ./scripts/wait_healthy.sh http://localhost:8080 120

      - name: Run tests
        run: |
          uv run pytest tests/ \
            --json-report \
            --json-report-file=results.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: results.json
```

## Infrastructure as Code Testing

We follow 2025 best practices for IaC testing without deploying real infrastructure.

### Test Pyramid for Infrastructure

```
                    ┌─────────────────────┐
                    │   Cloud Smoke       │  ← Real AWS/GCP ($0.10-0.30/run)
                    │   (Weekly/Manual)   │
                    └──────────┬──────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │        LocalStack Integration           │  ← Local AWS emulation
          │   (VPC, S3, Security Groups)            │
          └────────────────────┬────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────┐
    │           Terraform Native Tests                     │  ← Plan-only validation
    │   (Variable validation, naming, labels)              │
    └──────────────────────────┬──────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    Static Analysis                           │  ← Instant (no deploy)
│   (TFLint, Checkov, OPA Policies)                            │
└──────────────────────────────────────────────────────────────┘
```

### Quick Start: Infrastructure Testing

```bash
# 1. Static Analysis (instant)
cd terraform/
tflint --init && tflint hetzner/ gcp/ aws/
checkov -d . --config-file ../.checkov.yaml

# 2. Terraform Native Tests (no cloud access needed)
terraform -chdir=hetzner test
terraform -chdir=gcp test
terraform -chdir=aws test

# 3. LocalStack Integration (local Docker)
docker compose -f docker-compose.localstack.yml up -d
uv run pytest tests/integration/ -v -m localstack
docker compose -f docker-compose.localstack.yml down

# 4. Cloud Smoke Tests (real infrastructure, costs money)
uv run pytest tests/test_cloud_smoke.py --cloud gcp -v
```

### Terraform Modules

| Module | Purpose | Tests |
|--------|---------|-------|
| `terraform/hetzner/` | Bare metal base layer | Variable validation, naming, labels |
| `terraform/gcp/` | Spot VMs + GCS storage | Preemptible config, autoscaler, storage |
| `terraform/aws/` | S3 checkpoint storage | Bucket config, IAM, lifecycle |

### Static Analysis Tools

**TFLint** (`terraform/.tflint.hcl`):
- AWS and GCP rulesets
- Naming conventions
- Documentation requirements

**Checkov** (`.checkov.yaml`):
- Security scanning (SSH exposure, encryption)
- Dockerfile scanning
- SARIF output for GitHub Security

**OPA Policies** (`terraform/policies/`):
- `security.rego`: SSH restrictions, required labels
- `cost.rego`: Spot instance enforcement, expensive type warnings

### LocalStack Integration

Test AWS resources locally without cloud access:

```bash
# Start LocalStack + fake-gcs-server
docker compose -f docker-compose.localstack.yml up -d

# Run integration tests
uv run pytest tests/integration/test_terraform_integration.py -v

# Cleanup
docker compose -f docker-compose.localstack.yml down -v
```

### Pytest Markers for Infrastructure

```bash
# Run only LocalStack tests
uv run pytest -m localstack

# Run only Terraform tests
uv run pytest -m terraform

# Run config checks (no external dependencies)
uv run pytest tests/integration/ -k "Config"
```

## Troubleshooting

### Tests timeout waiting for health
```bash
# Check container logs
docker compose -f docker-compose.test.yml logs inference

# Common issues:
# - Model file missing (check volume mounts)
# - Insufficient memory (increase Docker memory limit)
# - Port conflict (change exposed port)
```

### Load tests show high latency
```bash
# Check CPU utilization
docker stats

# Reduce concurrency
uv run locust -f tests/locustfile.py -u 5 -r 1 -t 30s
```

### Terraform tests fail
```bash
# Ensure Terraform 1.6+ is installed
terraform version

# Initialize test fixtures first
cd terraform/hetzner/tests/setup && terraform init && terraform apply -auto-approve

# Run tests with verbose output
terraform -chdir=terraform/hetzner test -verbose
```
