# Inference Deployment Configs

SkyPilot configurations for deploying BitNet 1.58-bit inference servers on high-bandwidth CPU instances.

## Performance Overview

BitNet inference is **memory-bandwidth bound** - throughput scales linearly with:
1. Memory bandwidth (DDR5 >> DDR4)
2. Thread count (until bandwidth saturates)

### Benchmark Results

| Instance | Model | vCPUs | Throughput | Latency | Cost/hr |
|----------|-------|-------|------------|---------|---------|
| c3d-standard-16 | Falcon3-7B-1.58bit | 16 | **12.6 tok/s** | 79 ms/tok | $0.73 |
| c3d-standard-90 | Falcon3-7B-1.58bit | 90 | ~60-70 tok/s* | ~15 ms/tok* | $4.00 |
| h3-standard-88 | BitNet-2B-4T | 88 | ~40-50 tok/s* | ~20 ms/tok* | $1.76 |

*Estimated based on linear scaling with thread count. Actual results may vary.

**Benchmark details (c3d-standard-16, 2024-12-23):**
```
Model: Falcon3-7B-Instruct-1.58bit (7.46B params, 3.05 GiB)
Quantization: I2_S (2 bpw ternary)
CPU: AMD EPYC 9B14 (Genoa, 4th Gen)
Memory: 64 GB DDR5

Results:
- Prompt eval: 13.15 tok/s (76 ms/token, 5 tokens)
- Generation:  12.64 tok/s (79 ms/token, 92 tokens)
- Model load:  674 ms
- Total time:  7.69s for 97 tokens
```

## Available Instances

| Config | Instance | CPU | Memory BW | vCPUs | RAM | Cost/hr |
|--------|----------|-----|-----------|-------|-----|---------|
| `gcp_c3d.yaml` | c3d-standard-90 | AMD EPYC Genoa | ~460 GB/s | 90 | 360 GB | ~$4.00 |
| `gcp_h3.yaml` | h3-standard-88 | Intel Sapphire | ~307 GB/s | 88 | 352 GB | ~$1.76 |
| `runpod_cpu.yaml` | A40 host | Varies | ~696 GB/s | 16+ | 128+ GB | ~$0.80 |

### Instance Selection Guide

- **GCP C3D** (Recommended): Best memory bandwidth for BitNet. AMD DDR5 with AVX-512.
- **GCP H3**: Lower cost Intel option. Good price/performance balance.
- **RunPod**: Cheapest for development. Limited to 40GB disk.

## Why C3D instead of C4D?

GCP's newest C4D instances (AMD EPYC Turin, 5th Gen) offer slightly better performance,
but they **only support Hyperdisk** - not traditional Persistent Disk (pd-ssd, pd-balanced).

SkyPilot doesn't support Hyperdisk yet ([issue #4705](https://github.com/skypilot-org/skypilot/issues/4705)).
Once SkyPilot adds Hyperdisk support, we can switch to C4D for ~10% better performance.

C3D (AMD EPYC Genoa, 4th Gen) is still excellent - same DDR5 memory bandwidth (~460 GB/s).

## GCP Quota Requirements

Default GCP quotas may limit instance sizes. Check/request increases:

| Quota | Default | Required for c3d-90 | How to Check |
|-------|---------|---------------------|--------------|
| `CPUS_ALL_REGIONS` | 32 | 90+ | GCP Console → IAM & Admin → Quotas |
| `CPUS_PER_VM_FAMILY` | 24 | 90+ | Filter by "C3D" |

To request quota increase:
1. Go to [GCP Quotas](https://console.cloud.google.com/iam-admin/quotas)
2. Filter: `Service: Compute Engine API`, `Quota: C3D`
3. Select quota → Edit Quotas → Request increase

## Usage

### Quick Start

```bash
# GCP C3D (recommended for production)
sky launch skypilot/inference/gcp_c3d.yaml -y --cluster ie-c3d

# GCP H3 (cost-effective Intel)
sky launch skypilot/inference/gcp_h3.yaml -y --cluster ie-h3

# RunPod (development)
sky launch skypilot/inference/runpod_cpu.yaml -y --cluster ie-runpod
```

### Test the Endpoint

```bash
# Get endpoint
ENDPOINT=$(sky status ie-c3d --endpoint 8080)

# Health check
curl $ENDPOINT/health

# Generate text
curl $ENDPOINT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}'
```

### Run Benchmark

```bash
# Quick benchmark (128 tokens)
sky exec ie-c3d -- "cd ~/BitNet && python run_inference.py \
  -m models/*/ggml-model-*.gguf \
  -p 'The future of AI is' -n 128 -t \$(nproc)"

# Throughput test with concurrent requests
python scripts/benchmark_throughput.py --url $ENDPOINT --concurrency 4 --duration 60
```

### Teardown

```bash
sky down ie-c3d -y
```

## Environment Variables

All configs support these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_REPO` | `microsoft/BitNet-b1.58-2B-4T` | HuggingFace model repo |
| `QUANT_TYPE` | `tl2` | Quantization type (`tl2` for AVX512, `i2_s` for general) |
| `NUM_THREADS` | (auto) | CPU threads for inference |
| `CTX_SIZE` | `8192` | Context window size |
| `PORT` | `8080` | Server port |

### Customize Model

```bash
# Launch with different model
sky launch skypilot/inference/gcp_c3d.yaml -y \
  --env MODEL_REPO=tiiuae/Falcon3-7B-Instruct-1.58bit \
  --env QUANT_TYPE=i2_s \
  --cluster ie-falcon
```

## Scaling Expectations

BitNet throughput scales approximately linearly with threads until memory bandwidth saturates:

```
Theoretical throughput (7B model, DDR5):
  16 threads:  ~12-15 tok/s
  32 threads:  ~25-30 tok/s
  64 threads:  ~50-60 tok/s
  90 threads:  ~60-70 tok/s (approaching bandwidth limit)
  180 threads: ~80-100 tok/s (bandwidth saturated)
```

The saturation point depends on:
- Model size (larger = more memory reads per token)
- Memory bandwidth (DDR5 ~460 GB/s vs DDR4 ~200 GB/s)
- NUMA topology (cross-socket memory access adds latency)

## Troubleshooting

### "Quota exceeded" Error

```bash
# Check your quotas
gcloud compute regions describe us-central1 --format="table(quotas.metric,quotas.limit,quotas.usage)"

# Or in GCP Console: IAM & Admin → Quotas → Filter by C3D
```

### Build Takes Too Long

First build compiles BitNet.cpp (~5 min). Subsequent runs use GCS cache (~30s).

```bash
# Force cache rebuild
sky launch skypilot/inference/gcp_c3d.yaml -y \
  --env CACHE_VERSION=v2_force_rebuild
```

### Model Not Found

Check the model downloaded correctly:

```bash
sky exec ie-c3d -- "ls -la ~/BitNet/models/*/"
```

## Cost Optimization

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| Use H3 instead of C3D | 55% | ~30% slower |
| Use spot instances | 60-90% | May be preempted |
| Scale to zero | 100% when idle | Cold start latency |

```yaml
# Enable spot in YAML
resources:
  use_spot: true
  spot_recovery: FAILOVER
```

## Architecture Notes

```
┌─────────────────────────────────────────────────────────────┐
│                    BitNet Inference Flow                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Request → llama-server → TL2 Kernel → DDR5 Memory         │
│                     │           │              │             │
│                     │           │              │             │
│                  OpenAI     AVX-512      ~460 GB/s           │
│                   API      VNNI ops     bandwidth            │
│                                                              │
│   Memory-bound workload: more bandwidth = more tok/s         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

The TL2 (Ternary Lookup 2-bit) kernel is optimized for:
- AMD EPYC with AVX-512 VNNI
- Intel Sapphire Rapids with AMX
- Efficient 1.58-bit weight unpacking

## See Also

- [Serving Guide](../../docs/serving.md) - Full serving architecture
- [BitNet.cpp](https://github.com/microsoft/BitNet) - Inference engine
- [SkyPilot Docs](https://skypilot.readthedocs.io/) - Cloud deployment
