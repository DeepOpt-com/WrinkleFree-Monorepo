# Serving Guide

This guide covers deploying trained 1.58-bit quantized models for inference using SkyServe.

## Overview

WrinkleFree-Deployer uses a **spillover architecture** to minimize inference costs:

1. **Base Layer (Hetzner)**: Fixed-cost dedicated servers handle baseline traffic
2. **Burst Layer (AWS/GCP)**: Scale-to-zero spot instances handle traffic spikes
3. **Orchestration (SkyServe)**: Unified load balancing, autoscaling, and health checks

```
                              Client Requests
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │            SkyServe               │
                    │   (Load Balancer + Orchestrator)  │
                    │                                   │
                    │  • Unified endpoint               │
                    │  • QPS-based autoscaling          │
                    │  • Health checks                  │
                    └─────────────────┬─────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  Hetzner Replica  │     │  Hetzner Replica  │     │   AWS/GCP Spot    │
│  (SSH Node Pool)  │     │  (SSH Node Pool)  │     │   (On-demand)     │
│                   │     │                   │     │                   │
│  Always-on        │     │  Always-on        │     │  Scale-to-zero    │
│  Fixed cost       │     │  Fixed cost       │     │  Pay-per-use      │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

**Target**: 80% cost reduction vs. pure hyperscaler deployment.

## Quick Start

### Prerequisites

1. **Install dependencies**:
   ```bash
   cd WrinkleFree-Deployer
   uv sync
   ```

2. **Configure cloud credentials**:
   ```bash
   # Copy example and edit
   cp .env.example .env
   nano .env

   # Load environment
   source scripts/setup-env.sh

   # Verify SkyPilot can access your clouds
   sky check
   ```

3. **Get a model file** (GGUF format):
   ```bash
   # Option 1: Use a model you trained
   cp ../WrinkleFree-1.58Quant/outputs/model.gguf models/

   # Option 2: Download test model
   ./scripts/download_test_model.sh
   ```

### Deploy with SkyServe

```bash
# Deploy service (finds cheapest resources automatically)
sky serve up skypilot/service.yaml --name my-model

# Wait for replicas (2-5 minutes)
watch sky serve status my-model

# Get endpoint URL
sky serve status my-model
# Output: Endpoint: https://my-model-xxxx.sky.serve
```

### Test the Service

```bash
# Health check
curl https://my-model-xxxx.sky.serve/health

# Inference request
curl https://my-model-xxxx.sky.serve/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
```

### Manage the Service

```bash
# View all replicas
sky serve status my-model --all

# View logs
sky serve logs my-model

# Scale up
sky serve update my-model --min-replicas 5

# Tear down
sky serve down my-model
```

## Configuration

### Service YAML (`skypilot/service.yaml`)

```yaml
service:
  readiness_probe:
    path: /health
    initial_delay_seconds: 120
  replica_policy:
    min_replicas: 3       # Match Hetzner node count
    max_replicas: 20      # Include cloud burst capacity
    target_qps_per_replica: 5.0
    upscale_delay_seconds: 60
    downscale_delay_seconds: 300

resources:
  ports: 8080
  cpus: 16+
  memory: 128+
  use_spot: true
  spot_recovery: FAILOVER
```

### Autoscaling

| Setting | Description | Recommendation |
|---------|-------------|----------------|
| `min_replicas` | Always-on replicas | Set to Hetzner node count |
| `max_replicas` | Maximum during peaks | Budget-dependent |
| `target_qps_per_replica` | Queries/sec per replica | Lower = more replicas, better latency |
| `downscale_delay_seconds` | Wait before scaling down | Higher = more stable, higher cost |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND` | `bitnet` | Inference backend: `bitnet` or `vllm` |
| `MODEL_PATH` | `/models/model.gguf` | Path to model file |
| `NUM_THREADS` | `auto` | CPU threads for inference |
| `CONTEXT_SIZE` | `4096` | Maximum context length |

## Adding Hetzner (Cost Optimization)

Hetzner dedicated servers are **7x cheaper** than AWS for always-on workloads.

| Provider | Cost for 256GB RAM server |
|----------|---------------------------|
| AWS r7a.8xlarge | ~$1.50/hr = $1,080/month |
| Hetzner AX102 | ~$150/month fixed |

### Setup Steps

1. **Order server** at [hetzner.com/dedicated-rootserver](https://www.hetzner.com/dedicated-rootserver)

2. **Set up SSH access**:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/hetzner_ed25519
   ssh-copy-id -i ~/.ssh/hetzner_ed25519 root@YOUR_HETZNER_IP
   ```

3. **Register with SkyPilot**:
   ```bash
   cp skypilot/ssh_node_pools.yaml.example ~/.sky/ssh_node_pools.yaml
   # Edit with your server IP
   ```

   ```yaml
   # ~/.sky/ssh_node_pools.yaml
   hetzner-base:
     user: root
     identity_file: ~/.ssh/hetzner_ed25519
     hosts:
       - YOUR_HETZNER_IP
   ```

4. **Initialize node pool**:
   ```bash
   sky ssh up
   sky check ssh
   ```

5. **Deploy** - SkyServe now uses Hetzner first (cheapest):
   ```bash
   sky serve up skypilot/service.yaml --name my-model
   ```

## Inference Backends

| Backend | Optimized For | Model Format | Use Case |
|---------|--------------|--------------|----------|
| [BitNet](https://github.com/microsoft/BitNet) | 1.58-bit on CPU | GGUF | Maximum throughput on bare metal |
| [vLLM](https://github.com/vllm-project/vllm) | General serving | safetensors | Feature-rich, GPU support |

Both expose OpenAI-compatible `/v1/completions` API.

## GCP CPU Instances (High Memory Bandwidth)

For BitNet CPU inference, memory bandwidth is the bottleneck. Use DDR5-equipped instances.

### Benchmark Results

Tested on 2024-12-23 with Falcon3-7B-Instruct-1.58bit (7.46B params, 3.05 GiB):

| Instance | vCPUs | Throughput | Latency | Cost/hr | tok/$ |
|----------|-------|------------|---------|---------|-------|
| c3d-standard-16 | 16 | **12.6 tok/s** | 79 ms | $0.73 | 62k |
| c3d-standard-90 | 90 | ~65 tok/s* | ~15 ms* | $4.00 | 59k |
| h3-standard-88 | 88 | ~45 tok/s* | ~22 ms* | $1.76 | 92k |

*Estimated. H3 offers best cost-efficiency; C3D offers lowest latency.

### Instance Options

| Instance | CPU | Memory BW | vCPUs | RAM | Cost/hr | Best For |
|----------|-----|-----------|-------|-----|---------|----------|
| `c3d-standard-90` | AMD EPYC Genoa | ~460 GB/s | 90 | 360 GB | ~$4.00 | Production |
| `c3d-standard-180` | AMD EPYC Genoa | ~460 GB/s | 180 | 720 GB | ~$8.00 | Large models |
| `h3-standard-88` | Intel Sapphire | ~307 GB/s | 88 | 352 GB | ~$1.76 | Cost-effective |

**Note:** C4D (AMD Turin) requires Hyperdisk which SkyPilot doesn't support yet ([#4705](https://github.com/skypilot-org/skypilot/issues/4705)). Use C3D instead.

### Deploy to GCP C3D

```bash
# Launch C3D instance for inference
sky launch skypilot/inference/gcp_c3d.yaml -y --cluster ie-c3d

# Get endpoint
ENDPOINT=$(sky status ie-c3d --endpoint 8080)
curl $ENDPOINT/health

# Benchmark throughput
python scripts/benchmark_throughput.py --url $ENDPOINT --duration 60
```

### Deploy to GCP H3

```bash
sky launch skypilot/inference/gcp_h3.yaml -y --cluster ie-h3
```

### Switching Backends

```yaml
# In service.yaml
envs:
  BACKEND: vllm  # or bitnet
```

## Monitoring

### Check Replica Distribution

```bash
sky serve status my-model --all

# Example output:
# Replica 0: ssh/hetzner-base (10.100.1.1) - READY
# Replica 1: ssh/hetzner-base (10.100.1.2) - READY
# Replica 2: aws (us-east-1, r7a.xlarge) - READY  <- Burst replica
```

### View Logs

```bash
# Service logs
sky serve logs my-model

# Specific replica
sky serve logs my-model --replica-id 0

# Controller logs
sky serve logs my-model --controller
```

### Metrics

Key metrics to track:
- **QPS per replica**: Load distribution
- **Latency (TTFT)**: Time to first token
- **Memory bandwidth**: Saturation indicator

See [Monitoring Tutorial](tutorials/serving/06-monitoring.md) for Prometheus + Grafana setup.

## Troubleshooting

### Service Won't Start

```bash
# Check controller logs
sky serve logs my-model --controller

# Check if SSH Node Pool is accessible
sky check ssh
```

### Replicas Not Becoming Healthy

```bash
# Check replica logs
sky serve logs my-model --replica-id 0

# Common causes:
# - Model file not found (check file_mounts)
# - Not enough memory
# - Port mismatch
```

### High Latency

```bash
# Check if replicas are overloaded
sky serve status my-model --all

# Scale up
sky serve update my-model --min-replicas 5

# Or lower target QPS (more replicas per load)
# Edit service.yaml: target_qps_per_replica: 3.0
```

See [Troubleshooting Guide](reference/troubleshooting.md) for more issues.

## Cost Optimization

### Spot Instances

60-90% cheaper but can be interrupted:

```yaml
resources:
  use_spot: true
  spot_recovery: FAILOVER  # Auto-migrate on preemption
```

### Scale to Zero

For low-traffic periods:

```yaml
replica_policy:
  min_replicas: 0  # No cost when idle
```

### Spillover Strategy

1. SkyServe places replicas on cheapest infrastructure first
2. SSH Node Pool (Hetzner) = $0/hr marginal cost → used first
3. When Hetzner is full, spills to AWS/GCP spot
4. Cloud replicas scale to zero when load decreases

See [Pricing Guide](reference/pricing.md) for detailed cost analysis.

## Production Checklist

- [ ] Hetzner nodes registered in SSH Node Pool
- [ ] `sky check` shows all clouds enabled
- [ ] Model files accessible (local, S3, or HuggingFace)
- [ ] `min_replicas` set to Hetzner node count
- [ ] `target_qps_per_replica` tuned based on benchmarks
- [ ] Monitoring configured (Prometheus/Grafana)
- [ ] Cloudflare (optional) configured for edge caching

## Next Steps

- [Concepts Guide](reference/concepts.md) - Understand the terminology
- [Architecture Guide](reference/architecture.md) - Technical deep-dive
- [Tutorials](tutorials/serving/) - Step-by-step deployment guides
