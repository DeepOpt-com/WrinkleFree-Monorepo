# Architecture Guide

## System Overview

WrinkleFree-Deployer implements a **spillover architecture** using **SkyServe** as the unified orchestration layer. SkyServe manages replicas across bare metal (Hetzner), Kubernetes (OVHCloud), and cloud (AWS/GCP) infrastructure, handling load balancing, autoscaling, and health checks automatically.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Request Flow                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Client Request
         │
         ▼
┌─────────────────┐
│   Cloudflare    │  ◄── Edge CDN, DDoS protection, optional caching
│   (Optional)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SkyServe                                        │
│                     (Unified Load Balancer + Orchestrator)                   │
│                                                                              │
│   • Automatic replica placement across all available infrastructure          │
│   • Built-in load balancing (least-connections)                             │
│   • Health checks and automatic failover                                     │
│   • Autoscaling based on QPS                                                │
│   • Single endpoint for all replicas                                         │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
    ▼                            ▼                            ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Hetzner Replica │    │ OVHCloud MKS    │    │  AWS/GCP Spot   │
│   (SSH Pool)    │    │  (Kubernetes)   │    │   (Native)      │
│                 │    │                 │    │                 │
│  Always-on      │    │  Hourly billing │    │  Scale-to-zero  │
│  Fixed cost     │    │  EU compliance  │    │  Pay-per-use    │
│  $150/mo        │    │  ~$0.08-0.45/hr │    │  ~$0.15-0.80/hr │
└─────────────────┘    └─────────────────┘    └─────────────────┘
      Tier 1                 Tier 2                 Tier 3
   (Cheapest)           (Middle tier)        (Most elastic)
```

## Why SkyServe?

| Feature | Benefit |
|---------|---------|
| **Unified endpoint** | Single URL for all replicas, regardless of infrastructure |
| **Automatic placement** | SkyServe places replicas on cheapest available infra first |
| **Built-in autoscaling** | Scale based on QPS, no external autoscaler needed |
| **Health management** | Automatic health checks, replica replacement |
| **Multi-cloud + Kubernetes** | Seamlessly span Hetzner + OVHCloud + AWS + GCP |
| **Scale-to-zero** | Cloud replicas can scale to 0 when base layer handles load |

## SkyServe Architecture

### How SkyServe Spillover Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SkyServe Replica Placement                           │
└─────────────────────────────────────────────────────────────────────────────┘

1. Service deployed with: sky serve up service.yaml

2. SkyServe evaluates available infrastructure:
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐
   │ SSH Node Pool   │  │ Kubernetes      │  │ AWS             │  │ GCP        │
   │ (Hetzner)       │  │ (OVHCloud MKS)  │  │                 │  │            │
   │                 │  │                 │  │                 │  │            │
   │ Cost: $0/hr     │  │ Cost: $0.08/hr  │  │ Cost: $0.15/hr  │  │ $0.12/hr   │
   │ (fixed monthly) │  │ (hourly)        │  │ (spot)          │  │ (spot)     │
   │                 │  │                 │  │                 │  │            │
   │ Capacity: 3     │  │ Capacity: 10    │  │ Capacity: ∞     │  │ ∞          │
   └─────────────────┘  └─────────────────┘  └─────────────────┘  └────────────┘
        Priority 1           Priority 2           Priority 3        Priority 4

3. SkyServe places replicas on cheapest available first:

   min_replicas=3 → All 3 go to Hetzner (cheapest, fixed cost)

   Load increases, need 8 replicas:
   → 3 on Hetzner (at capacity)
   → 5 on OVHCloud Kubernetes (next cheapest)

   Load increases more, need 15 replicas:
   → 3 on Hetzner (at capacity)
   → 10 on OVHCloud (at capacity)
   → 2 on AWS/GCP (cheapest spot available)

   Load decreases:
   → Cloud replicas scale down first (most expensive)
   → OVHCloud scales down next
   → Hetzner replicas remain (always-on, already paid)
```

### SkyServe Service Lifecycle

```bash
# 1. Register Hetzner nodes as SSH Node Pool
sky ssh up

# 2. Deploy service (SkyServe manages everything)
sky serve up skypilot/service.yaml --name inference

# 3. Get the unified endpoint
sky serve status inference
# Returns: https://<endpoint>.sky.serve/

# 4. All requests go through SkyServe load balancer
curl https://<endpoint>.sky.serve/v1/completions

# 5. Monitor replicas
sky serve status inference --all

# 6. Scale manually if needed
sky serve update inference --min-replicas 5

# 7. Tear down
sky serve down inference
```

### Key SkyServe Concepts

| Concept | Description |
|---------|-------------|
| **Service** | A long-running deployment with load balancing |
| **Replica** | One instance of your service running on some infra |
| **Endpoint** | The URL SkyServe provides for your service |
| **Readiness Probe** | Health check that determines if replica can receive traffic |
| **Replica Policy** | Autoscaling rules (min/max replicas, target QPS) |

## Component Architecture

### 1. SkyPilot SSH Node Pools (Hetzner Integration)

SkyPilot doesn't natively support Hetzner, but we leverage **SSH Node Pools** to register bare metal servers as managed infrastructure.

**Workflow:**
```
Terraform (Provision)  ───►  ssh_node_pools.yaml  ───►  SkyPilot (Orchestrate)
       │                            │                           │
       │                            │                           │
  Hetzner Robot API           ~/.sky/ssh_node_pools.yaml    sky launch/serve
  - Order servers             - Register IPs                - Deploy workloads
  - Configure networking      - SSH credentials             - Health monitoring
  - Install base OS           - Resource specs              - Rolling updates
```

**Configuration (`~/.sky/ssh_node_pools.yaml`):**
```yaml
hetzner-base:
  user: root
  identity_file: ~/.ssh/hetzner_ed25519
  hosts:
    - ip: 10.0.1.100
      # Optional: Override per-node
      # user: deploy
      # identity_file: ~/.ssh/node1_key
    - ip: 10.0.1.101
    - ip: 10.0.1.102
```

**Commands:**
```bash
# Initialize node pool (installs SkyPilot runtime)
sky ssh up

# Verify nodes are available
sky check ssh

# Deploy to specific pool
sky launch --infra ssh/hetzner-base service.yaml

# Deploy service with autoscaling
sky serve up --infra ssh/hetzner-base service.yaml
```

### 2. Inference Backend Abstraction

Both inference engines expose the same OpenAI-compatible API, allowing seamless switching.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Inference Abstraction Layer                         │
└─────────────────────────────────────────────────────────────────────────────┘

                    /v1/completions
                    /v1/chat/completions
                    /health
                           │
                           ▼
              ┌────────────────────────┐
              │    Backend Router      │
              │    (BACKEND env var)   │
              └───────────┬────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐        ┌─────────────────────┐
│       BitNet        │        │        vLLM         │
│  (llama.cpp fork)   │        │    (CPU build)      │
├─────────────────────┤        ├─────────────────────┤
│ Model: GGUF         │        │ Model: safetensors  │
│ Optimized: 1.58-bit │        │ Optimized: General  │
│ Target: CPU (AVX512)│        │ Target: CPU/GPU     │
│ Tokens/s: Higher    │        │ Features: Rich      │
└─────────────────────┘        └─────────────────────┘
```

**Selection Criteria:**

| Criteria | BitNet | vLLM |
|----------|--------|------|
| 1.58-bit models | Optimal | Supported |
| CPU-only inference | Optimal | Good |
| GPU support | Limited | Full |
| Continuous batching | Basic | Advanced |
| Speculative decoding | No | Yes |
| Model format | GGUF only | Multiple |

### 3. Traffic Routing Strategy

**Priority-Based Spillover:**

```
                    Incoming QPS
                         │
                         ▼
              ┌──────────────────┐
              │  Cloudflare LB   │
              │                  │
              │  Pool A: w=1.0   │──────► Hetzner (Primary)
              │  Pool B: w=0.0   │        │
              └──────────────────┘        │
                                          ▼
                              ┌─────────────────────┐
                              │   Health Check      │
                              │   GET /health       │
                              │   Latency < 200ms   │
                              └──────────┬──────────┘
                                         │
                          ┌──────────────┴──────────────┐
                          │                             │
                     Healthy                       Unhealthy
                          │                             │
                          ▼                             ▼
                   Continue routing            Route to Pool B (AWS)
                   to Pool A                   Trigger scale-up
```

**Health Check Conditions:**
- HTTP 200 on `/health` endpoint
- Response time < 200ms
- Memory bandwidth utilization < 90%
- Queue depth < threshold

### 4. Monitoring Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Observability                                      │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌────────────┐     ┌────────────┐     ┌────────────┐
    │  BitNet/   │     │ Prometheus │     │  Grafana   │
    │   vLLM     │────►│            │────►│            │
    │  Metrics   │     │  Scraping  │     │ Dashboards │
    └────────────┘     └────────────┘     └────────────┘
          │
          │  Exported Metrics:
          │
          ├── tokens_per_second          (throughput)
          ├── time_to_first_token        (latency)
          ├── memory_bandwidth_utilization (saturation)
          ├── active_requests            (concurrency)
          └── queue_depth                (backpressure)
```

**Alert Thresholds:**
- `memory_bandwidth_utilization > 90%` → Mark unhealthy, trigger spillover
- `time_to_first_token > 500ms` → Scale up burst capacity
- `queue_depth > 100` → Emergency scale-up

## Deployment Patterns

### Pattern 1: Development (Single Node)

```yaml
# skypilot/service-dev.yaml
resources:
  cpus: 8+
  memory: 64+

run: |
  python -m inference_server --backend bitnet --model /models/dev.gguf
```

### Pattern 2: Production (Hybrid)

```yaml
# skypilot/service-prod.yaml
service:
  readiness_probe: /health
  replica_policy:
    min_replicas: 3      # Hetzner base
    max_replicas: 20     # Including burst
    target_qps_per_replica: 5.0
    upscale_delay_seconds: 30
    downscale_delay_seconds: 300

resources:
  ports: 8080
  cpus: 16+
  memory: 256+
```

### Pattern 3: Cost-Optimized (Spot Heavy)

```yaml
# skypilot/service-spot.yaml
resources:
  use_spot: true
  spot_recovery: EAGER_NEXT_REGION

replica_policy:
  min_replicas: 0       # Scale to zero
  max_replicas: 50
  target_qps_per_replica: 3.0
```

## Training Architecture

### Multi-Stage Training Pipeline

WrinkleFree converts full-precision models to 1.58-bit quantized models through a multi-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Full Precision Model                     1.58-bit Quantized Model
    (e.g., Qwen3-4B)                        (WrinkleFree-Qwen3-4B)
           │                                            ▲
           │                                            │
           ▼                                            │
    ┌──────────────┐                            ┌──────────────┐
    │   Stage 1    │                            │   Stage 3    │
    │              │                            │              │
    │ SubLN Insert │                            │  Distill FT  │
    │  (No Train)  │                            │  (Task Data) │
    │              │                            │              │
    │ Seconds      │                            │ ~500-2k steps│
    │ $0           │                            │ ~$10-50      │
    └──────┬───────┘                            └──────▲───────┘
           │                                           │
           │ Modified Architecture                     │
           ▼                                           │
    ┌──────────────┐                            ┌─────────────┐
    │  Stage 1.9   │                            │   Stage 2   │
    │              │────────────────────────────►              │
    │  Layer-wise  │    Quantized Weights       │   Continue  │
    │  Distill     │                            │  Pretrain   │
    │              │                            │             │
    │ ~500-2k steps│                            │ ~1k-10k steps│
    │ ~$10-50      │                            │ ~$50-200    │
    └──────────────┘                            └─────────────┘
```

### Training Infrastructure

Training jobs use **SkyPilot Managed Jobs** for automatic recovery and checkpoint management:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Training Job Lifecycle                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    Developer                    SkyPilot                    Cloud Provider
        │                           │                              │
        │  sky jobs launch          │                              │
        │  train.yaml               │                              │
        ├──────────────────────────►│                              │
        │                           │                              │
        │                           │  Provision GPU cluster       │
        │                           ├─────────────────────────────►│
        │                           │                              │
        │                           │◄─────────────────────────────┤
        │                           │  cluster_id, GPU IPs         │
        │                           │                              │
        │                           │  Mount checkpoint bucket     │
        │                           │  (S3/GCS MOUNT_CACHED)       │
        │                           ├──────────────┐               │
        │                           │              │               │
        │                           │◄─────────────┘               │
        │                           │                              │
        │                           │  Start training              │
        │                           │  - Stage 1.9 (2000 steps)    │
        │                           │  - Save checkpoint @500      │
        │                           ├──────────────┐               │
        │                           │              │               │
        │  sky jobs logs <id>       │              │               │
        ├──────────────────────────►│              │               │
        │◄──────────────────────────┤              │               │
        │  Step 500: loss=2.3       │              │               │
        │                           │              │               │
        │                           │ [Spot Preemption!]           │
        │                           │◄─────────────────────────────┤
        │                           │                              │
        │                           │  Auto-restart job            │
        │                           │  Resume from checkpoint-500  │
        │                           ├─────────────────────────────►│
        │                           │                              │
        │                           │              │               │
        │                           │◄─────────────┘               │
        │                           │  Training complete           │
        │                           │  Final checkpoint saved      │
        │                           │                              │
        │  Download checkpoints     │                              │
        ├──────────────────────────►│                              │
        │◄──────────────────────────┤                              │
        │  s3://checkpoints/final/  │                              │
```

### Checkpoint Storage Architecture

Training uses **SkyPilot Storage** for automatic checkpoint management:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Checkpoint Storage Flow                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    Training Node                Storage Layer              Cloud Storage
    (RunPod GPU)                (SkyPilot)                 (S3/GCS/R2)
         │                           │                           │
         │  Save checkpoint          │                           │
         │  /checkpoint/step-500/    │                           │
         ├──────────────────────────►│                           │
         │                           │                           │
         │                           │  MOUNT_CACHED mode:       │
         │                           │  - Async upload           │
         │                           │  - 9.6x faster           │
         │                           │  - No disk blocking       │
         │                           ├──────────────────────────►│
         │                           │                           │
         │  Continue training        │                           │
         │  (Not blocked on upload)  │  Background upload        │
         ├──────────────┐            │  continues...             │
         │              │            │                           │
         │              │            │                           │
         │◄─────────────┘            │                           │
         │                           │                           │
         │  [Spot Preemption]        │                           │
         │  Job terminated           │                           │
         ├──────────────────────────►│                           │
         │                           │                           │
         │                           │  Download checkpoint      │
         │                           │◄──────────────────────────┤
         │  Mount /checkpoint        │                           │
         │◄──────────────────────────┤                           │
         │                           │                           │
         │  Resume from step-500     │                           │
         ├──────────────┐            │                           │
         │              │            │                           │
         │◄─────────────┘            │                           │
```

**Storage Options:**
- **S3** (AWS): Default, good performance
- **GCS** (GCP): Good for GCP-hosted training
- **R2** (Cloudflare): Cheaper egress, S3-compatible
- **Azure Blob**: For Azure-hosted training

**Mount Modes:**
- **MOUNT_CACHED** (recommended): 9.6x faster writes, async uploads
- **MOUNT**: Direct mount, slower writes
- **COPY**: Copy on start/end, simplest

### Training Job Recovery

SkyPilot automatically handles failures:

```yaml
# In train.yaml
resources:
  job_recovery:
    max_restarts_on_errors: 3  # Auto-retry on failure
  use_spot: true  # Use cheap spot instances
  spot_recovery: FAILOVER  # Migrate to new instance on preemption
```

**Recovery scenarios:**
1. **Spot Preemption**: Auto-restart on new instance, resume from checkpoint
2. **NCCL Timeout**: Retry training step (transient network error)
3. **Driver Crash**: Restart job, resume from checkpoint
4. **OOM Error**: Job fails (no auto-restart, requires config change)

### Monitoring & Observability

Training jobs integrate with W&B for metrics tracking:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Training Observability                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    Training Process              W&B API                  Dashboard
         │                           │                         │
         │  Log metrics:             │                         │
         │  - loss: 2.34             │                         │
         │  - lr: 5e-5               │                         │
         │  - tokens/s: 1200         │                         │
         ├──────────────────────────►│                         │
         │                           │                         │
         │                           │  Store & aggregate      │
         │                           ├────────────────────────►│
         │                           │                         │
         │                           │                         │
         │  Log GPU stats:           │                         │
         │  - util: 98%              │                         │
         │  - memory: 73GB/80GB      │                         │
         │  - power: 450W            │                         │
         ├──────────────────────────►│                         │
         │                           │                         │
         │                           │  Update charts          │
         │                           ├────────────────────────►│
         │                           │  (real-time)            │
         │                           │                         │
         │  Save checkpoint          │                         │
         │  - size: 8.2GB            │                         │
         │  - step: 1000             │                         │
         ├──────────────────────────►│                         │
         │                           │                         │
         │                           │  Log artifact           │
         │                           ├────────────────────────►│
         │                           │                         │
         │                           │                    ┌────┴────┐
         │                           │                    │ Alerts  │
         │                           │                    │ - GPU   │
         │                           │◄───────────────────┤   util  │
         │                           │                    │   <70%  │
         │                           │                    └─────────┘
```

**Metrics tracked:**
- Training loss & perplexity
- Learning rate schedule
- Throughput (tokens/sec, samples/sec)
- GPU utilization & memory
- Gradient norms
- Checkpoint sizes
- Dataset progress

**GPU Monitoring:**

Training jobs automatically log GPU stats:

```bash
=== GPU Utilization 2025-12-21 10:00:00 ===
utilization.gpu [%], utilization.memory [%], memory.used [MiB], power.draw [W]
98 %, 92 %, 73728 MiB, 450.00 W
```

Logged every 60-300 seconds to detect:
- Low GPU utilization (<70%): Data loading bottleneck
- Low memory usage (<60%): Batch size too small
- Power throttling: Thermal issues

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Security Layers                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Internet
        │
        ▼
┌───────────────────┐
│    Cloudflare     │  ◄── DDoS protection, WAF
│    (Edge)         │      API key validation
└────────┬──────────┘      Rate limiting
         │
         │ mTLS
         ▼
┌───────────────────┐
│   Bastion Host    │  ◄── SSH jump host only
│   (Management)    │      No direct access to inference nodes
└────────┬──────────┘
         │
         │ Private Network (vSwitch 10G)
         ▼
┌───────────────────┐
│  Inference Nodes  │  ◄── No public IP
│  (Hetzner/AWS)    │      Firewall: Cloudflare IPs only
└───────────────────┘      mlock enabled (no swap)
```

**Key Security Controls:**
1. **mTLS**: All traffic between Cloudflare and origins encrypted
2. **Network Isolation**: Inference nodes on private network only
3. **Auth Termination**: API keys validated at edge, not on inference servers
4. **Memory Protection**: `mlock` prevents model weights from swapping to disk
