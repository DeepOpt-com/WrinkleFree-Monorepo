# Concepts & Glossary

This document explains the key concepts and terminology used in WrinkleFree-Deployer.

## Table of Contents

- [Training Concepts](#training-concepts)
- [Infrastructure Concepts](#infrastructure-concepts)
- [SkyPilot Concepts](#skypilot-concepts)
- [Inference Concepts](#inference-concepts)
- [Networking Concepts](#networking-concepts)
- [Cost Concepts](#cost-concepts)

---

## Training Concepts

### Training Pipeline (Stages)

WrinkleFree uses a multi-stage pipeline to convert full-precision models to 1.58-bit:

```
Stage 1 → Stage 1.9 → Stage 2 → Stage 3
```

| Stage | Name | What It Does | Training? |
|-------|------|--------------|-----------|
| **1** | SubLN Insertion | Modifies model architecture for quantization | No |
| **1.9** | Layer-wise Distillation | Transfers knowledge from teacher to quantized student | Yes |
| **2** | Continue Pretrain | Additional training on 10B+ tokens | Yes |
| **3** | Distillation Fine-tuning | Task-specific fine-tuning | Yes |

### Quantization

Reducing the precision of model weights to save memory and speed up inference.

```
Full precision (FP32):  32 bits per weight   100% quality
1.58-bit (Ternary):     ~2 bits per weight   ~90% quality

1.58-bit weights are {-1, 0, +1} only
→ 10x smaller models
→ 2-6x faster inference on CPU
```

### Knowledge Distillation

Training a smaller "student" model to mimic a larger "teacher" model.

```
Teacher Model (Full Precision)
        │
        │ "Here's what I would predict"
        ▼
Student Model (1.58-bit)
        │
        │ Learns to match teacher's outputs
        ▼
Final Quantized Model
```

### SubLN (Sub-Layer Normalization)

A technique that adds normalization layers inside transformer blocks to stabilize low-bit training.

### FSDP (Fully Sharded Data Parallel)

A distributed training technique that shards model weights across multiple GPUs:

```
Standard Data Parallel:          FSDP:
┌─────────────────────┐         ┌─────────────────────┐
│ GPU 0: Full Model   │         │ GPU 0: Shard 1      │
│ GPU 1: Full Model   │         │ GPU 1: Shard 2      │
│ GPU 2: Full Model   │         │ GPU 2: Shard 3      │
│ (3x memory used)    │         │ (1x memory used)    │
└─────────────────────┘         └─────────────────────┘
```

**Use case**: Training models too large for a single GPU.

### Managed Jobs (SkyPilot)

Long-running training jobs with automatic recovery:

```
sky jobs launch train.yaml

Features:
├── Automatic spot recovery (if preempted, restarts elsewhere)
├── Checkpoint resumption (continues from last save)
├── Multi-cloud failover (tries different regions/clouds)
└── Cost tracking
```

### Checkpoint

A saved snapshot of model weights during training.

```
checkpoints/
├── checkpoint-500/    # Saved at step 500
├── checkpoint-1000/   # Saved at step 1000
└── final/             # Final trained model

Used for:
├── Resuming after interruption
├── Evaluating intermediate results
└── Rollback if training degrades
```

### W&B (Weights & Biases)

A platform for tracking ML experiments:

```
Training Run → W&B Dashboard
    │
    ├── Loss curves
    ├── Learning rate schedule
    ├── GPU utilization
    ├── Hyperparameters
    └── Artifacts (checkpoints)
```

---

## Infrastructure Concepts

### Cloud Provider

A company that rents computing resources over the internet.

| Provider | Type | Best For |
|----------|------|----------|
| AWS (Amazon) | Public cloud | Flexibility, global reach |
| GCP (Google) | Public cloud | ML/AI workloads |
| Azure (Microsoft) | Public cloud | Enterprise integration |
| Hetzner | Dedicated servers | Cost-effective base capacity |

### Instance / VM / Server

A virtual or physical computer you can rent.

```
┌─────────────────────────────────────┐
│            Instance                  │
│                                     │
│  CPU: 16 cores                      │
│  RAM: 256 GB                        │
│  Disk: 500 GB NVMe                  │
│  Network: 10 Gbps                   │
│                                     │
│  Running: Ubuntu 22.04              │
│  Your app runs here                 │
└─────────────────────────────────────┘
```

**Types of instances:**

| Type | Description | Cost | Availability |
|------|-------------|------|--------------|
| **On-demand** | Pay by the hour, always available | $$$ | Guaranteed |
| **Spot/Preemptible** | Spare capacity, can be taken back | $ | Not guaranteed |
| **Reserved** | Commit for 1-3 years | $$ | Guaranteed |
| **Dedicated** | Physical server just for you | $$ | Guaranteed |

### Spot Instance

Spare cloud capacity sold at a discount (60-90% off).

**The catch:** The cloud provider can take it back with 2 minutes notice when they need the capacity.

```
Normal pricing:    $1.00/hour
Spot pricing:      $0.30/hour  (70% off!)

But sometimes:     "Your spot instance will be terminated in 2 minutes"
```

**Why we use them:** SkyServe automatically handles spot interruptions by starting a new instance elsewhere.

### Bare Metal / Dedicated Server

A physical server that's entirely yours (not shared with others).

```
Cloud VM (shared):              Dedicated Server:
┌─────────────────┐             ┌─────────────────┐
│ Your VM         │             │                 │
├─────────────────┤             │  Your Server    │
│ Someone else's  │             │  (all of it)    │
├─────────────────┤             │                 │
│ Another person  │             │                 │
└─────────────────┘             └─────────────────┘
    One physical                   One physical
    machine shared                 machine, all yours
```

**Advantages:**
- Predictable performance (no "noisy neighbors")
- Often cheaper for 24/7 workloads
- Full hardware access

**Disadvantages:**
- Takes longer to provision (hours vs minutes)
- Less flexible scaling

### Region / Availability Zone

Physical location of cloud data centers.

```
AWS Regions:
├── us-east-1 (Virginia)
│   ├── us-east-1a (Availability Zone 1)
│   ├── us-east-1b (Availability Zone 2)
│   └── us-east-1c (Availability Zone 3)
├── us-west-2 (Oregon)
├── eu-west-1 (Ireland)
└── ...

Hetzner Locations:
├── fsn1 (Falkenstein, Germany)
├── nbg1 (Nuremberg, Germany)
└── hel1 (Helsinki, Finland)
```

**Why it matters:**
- **Latency:** Closer to users = faster responses
- **Cost:** Prices vary by region
- **Availability:** Some instance types only in certain regions

---

## SkyPilot Concepts

### SkyPilot

An open-source tool that makes it easy to run workloads on any cloud.

**Without SkyPilot:**
```bash
# AWS
aws ec2 run-instances --image-id ami-xxx --instance-type r7a.xlarge ...
# GCP
gcloud compute instances create my-vm --machine-type n2d-highmem-8 ...
# Different commands, different concepts for each cloud!
```

**With SkyPilot:**
```bash
# Works on any cloud!
sky launch my-task.yaml
```

### Task (SkyPilot)

A YAML file describing what you want to run.

```yaml
# my-task.yaml
resources:
  cpus: 16+        # What you need
  memory: 128+

setup: |           # How to set up the environment
  pip install torch

run: |             # What to run
  python train.py
```

### Cluster (SkyPilot)

A group of instances working together on a task.

```
sky launch -c my-cluster task.yaml

Creates:
┌─────────────────────────────────────────────────┐
│  Cluster: my-cluster                            │
│                                                 │
│  ┌─────────────┐  ┌─────────────┐              │
│  │ Head Node   │  │ Worker Node │              │
│  │ (manages)   │  │ (computes)  │              │
│  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────┘
```

### SkyServe

SkyPilot's feature for deploying long-running services with autoscaling.

**Cluster vs Service:**

| Cluster (`sky launch`) | Service (`sky serve`) |
|------------------------|----------------------|
| Runs once, then stops | Runs continuously |
| You manage scaling | Automatic scaling |
| Direct SSH access | Load-balanced endpoint |
| For batch jobs, training | For serving, APIs |

### Service (SkyServe)

A deployed application with:
- **Endpoint:** A URL to access it
- **Replicas:** Multiple copies for reliability
- **Autoscaling:** Adjusts capacity based on load

```
sky serve up service.yaml --name my-api

Creates:
┌─────────────────────────────────────────────────┐
│  Service: my-api                                │
│  Endpoint: https://my-api-xxx.sky.serve         │
│                                                 │
│  ┌─────────────┐                               │
│  │ Controller  │  (manages everything)          │
│  └──────┬──────┘                               │
│         │                                       │
│  ┌──────┴──────┬──────────────┐                │
│  ▼             ▼              ▼                │
│ ┌────────┐  ┌────────┐  ┌────────┐            │
│ │Replica │  │Replica │  │Replica │            │
│ │   0    │  │   1    │  │   2    │            │
│ └────────┘  └────────┘  └────────┘            │
└─────────────────────────────────────────────────┘
```

### Replica

One copy of your service running on one instance.

**Why multiple replicas?**
1. **Reliability:** If one crashes, others keep serving
2. **Capacity:** More replicas = more requests/second
3. **Latency:** Distribute load so each replica stays fast

### SSH Node Pool

A way to add your own servers (like Hetzner) to SkyPilot.

```yaml
# ~/.sky/ssh_node_pools.yaml
my-servers:
  user: root
  hosts:
    - 10.0.1.100  # Your server 1
    - 10.0.1.101  # Your server 2
```

After setup, SkyPilot treats these servers just like cloud instances:
```bash
sky launch --infra ssh/my-servers task.yaml
```

### Readiness Probe

A health check that determines if a replica is ready to receive traffic.

```yaml
readiness_probe:
  path: /health              # URL to check
  initial_delay_seconds: 60  # Wait before first check
  period_seconds: 10         # Check every 10 seconds
```

```
Replica starting...
│
├── [0s] Setup script runs
├── [30s] Run script starts
├── [45s] Model loading...
├── [60s] First health check → FAIL (still loading)
├── [70s] Health check → FAIL
├── [80s] Health check → PASS ✓
│
└── Replica marked READY, receives traffic
```

---

## Inference Concepts

### Inference

Using a trained AI model to make predictions.

```
Training (expensive, done once):
    Data → [Train] → Model

Inference (cheap, done many times):
    Input → [Model] → Output
    "Hello" → [LLM] → "Hello! How can I help?"
```

### Model Serving

Making a model available as an API.

```
Without serving:
    python inference.py --prompt "Hello"
    (Run manually each time)

With serving:
    curl http://api/v1/completions -d '{"prompt": "Hello"}'
    (Always available, handles multiple requests)
```

### Tokens

The units that language models work with. Roughly:
- 1 token ≈ 4 characters
- 1 token ≈ 0.75 words

```
"Hello, how are you today?"
    │
    ▼
["Hello", ",", " how", " are", " you", " today", "?"]
    7 tokens
```

**Why it matters:**
- Models have maximum token limits (context window)
- You're often charged per token
- Inference speed measured in tokens/second

### Tokens Per Second (TPS)

How fast a model generates output.

```
Slow:   5 tokens/second   (readable typing speed)
Fast:   50 tokens/second  (instant feel)
Very fast: 200+ tokens/second (batch processing)
```

### Quantization

Reducing model precision to save memory and speed up inference.

```
Full precision (FP32):  32 bits per number   100% quality
Half precision (FP16):  16 bits per number   ~99% quality
8-bit (INT8):           8 bits per number    ~98% quality
4-bit (INT4):           4 bits per number    ~95% quality
1.58-bit (Ternary):     ~2 bits per number   ~90% quality  ◄── What we use!
```

**Our models use 1.58-bit quantization:**
- Weights are {-1, 0, +1} only
- 10x smaller than full precision
- 2-6x faster on CPU
- Slight quality loss (minimized by training technique)

### GGUF

A file format for storing quantized models.

```
model.gguf
├── Metadata (model name, architecture, etc.)
├── Tokenizer (how to convert text to tokens)
└── Weights (the actual model parameters)
```

**Used by:** llama.cpp, BitNet, and many other inference engines.

### Inference Backend / Engine

Software that runs the model.

| Backend | Optimized For | Notes |
|---------|---------------|-------|
| **BitNet** | 1.58-bit models on CPU | What we use for base layer |
| **vLLM** | General GPU serving | Feature-rich, good for GPUs |
| **llama.cpp** | CPU inference | BitNet is based on this |
| **TensorRT-LLM** | NVIDIA GPUs | Maximum GPU performance |

---

## Networking Concepts

### Load Balancer

Distributes incoming requests across multiple servers.

```
                    Requests
                        │
                        ▼
               ┌────────────────┐
               │ Load Balancer  │
               └───────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │Server 1 │   │Server 2 │   │Server 3 │
    └─────────┘   └─────────┘   └─────────┘
```

**Strategies:**
- **Round-robin:** Each server gets requests in turn
- **Least connections:** Send to server with fewest active requests
- **Weighted:** Some servers get more traffic than others

### Endpoint

A URL where your service can be accessed.

```
https://my-model-abc123.sky.serve/v1/completions
│       │                        │
│       │                        └── Path (which API)
│       └── Host (where to connect)
└── Protocol (how to connect)
```

### Health Check

A way to verify a server is working.

```
Load Balancer                    Server
      │                             │
      │──── GET /health ───────────►│
      │                             │
      │◄─── 200 OK ─────────────────│  (Healthy!)
      │                             │
      │──── GET /health ───────────►│
      │                             │
      │◄─── Connection refused ─────│  (Unhealthy!)
      │                             │
      └── Remove from rotation ─────┘
```

### VPN / WireGuard

A secure tunnel between networks.

```
Without VPN:
    Hetzner ──── Public Internet ──── AWS
                 (unencrypted)

With WireGuard VPN:
    Hetzner ════ Encrypted Tunnel ════ AWS
                 (secure, private)
```

**Why we use it:**
- Secure communication between Hetzner and cloud
- Private IP addresses across different networks
- Lower latency than going through public internet

---

## Cost Concepts

### On-Demand vs Spot Pricing

```
On-demand: $1.00/hour
├── Always available
├── Never interrupted
└── Full price

Spot: $0.30/hour (70% off!)
├── Uses spare capacity
├── Can be interrupted
└── Great for fault-tolerant workloads
```

### Cost Per Token

How much it costs to generate output.

```
Cloud API (OpenAI, Anthropic):
    $0.01 - $0.10 per 1K tokens

Self-hosted (WrinkleFree):
    $0.0001 - $0.001 per 1K tokens

That's 10-100x cheaper!
```

### TCO (Total Cost of Ownership)

All costs combined, not just compute.

```
Obvious costs:
├── Instance hours
└── Data transfer

Hidden costs:
├── Engineering time
├── Monitoring tools
├── On-call burden
├── Training/learning
└── Opportunity cost
```

### Spillover Architecture

Our cost optimization strategy:

```
Traffic
  │
  │     ┌─────────────────────────────────────────┐
  │     │                                         │
  │     │            Cloud Burst Layer            │
  │     │           (expensive, elastic)          │
  │     │                                         │
  │   ──┼─────────────────────────────────────────┤ Spillover threshold
  │     │                                         │
  │     │            Hetzner Base Layer           │
  │     │            (cheap, fixed)               │
  │     │                                         │
  └─────┴─────────────────────────────────────────┘
        Time →

Below threshold: Only use cheap Hetzner servers
Above threshold: Spill over to cloud (AWS/GCP)
```

---

## Quick Reference

| Term | One-Line Definition |
|------|---------------------|
| **Autoscaling** | Automatically adding/removing servers based on load |
| **Backend** | Software that runs the AI model |
| **Bare metal** | A physical server (not virtualized) |
| **Cluster** | A group of servers working together |
| **Endpoint** | URL where your service is accessible |
| **GGUF** | File format for quantized models |
| **Health check** | Test to verify a server is working |
| **Inference** | Using a model to make predictions |
| **Instance** | A virtual or physical server |
| **Load balancer** | Distributes requests across servers |
| **Node pool** | A group of servers registered with SkyPilot |
| **Quantization** | Reducing model precision to save memory |
| **Replica** | One copy of your service |
| **Service** | A deployed application with load balancing |
| **SkyPilot** | Tool for running workloads on any cloud |
| **SkyServe** | SkyPilot's service deployment feature |
| **Spot instance** | Discounted spare cloud capacity |
| **Token** | Basic unit of text for language models |
| **VPN** | Secure tunnel between networks |
