# GPU Cloud Providers Comparison (December 2025)

## Executive Summary

For WrinkleFree training workloads, we recommend:
- **Primary**: Nebius AI Cloud ($1.99-2.15/hr H100)
- **Fallback**: RunPod ($2.39-2.79/hr H100)
- **Orchestration**: SkyPilot for multi-cloud job management

---

## H100 Pricing Comparison

| Provider | H100 Price/hr | Notes |
|----------|---------------|-------|
| **Nebius** | $1.99-2.15 | Explorer Tier at $1.99 for up to 1000 GPU-hrs/month |
| **RunPod** | $1.99-2.79 | Community pool $1.99, Secure $2.39 |
| **Lambda Labs** | $2.99 | 1-click clusters, sometimes low stock |
| **CoreWeave** | $6.16 | HGX nodes at $49.24/8-GPU, enterprise HPC |
| AWS (P5) | ~$3.90 | Hyperscaler pricing |
| GCP (A3-high) | ~$3.00 | Hyperscaler pricing |
| Azure (NC H100) | ~$6.98 | Hyperscaler pricing |

*Prices as of November-December 2025. Market saw 44% AWS cuts in June 2025.*

---

## Provider Details

### Nebius AI Cloud (Recommended Primary)

**Pros:**
- Lowest H100 pricing ($1.99/hr Explorer Tier)
- Native SkyPilot integration (managed API server available)
- EU data centers (Finland, Paris) for GDPR compliance
- InfiniBand networking for distributed training
- End-to-end AI workflows: managed MLflow, AI Studio for fine-tuning

**Cons:**
- Newer provider (emerged from Yandex AI team)
- Smaller geographic footprint than hyperscalers

**SkyPilot Setup:**
```bash
pip install "skypilot-nightly[nebius]"
wget https://raw.githubusercontent.com/nebius/nebius-solution-library/main/skypilot/nebius-setup.sh
chmod +x nebius-setup.sh && ./nebius-setup.sh
sky check nebius
```

---

### RunPod (Recommended Fallback)

**Pros:**
- Up to 90% savings on spot instances
- Per-second billing, transparent pricing
- Serverless GPU functions with <3s cold start
- Full Docker control
- Good availability across GPU types

**Cons:**
- Community pool can be less reliable
- Limited enterprise features vs CoreWeave

**SkyPilot Setup:**
```bash
pip install "skypilot[runpod]"
export RUNPOD_API_KEY="your-key"
sky check runpod
```

---

### Lambda Labs

**Pros:**
- One-click GPU cluster deployments
- Pre-installed ML frameworks, Jupyter notebooks
- NVIDIA Quantum-2 400Gb/s InfiniBand
- No egress fees
- NVIDIA investor (early GPU access)

**Cons:**
- Often out of capacity ("excellent but often out of stock")
- Limited monitoring tools
- Higher price than Nebius/RunPod

**SkyPilot Setup:**
```bash
pip install "skypilot[lambda]"
# Configure API key via Lambda dashboard
sky check lambda
```

---

### CoreWeave

**Pros:**
- Enterprise-grade InfiniBand + GPUDirect RDMA
- Kubernetes-native platform
- Up to 60% reserved capacity discounts
- Serves Microsoft, OpenAI, Google, NVIDIA
- Best for large multi-GPU distributed training

**Cons:**
- 3x more expensive than Nebius/RunPod
- $8B debt burden (financial sustainability questions)
- Overkill for smaller projects

**SkyPilot Setup:**
```bash
pip install "skypilot[coreweave]"
# Requires CKS cluster + kubeconfig
sky check coreweave
```

---

## SkyPilot Multi-Cloud Configuration

### Recommended YAML for WrinkleFree

```yaml
resources:
  cloud: nebius  # Primary
  accelerators: H100:4
  use_spot: true
  disk_tier: best
  network_tier: best  # InfiniBand on supported clouds

# Automatic fallback to RunPod if Nebius unavailable
candidates:
  - cloud: nebius
    accelerators: H100:4
  - cloud: runpod
    accelerators: H100:4
```

### Key SkyPilot Features

1. **Auto-failover**: Falls back to RunPod if Nebius has no capacity
2. **Spot recovery**: Auto-restart on preemption (up to 3 retries)
3. **InfiniBand**: `network_tier: best` provisions 400Gb/s where available
4. **Storage**: `MOUNT_CACHED` mode for 9.6x faster checkpoint writes
5. **Cost optimization**: Automatically selects cheapest available region

---

## Decision Matrix

| Use Case | Recommended Provider |
|----------|---------------------|
| Development/testing | RunPod (A10G spot) |
| Single-GPU training | Nebius H100 |
| Multi-GPU FSDP (4-8 GPUs) | Nebius H100 |
| Large distributed (16+ GPUs) | CoreWeave (if budget allows) |
| Cost-sensitive batch jobs | RunPod spot |

---

## Cost Estimates

### WrinkleFree Training Scenarios

| Scenario | GPUs | Duration | Nebius Cost | RunPod Cost |
|----------|------|----------|-------------|-------------|
| Stage 2 (SmolLM2-135M) | 1x H100 | ~4 hrs | ~$8 | ~$10 |
| Stage 2 (Qwen3-4B) | 4x H100 | ~24 hrs | ~$192 | ~$230 |
| Stage 3 distillation | 4x H100 | ~12 hrs | ~$96 | ~$115 |

*Estimates based on on-demand pricing. Spot instances reduce costs 50-70%.*

---

## Sources

- [H100 Rental Prices: Cloud Cost Comparison (Nov 2025)](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [Top 12 Cloud GPU Providers 2025](https://www.runpod.io/articles/guides/top-cloud-gpu-providers)
- [SkyPilot + Nebius Setup](https://docs.skypilot.co/en/latest/cloud-setup/cloud-permissions/nebius.html)
- [Nebius SkyPilot Integration](https://docs.nebius.com/3p-integrations/skypilot)
- [CoreWeave SkyPilot Blog](https://www.coreweave.com/blog/coreweave-adds-skypilot-support-for-effortless-multi-cloud-ai-orchestration)
