# Tutorials

Step-by-step guides for training and serving 1.58-bit quantized models.

## Training Tutorials

| Tutorial | Time | Description |
|----------|------|-------------|
| [Smoke Test](training/01-smoke-test.md) | 15 min | Validate training pipeline works |

## Serving Tutorials

| Tutorial | Time | Description |
|----------|------|-------------|
| [Local Testing](serving/01-local-testing.md) | 15 min | Test everything locally before using cloud |
| [First Cloud Deployment](serving/02-first-cloud-deployment.md) | 30 min | Deploy to AWS or GCP |
| [Adding Hetzner](serving/03-adding-hetzner.md) | 45 min | Set up cost-effective base layer |
| [Hybrid Setup](serving/04-hybrid-setup.md) | 60 min | Full Hetzner + cloud spillover |
| [Custom Domain](serving/05-custom-domain.md) | 20 min | Use your own domain with Cloudflare |
| [Monitoring](serving/06-monitoring.md) | 30 min | Set up Prometheus + Grafana |
| [Adding OVHCloud](serving/07-adding-ovhcloud.md) | 45 min | Use OVHCloud Kubernetes |

## Prerequisites

Before starting any tutorial:

1. **Clone and install:**
   ```bash
   cd WrinkleFree-Deployer
   uv sync
   ```

2. **Configure credentials:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   source scripts/setup-env.sh
   ```

3. **Verify setup:**
   ```bash
   sky check
   ```

## Which Tutorial Should I Start With?

```
What do you want to do?
│
├── Train a model?
│   └── Start with "Smoke Test" (training/01-smoke-test.md)
│
└── Serve a model?
    │
    ├── New to this?
    │   └── Start with "Local Testing" (serving/01-local-testing.md)
    │
    ├── Have cloud experience?
    │   └── Start with "First Cloud Deployment"
    │
    └── Want cost optimization?
        └── "Adding Hetzner" → "Hybrid Setup"
```

## Provider Comparison

| Provider | Type | Billing | Best For |
|----------|------|---------|----------|
| **Hetzner Dedicated** | SSH Node Pool | Monthly | Always-on base layer (cheapest) |
| **OVHCloud MKS** | Kubernetes | Hourly | EU compliance, GPU |
| **AWS/GCP** | Native | Hourly | Maximum elasticity |
| **RunPod** | Native | Hourly | Training (GPU availability) |
