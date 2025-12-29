# Tutorial: Hybrid Setup

Deploy a production-ready hybrid infrastructure with Hetzner base layer and cloud burst capacity.

**Time:** ~60 minutes
**Cost:** Hetzner monthly + cloud usage during testing
**Requirements:** Completed [Adding Hetzner](03-adding-hetzner.md) + cloud credentials

## What You'll Learn

- How to configure spillover from Hetzner to cloud
- How to set up autoscaling policies
- How to handle spot instance interruptions
- How to optimize for cost vs. reliability

## The Hybrid Architecture

```
                        Your Users
                            │
                            ▼
                    ┌───────────────┐
                    │   SkyServe    │
                    │   Endpoint    │
                    │  (Load Bal)   │
                    └───────┬───────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   Hetzner     │ │   Hetzner     │ │   AWS/GCP     │
    │   Replica 1   │ │   Replica 2   │ │   Spot        │
    │   (base)      │ │   (base)      │ │   (burst)     │
    └───────────────┘ └───────────────┘ └───────────────┘
         Always on        Always on       Scales up/down
          $150/mo          $150/mo          ~$0.15/hr
```

## Prerequisites

1. [Adding Hetzner](03-adding-hetzner.md) completed
2. Cloud credentials configured (AWS or GCP)
3. Model files ready in `models/` directory

## Step 1: Plan Your Capacity

### Understand Your Traffic

First, estimate your baseline and peak traffic:

```
Questions to answer:
│
├── What's your baseline traffic?
│   → e.g., 50 requests/second 24/7
│
├── What's your peak traffic?
│   → e.g., 200 requests/second during business hours
│
├── How quickly do peaks happen?
│   → e.g., Traffic ramps up over 5 minutes
│
└── How much latency is acceptable?
    → e.g., P99 < 500ms
```

### Capacity Planning

Each replica can handle a certain QPS (queries per second):

```
Example: 1.58-bit 7B model on Hetzner AX102
  → ~5-10 QPS per replica (depending on prompt length)

Your needs:
  Baseline: 50 QPS → Need ~5-10 Hetzner replicas
  Peak: 200 QPS → Need ~20-40 total replicas

Plan:
  Hetzner base: 2 servers × 2 replicas each = 4 replicas (~40 QPS)
  Cloud burst: Up to 20 spot replicas for peaks
```

## Step 2: Configure Hetzner Base Layer

### Ensure Hetzner Nodes Are Ready

```bash
# Check Hetzner nodes are available
sky ssh status

# Should show:
# NODE POOL      NODES   STATUS
# hetzner-base   2       READY
```

### Create Hybrid Service Configuration

```bash
cat > skypilot/service-hybrid.yaml << 'EOF'
# Hybrid deployment: Hetzner base + cloud burst
name: hybrid-inference

service:
  readiness_probe:
    path: /health
    initial_delay_seconds: 120
    timeout_seconds: 10

  replica_policy:
    # Base capacity (Hetzner handles this)
    min_replicas: 2

    # Burst capacity (cloud handles overflow)
    max_replicas: 20

    # When to scale up
    # Lower = more aggressive scaling = higher cost but better latency
    target_qps_per_replica: 5.0

    # Scale down delay (avoid flapping)
    scale_down_delay_seconds: 300  # Wait 5 min before scaling down

resources:
  ports: 8080
  cpus: 16+
  memory: 128+

  # Use spot instances for cloud (cheaper)
  use_spot: true

  # If spot interrupted, try different regions
  spot_recovery: FAILOVER

file_mounts:
  /models:
    source: ./models
    mode: COPY

envs:
  MODEL_PATH: /models/model.gguf
  PORT: "8080"
  # Adjust based on your model
  NUM_THREADS: "16"
  CONTEXT_SIZE: "4096"

setup: |
  # Install inference dependencies
  pip install -r requirements-inference.txt

run: |
  # Start the inference server
  python3 -m bitnet_server \
    --model $MODEL_PATH \
    --port $PORT \
    --threads $NUM_THREADS \
    --ctx-size $CONTEXT_SIZE
EOF
```

## Step 3: Understand SkyServe's Provider Selection

SkyServe automatically chooses where to place replicas:

```
When you request a new replica:
│
├── Check SSH Node Pool (Hetzner)
│   └── Has capacity? → Use it (cost: $0 marginal)
│
├── If Hetzner full, check Spot instances
│   └── Available? → Use it (cost: ~$0.15/hr)
│
└── If no spot, check On-demand
    └── Use as last resort (cost: ~$0.50/hr)
```

### Priority Configuration (Optional)

If you want explicit control, you can set priorities:

```yaml
# In service.yaml (advanced)
resources:
  # This tells SkyPilot to try SSH (Hetzner) first
  cloud: ssh

  # Fallback options if SSH is full
  any_of:
    - cloud: ssh    # Try Hetzner first
    - cloud: aws    # Then AWS spot
      use_spot: true
    - cloud: gcp    # Then GCP spot
      use_spot: true
```

## Step 4: Deploy the Hybrid Service

### Deploy

```bash
# Deploy with hybrid configuration
sky serve up skypilot/service-hybrid.yaml --name production

# This will:
# 1. Start min_replicas (2) on cheapest option (Hetzner)
# 2. Set up load balancer
# 3. Begin monitoring for scaling
```

**Expected output:**
```
Launching service 'production'...
Finding resources for 2 replicas...
  Replica 0: Using SSH node pool (hetzner-base)
  Replica 1: Using SSH node pool (hetzner-base)

Provisioning...
  Replica 0: Initializing on hetzner-base
  Replica 1: Initializing on hetzner-base

Service 'production' is READY!
Endpoint: https://production-abc123.sky.serve
Replicas: 2/2 ready (min: 2, max: 20)
```

### Verify Placement

```bash
# Check where replicas are running
sky serve status production --all

# Output shows each replica's location:
# REPLICA_ID  STATUS  CLOUD   REGION  INSTANCE
# 0           READY   ssh     -       hetzner-base
# 1           READY   ssh     -       hetzner-base
```

## Step 5: Test Autoscaling

### Generate Load

Let's send traffic to trigger scaling:

```bash
# Install load testing tool
pip install locust

# Or use our built-in load test
uv run python scripts/run_tests.py --suite load --duration 60 --users 50
```

### Watch Scaling in Action

In another terminal:

```bash
# Watch replicas scale
watch -n 5 'sky serve status production --all'
```

**What you should see:**

```
# Initial state (low traffic)
REPLICA_ID  STATUS  CLOUD   INSTANCE
0           READY   ssh     hetzner-base
1           READY   ssh     hetzner-base

# Under load (scaling up)
REPLICA_ID  STATUS       CLOUD   INSTANCE
0           READY        ssh     hetzner-base
1           READY        ssh     hetzner-base
2           STARTING     aws     r7a.4xlarge (spot)
3           STARTING     aws     r7a.4xlarge (spot)

# After load (scaling down, after 5 min delay)
REPLICA_ID  STATUS  CLOUD   INSTANCE
0           READY   ssh     hetzner-base
1           READY   ssh     hetzner-base
```

### View Scaling Logs

```bash
# See autoscaling decisions
sky serve logs production --controller

# Look for messages like:
# [AutoScaler] QPS: 45.2, target: 5.0/replica, current: 2 replicas
# [AutoScaler] Scaling up: need 9 replicas for current load
# [AutoScaler] Requesting 7 new replicas...
```

## Step 6: Handle Spot Interruptions

Spot instances can be reclaimed by cloud providers. SkyServe handles this automatically.

### How It Works

```
Spot interruption happens:
│
├── Cloud provider sends 2-min warning
│
├── SkyServe receives notification
│
├── SkyServe starts replacement replica
│   └── Tries different region/zone
│
├── Old replica terminates
│
└── Traffic continues on remaining replicas
    (slight capacity reduction for ~2 min)
```

### Test Interruption Handling (Optional)

```bash
# Simulate a spot interruption (AWS)
# This terminates a spot instance to see recovery

# Get instance ID from status
sky serve status production --all

# Terminate a spot replica manually
aws ec2 terminate-instances --instance-ids i-xxxxx

# Watch recovery
watch -n 5 'sky serve status production --all'
```

### Configure Interruption Behavior

```yaml
# In service.yaml
resources:
  use_spot: true

  # Options for handling interruptions:
  spot_recovery: FAILOVER
  # FAILOVER: Try different region/zone
  # FAILOVER_NO_OPTIMIZER: Try anywhere available
```

## Step 7: Optimize Cost vs. Reliability

### Cost-Optimized (Default)

Prioritize lowest cost, accept brief capacity drops during interruptions:

```yaml
resources:
  use_spot: true
  spot_recovery: FAILOVER

replica_policy:
  min_replicas: 2        # Hetzner handles base
  max_replicas: 20       # Cloud handles burst
  target_qps_per_replica: 5.0
```

**Cost:** ~$300/month base + variable cloud
**Reliability:** 99.5% (brief drops during interruptions)

### Reliability-Optimized

Keep spare capacity for interruption resilience:

```yaml
resources:
  use_spot: true
  spot_recovery: FAILOVER

replica_policy:
  min_replicas: 4        # Extra capacity as buffer
  max_replicas: 25
  target_qps_per_replica: 4.0  # Lower = more headroom
```

**Cost:** ~$400/month base + variable cloud
**Reliability:** 99.9%

### High Reliability (Mixed Spot + On-Demand)

Use on-demand for critical base capacity:

```yaml
# Create two resource pools
resources:
  any_of:
    # Hetzner (always available)
    - cloud: ssh

    # On-demand for guaranteed capacity
    - cloud: aws
      use_spot: false

    # Spot for cost-effective burst
    - cloud: aws
      use_spot: true

replica_policy:
  min_replicas: 4        # Some will be on-demand
  max_replicas: 30
```

**Cost:** ~$500/month base + variable cloud
**Reliability:** 99.99%

## Step 8: Monitor the Hybrid Setup

### Check Current Status

```bash
# Overall service status
sky serve status production

# Detailed replica info
sky serve status production --all

# View metrics
sky serve logs production --controller | grep -i "qps\|replica\|scale"
```

### Key Metrics to Watch

| Metric | Healthy Range | Action if Outside |
|--------|---------------|-------------------|
| QPS per replica | 3-7 | Adjust target_qps_per_replica |
| Replica count | min - max | Check if scaling is working |
| P99 latency | < 500ms | Add more base capacity |
| Spot interruptions | < 1/hour | Consider more regions |

### Set Up Alerts (See [Monitoring Tutorial](06-monitoring.md))

```bash
# Quick health check script
cat > scripts/check_health.sh << 'EOF'
#!/bin/bash
ENDPOINT=$(sky serve status production | grep "Endpoint:" | awk '{print $2}')
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$ENDPOINT/health")
if [ "$RESPONSE" != "200" ]; then
  echo "ALERT: Service unhealthy! HTTP $RESPONSE"
  exit 1
fi
echo "OK: Service healthy"
EOF
chmod +x scripts/check_health.sh
```

## Step 9: Update the Service

### Rolling Updates

When you need to update your model or configuration:

```bash
# Update service configuration
sky serve update production skypilot/service-hybrid.yaml

# This performs a rolling update:
# 1. Starts new replicas with new config
# 2. Waits for them to be healthy
# 3. Drains old replicas
# 4. Terminates old replicas
```

### Blue-Green Deployment (Zero Downtime)

For major changes:

```bash
# Deploy new version as separate service
sky serve up skypilot/service-hybrid.yaml --name production-v2

# Test new version
curl https://production-v2-xxx.sky.serve/health

# Switch traffic (in your DNS/load balancer)
# ...

# Tear down old version
sky serve down production
```

## Troubleshooting

### Replicas not scaling up

```bash
# Check controller logs
sky serve logs production --controller

# Common causes:
# - Cloud quota limits
# - No spot capacity in region
# - Scaling delay hasn't elapsed

# Fix: Try different regions
# Add to service.yaml:
resources:
  region: us-east-1,us-west-2,eu-west-1
```

### Replicas not scaling down

```bash
# Check scale_down_delay_seconds
# Default wait is 5 minutes after load drops

# Check if there's still traffic
sky serve logs production --controller | grep QPS
```

### Hetzner replicas not being used

```bash
# Check SSH node pool status
sky ssh status

# If showing "NOT READY", reinitialize:
sky ssh down
sky ssh up
```

### High latency during scaling

During scale-up, new replicas take time to become ready:

```yaml
# Reduce initial delay if your model loads fast
service:
  readiness_probe:
    initial_delay_seconds: 60  # Reduce from 120
```

## Summary

You've learned how to:

1. Plan capacity for hybrid deployment
2. Configure Hetzner as base layer
3. Set up cloud burst with spot instances
4. Handle spot interruptions gracefully
5. Optimize for cost vs. reliability
6. Monitor and update the service

**Architecture:**
```
                    SkyServe
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    Hetzner        Hetzner        AWS Spot
    (always on)    (always on)    (elastic)
```

**Key commands:**
```bash
# Deploy hybrid service
sky serve up skypilot/service-hybrid.yaml --name production

# Check replica placement
sky serve status production --all

# View scaling logs
sky serve logs production --controller

# Update service
sky serve update production skypilot/service-hybrid.yaml
```

## What's Next?

- **Custom domain** → [Custom Domain](05-custom-domain.md)
- **Monitoring** → [Monitoring](06-monitoring.md)
- **Cost optimization** → Fine-tune `target_qps_per_replica`
