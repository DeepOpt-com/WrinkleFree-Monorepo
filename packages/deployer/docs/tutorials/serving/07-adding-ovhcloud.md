# Tutorial: Adding OVHCloud

Integrate OVHCloud Managed Kubernetes Service (MKS) as a cost-effective compute tier.

**Time:** ~45 minutes
**Cost:** Free control plane + node costs (~$0.03-0.50/hr depending on instance)
**Requirements:** OVHCloud account, `kubectl` installed

## What You'll Learn

- How OVHCloud integrates with SkyPilot via Kubernetes
- How to set up OVHCloud MKS cluster
- How to configure GPU nodes
- How to add OVHCloud to your hybrid architecture

## How It Works

SkyPilot doesn't have native OVHCloud support, but it **fully supports Kubernetes clusters**. OVHCloud's Managed Kubernetes Service (MKS) lets us use OVHCloud as a SkyPilot backend:

```
SkyPilot
    │
    ├── Native clouds (AWS, GCP, Azure...)
    │
    ├── SSH Node Pools (Hetzner dedicated)
    │
    └── Kubernetes ◄── OVHCloud MKS
                        (via kubeconfig)
```

## Why OVHCloud?

### Pricing Comparison

| Instance | vCPUs | RAM | GPU | OVHCloud | AWS |
|----------|-------|-----|-----|----------|-----|
| b3-32 | 8 | 32GB | - | ~$0.08/hr | ~$0.30/hr |
| b3-64 | 16 | 64GB | - | ~$0.16/hr | ~$0.60/hr |
| t2-45 | 8 | 45GB | T4 | ~$0.45/hr | ~$0.80/hr |
| t2-90 | 16 | 90GB | 2×T4 | ~$0.90/hr | ~$1.60/hr |

**Key benefits:**
- **Free control plane** - You only pay for worker nodes
- **Predictable pricing** - Bandwidth included (no surprise egress costs)
- **European data centers** - GDPR compliance, low latency for EU users
- **GPU availability** - Often better GPU availability than hyperscalers

## Prerequisites

1. OVHCloud account at [ovhcloud.com](https://www.ovhcloud.com)
2. Public Cloud project created
3. `kubectl` installed locally
4. SkyPilot installed (`uv sync`)

## Step 1: Create OVHCloud Public Cloud Project

### Access the Console

1. Go to [OVHcloud Control Panel](https://www.ovh.com/manager/)
2. Navigate to **Public Cloud**
3. Click **Create a project** (if you don't have one)

### Note Your Project ID

```
Project ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

You'll need this for API access.

## Step 2: Create Kubernetes Cluster

### Via OVHCloud Console

1. Go to **Public Cloud** → **Managed Kubernetes Service**
2. Click **Create a Kubernetes cluster**
3. Configure:
   - **Region:** Choose based on your users (e.g., `GRA7` for Gravelines, France)
   - **Version:** Latest stable (1.28+)
   - **Network:** Private network (recommended) or public
4. Click **Create**

### Via CLI (Optional)

Install OVHCloud CLI:

```bash
# Install ovhcloud CLI
pip install ovh

# Or use their dedicated CLI
curl -fsSL https://cli.ovh.net/install.sh | bash
```

Create cluster:

```bash
ovh cloud project $PROJECT_ID kube create \
  --name inference-cluster \
  --region GRA7 \
  --version 1.28
```

### Wait for Cluster

The cluster takes 5-10 minutes to provision. Check status in the console or:

```bash
ovh cloud project $PROJECT_ID kube list
```

## Step 3: Add Node Pools

### Create CPU Node Pool

For CPU-based inference (1.58-bit models):

1. In your cluster dashboard, click **Node pools** → **Add a node pool**
2. Configure:
   - **Name:** `cpu-inference`
   - **Flavor:** `b3-64` (16 vCPUs, 64GB RAM) or `b3-128` (32 vCPUs, 128GB RAM)
   - **Autoscaling:** Enable
     - Min nodes: 1
     - Max nodes: 5
   - **Anti-affinity:** Enable (spreads pods across physical hosts)

3. Click **Create**

### Create GPU Node Pool (Optional)

For GPU inference or mixed workloads:

1. Click **Add a node pool**
2. Configure:
   - **Name:** `gpu-inference`
   - **Flavor:** `t2-45` (T4 GPU) or `t2-90` (2×T4)
   - **Autoscaling:** Enable
     - Min nodes: 0 (scale to zero when idle)
     - Max nodes: 3

3. Click **Create**

### Install NVIDIA GPU Operator (for GPU nodes)

If using GPU nodes, install the NVIDIA GPU operator:

```bash
# Add NVIDIA Helm repo
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true
```

## Step 4: Get Kubeconfig

### Download from Console

1. Go to your cluster dashboard
2. Click **kubeconfig file** → **Download**
3. Save as `~/.kube/ovhcloud-config`

### Or Via CLI

```bash
ovh cloud project $PROJECT_ID kube kubeconfig \
  --cluster-id $CLUSTER_ID > ~/.kube/ovhcloud-config
```

### Merge with Existing Kubeconfig

```bash
# Backup existing config
cp ~/.kube/config ~/.kube/config.backup

# Merge configs
KUBECONFIG=~/.kube/config:~/.kube/ovhcloud-config kubectl config view --flatten > ~/.kube/config.merged
mv ~/.kube/config.merged ~/.kube/config

# Verify contexts
kubectl config get-contexts
```

### Verify Connection

```bash
# Switch to OVHCloud context
kubectl config use-context kubernetes-admin@inference-cluster

# Verify nodes
kubectl get nodes

# Should show your worker nodes:
# NAME                  STATUS   ROLES    AGE   VERSION
# node-pool-xxx-yyy     Ready    <none>   5m    v1.28.x
```

## Step 5: Configure SkyPilot

### Update SkyPilot Config

Create or update `~/.sky/config.yaml`:

```yaml
kubernetes:
  # Allow SkyPilot to use OVHCloud cluster
  allowed_contexts:
    - kubernetes-admin@inference-cluster  # Your OVHCloud context name

  # Networking mode
  networking: portforward  # or 'nodeport' if you have LoadBalancer

  # Pod configuration
  pod_config:
    spec:
      containers:
        - resources:
            requests:
              cpu: "4"
              memory: "16Gi"
```

### Verify SkyPilot Sees Kubernetes

```bash
# Check SkyPilot configuration
sky check

# Should show:
# Kubernetes: enabled
#   Context: kubernetes-admin@inference-cluster
#   GPU: NVIDIA T4 (if GPU nodes exist)
```

## Step 6: Deploy to OVHCloud

### Create OVHCloud-Specific Service Config

```bash
cat > skypilot/service-ovhcloud.yaml << 'EOF'
# Deploy to OVHCloud Kubernetes
name: ovhcloud-inference

service:
  readiness_probe:
    path: /health
    initial_delay_seconds: 120

  replica_policy:
    min_replicas: 1
    max_replicas: 5
    target_qps_per_replica: 5.0

resources:
  # Force Kubernetes (OVHCloud)
  cloud: kubernetes

  # Or specify context explicitly
  # kubernetes:
  #   context: kubernetes-admin@inference-cluster

  ports: 8080
  cpus: 8+
  memory: 32+

  # For GPU workloads
  # accelerators: T4:1

file_mounts:
  /models:
    source: ./models
    mode: COPY

envs:
  MODEL_PATH: /models/model.gguf
  PORT: "8080"

setup: |
  pip install -r requirements-inference.txt

run: |
  python3 -m inference_server --model $MODEL_PATH --port $PORT
EOF
```

### Deploy

```bash
# Deploy to OVHCloud Kubernetes
sky serve up skypilot/service-ovhcloud.yaml --name ovhcloud-test

# Check status
sky serve status ovhcloud-test --all
```

## Step 7: Integrate with Hybrid Architecture

Now add OVHCloud to your multi-tier architecture:

```
Tier 1: Hetzner Dedicated (SSH Node Pools)
    ↓ overflow
Tier 2: OVHCloud MKS (Kubernetes) ◄── NEW
    ↓ overflow
Tier 3: AWS/GCP Spot (native clouds)
```

### Create Multi-Provider Service Config

```bash
cat > skypilot/service-hybrid-ovh.yaml << 'EOF'
name: production-hybrid

service:
  readiness_probe:
    path: /health
    initial_delay_seconds: 120

  replica_policy:
    min_replicas: 3
    max_replicas: 20
    target_qps_per_replica: 5.0

resources:
  ports: 8080
  cpus: 16+
  memory: 64+
  use_spot: true

  # Priority order: cheapest first
  any_of:
    # 1. Hetzner dedicated (cheapest - already paid)
    - cloud: ssh

    # 2. OVHCloud Kubernetes (cheap hourly)
    - cloud: kubernetes
      kubernetes:
        context: kubernetes-admin@inference-cluster

    # 3. AWS spot (elastic)
    - cloud: aws
      use_spot: true

    # 4. GCP spot (elastic)
    - cloud: gcp
      use_spot: true

file_mounts:
  /models:
    source: ./models
    mode: COPY

envs:
  MODEL_PATH: /models/model.gguf
  PORT: "8080"

run: |
  python3 -m inference_server --model $MODEL_PATH --port $PORT
EOF
```

## Step 8: Set Up Autoscaling

### Configure Cluster Autoscaler

OVHCloud MKS has built-in autoscaling. Ensure it's enabled:

1. Go to your node pool in the console
2. Enable **Autoscaling**
3. Set min/max nodes

### Configure SkyPilot for Kubernetes Autoscaling

Update `~/.sky/config.yaml`:

```yaml
kubernetes:
  allowed_contexts:
    - kubernetes-admin@inference-cluster

  # Tell SkyPilot about the autoscaler
  autoscaler: generic  # Works with OVHCloud MKS autoscaler

  # Timeout for autoscaler to provision nodes
  provision_timeout: 600  # 10 minutes
```

### Verify Autoscaling

```bash
# Scale up demand
sky serve update production-hybrid --min-replicas 5

# Watch nodes scale
watch kubectl get nodes

# You should see new nodes appearing
```

## Step 9: Persistent Storage (Optional)

### Use OVHCloud Block Storage

For model caching across pod restarts:

```yaml
# In service.yaml
file_mounts:
  /models:
    source: ./models
    mode: COPY

  # Persistent volume for cache
  /cache:
    name: model-cache
    store: kubernetes
    mode: MOUNT
```

### Create PVC

```bash
cat > ovhcloud-pvc.yaml << 'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: csi-cinder-high-speed  # OVHCloud fast storage
EOF

kubectl apply -f ovhcloud-pvc.yaml
```

## Troubleshooting

### "No nodes available" error

```bash
# Check node pool status
kubectl get nodes

# Check if autoscaler is scaling
kubectl describe nodes | grep -A5 "Conditions"

# Check pending pods
kubectl get pods -A | grep Pending
```

**Common causes:**
- Autoscaler min=0 and no traffic
- Node pool at max capacity
- Flavor not available in region

### "GPU not detected"

```bash
# Check GPU operator status
kubectl get pods -n gpu-operator

# Check GPU resources
kubectl describe nodes | grep nvidia

# If missing, reinstall GPU operator
helm upgrade gpu-operator nvidia/gpu-operator -n gpu-operator
```

### "Connection refused" from SkyPilot

```bash
# Verify kubeconfig
kubectl cluster-info

# Check context in SkyPilot
sky check

# Try refreshing kubeconfig
ovh cloud project $PROJECT_ID kube kubeconfig \
  --cluster-id $CLUSTER_ID > ~/.kube/ovhcloud-config
```

### Pods stuck in "Pending"

```bash
# Check events
kubectl describe pod <pod-name>

# Common causes:
# - Insufficient resources: Scale up node pool
# - PVC not bound: Check storage class
# - GPU requested but none available: Add GPU node pool
```

## Cost Optimization

### Use Savings Plans

OVHCloud offers savings plans for predictable discounts:

1. Go to **Public Cloud** → **Savings Plans**
2. Select instance types you use
3. Commit to 1 or 3 years for 25-50% discount

### Scale to Zero

Configure node pool to scale to zero when idle:

```yaml
# Node pool config
autoscaling:
  enabled: true
  min_nodes: 0  # Scale to zero
  max_nodes: 5
  scale_down_delay: 10m  # Wait before scaling down
```

### Monitor Costs

```bash
# OVHCloud CLI
ovh cloud project $PROJECT_ID usage current

# Or check console: Public Cloud → Project Management → Billing
```

## Summary

You've learned how to:

1. Create OVHCloud MKS cluster
2. Add CPU and GPU node pools
3. Configure SkyPilot to use Kubernetes
4. Deploy services to OVHCloud
5. Integrate with hybrid architecture

**Architecture with OVHCloud:**
```
                    SkyServe
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
Hetzner           OVHCloud MKS         AWS/GCP
(dedicated)       (Kubernetes)          (spot)
$150/mo fixed     ~$0.08-0.45/hr       ~$0.15-0.80/hr
```

**Key commands:**
```bash
# Check Kubernetes connection
kubectl get nodes

# Verify SkyPilot sees it
sky check

# Deploy to OVHCloud
sky serve up skypilot/service-ovhcloud.yaml --name my-service

# Check status
sky serve status my-service --all
```

## Sources

- [SkyPilot Kubernetes Documentation](https://docs.skypilot.co/en/latest/reference/kubernetes/index.html)
- [SkyPilot Kubernetes Setup](https://docs.skypilot.co/en/latest/reference/kubernetes/kubernetes-setup.html)
- [OVHCloud Managed Kubernetes](https://www.ovhcloud.com/en/public-cloud/kubernetes/)
- [OVHCloud GPU on Kubernetes](https://blog.ovhcloud.com/using-gpu-on-managed-kubernetes-service-with-nvidia-gpu-operator/)
- [SkyPilot Multi-Kubernetes Clusters](https://docs.skypilot.co/en/latest/reference/kubernetes/multi-kubernetes.html)
