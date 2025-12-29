# Hybrid Networking with WireGuard

Secure, low-latency networking between Hetzner (base layer) and AWS/GCP (burst layer) using WireGuard VPN.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              WireGuard Mesh Network                              │
│                                  10.100.0.0/16                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────┐         ┌─────────────────────────────┐
    │     Hetzner (Base Layer)    │         │    AWS/GCP (Burst Layer)    │
    │        10.100.1.0/24        │         │       10.100.2.0/24         │
    │                             │         │                             │
    │  ┌───────┐    ┌───────┐    │         │    ┌───────┐    ┌───────┐  │
    │  │ Node1 │    │ Node2 │    │         │    │ Spot1 │    │ Spot2 │  │
    │  │ .1    │    │ .2    │    │   WG    │    │ .1    │    │ .2    │  │
    │  └───┬───┘    └───┬───┘    │◄───────►│    └───┬───┘    └───┬───┘  │
    │      │            │        │ Tunnel  │        │            │      │
    │      └─────┬──────┘        │         │        └─────┬──────┘      │
    │            │               │         │              │             │
    │     ┌──────▼──────┐        │         │       ┌──────▼──────┐      │
    │     │  WG Gateway │        │         │       │  WG Gateway │      │
    │     │  10.100.1.254│◄──────┼─────────┼──────►│  10.100.2.254│     │
    │     │  (hetzner-gw)│        │         │       │  (aws-gw)   │      │
    │     └──────────────┘        │         │       └─────────────┘      │
    │                             │         │                             │
    └─────────────────────────────┘         └─────────────────────────────┘
              │                                           │
              │         Public Internet                   │
              └───────────────────────────────────────────┘
```

## Why WireGuard?

| Feature | Benefit |
|---------|---------|
| **Low latency** | ~1ms overhead, critical for inference routing |
| **Simple config** | Single config file per node |
| **Auto-reconnect** | Handles cloud instance restarts |
| **NAT traversal** | Works behind cloud NAT gateways |
| **Modern crypto** | ChaCha20, Curve25519, no legacy overhead |

## Architecture Options

### Option 1: Hub-and-Spoke (Recommended for Start)

One gateway node routes all cross-cloud traffic.

```
                    ┌─────────────────┐
                    │  Hetzner GW     │
                    │  (Hub)          │
                    │  10.100.0.1     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │Hetzner 1│        │ AWS 1   │        │ GCP 1   │
    │10.100.1.1│       │10.100.2.1│       │10.100.3.1│
    └─────────┘        └─────────┘        └─────────┘
```

**Pros:** Simple, easy to manage, single point of control
**Cons:** Hub is single point of failure, potential bottleneck

### Option 2: Full Mesh (Production)

Every node connects to every other node.

```
    ┌─────────┐◄──────────────────────►┌─────────┐
    │Hetzner 1│                        │ AWS 1   │
    │10.100.1.1│◄─────────┐    ┌──────►│10.100.2.1│
    └─────────┘           │    │       └─────────┘
         ▲                │    │             ▲
         │                │    │             │
         │           ┌────▼────▼────┐        │
         │           │   Hetzner 2  │        │
         └──────────►│  10.100.1.2  │◄───────┘
                     └──────────────┘
```

**Pros:** No single point of failure, optimal routing
**Cons:** More complex, O(n²) connections

### Option 3: Tailscale/Netbird (Managed)

Use a managed WireGuard service for easier operations.

**Pros:** Zero-config, automatic key distribution, web UI
**Cons:** Dependency on external service, cost at scale

## Setup Guide: Hub-and-Spoke

### 1. Install WireGuard

```bash
# On all nodes (Ubuntu 22.04)
sudo apt update
sudo apt install -y wireguard wireguard-tools
```

### 2. Generate Keys

```bash
# On each node
wg genkey | tee privatekey | wg pubkey > publickey

# Store keys securely (use Vault, AWS Secrets Manager, etc.)
cat privatekey  # Keep secret!
cat publickey   # Share with peers
```

### 3. Configure Hub (Hetzner Gateway)

```ini
# /etc/wireguard/wg0.conf on Hetzner Gateway

[Interface]
Address = 10.100.0.1/16
ListenPort = 51820
PrivateKey = <HETZNER_GW_PRIVATE_KEY>

# Enable IP forwarding
PostUp = sysctl -w net.ipv4.ip_forward=1
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT
PostUp = iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT
PostDown = iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

# Hetzner Node 1
[Peer]
PublicKey = <HETZNER_NODE1_PUBLIC_KEY>
AllowedIPs = 10.100.1.1/32

# Hetzner Node 2
[Peer]
PublicKey = <HETZNER_NODE2_PUBLIC_KEY>
AllowedIPs = 10.100.1.2/32

# AWS Node 1 (dynamic IP, so we allow any endpoint)
[Peer]
PublicKey = <AWS_NODE1_PUBLIC_KEY>
AllowedIPs = 10.100.2.1/32

# AWS Node 2
[Peer]
PublicKey = <AWS_NODE2_PUBLIC_KEY>
AllowedIPs = 10.100.2.2/32
```

### 4. Configure Spoke (Inference Nodes)

```ini
# /etc/wireguard/wg0.conf on Hetzner inference node

[Interface]
Address = 10.100.1.1/32
PrivateKey = <NODE_PRIVATE_KEY>

# Connect to hub
[Peer]
PublicKey = <HETZNER_GW_PUBLIC_KEY>
Endpoint = <HETZNER_GW_PUBLIC_IP>:51820
AllowedIPs = 10.100.0.0/16
PersistentKeepalive = 25
```

```ini
# /etc/wireguard/wg0.conf on AWS spot instance

[Interface]
Address = 10.100.2.1/32
PrivateKey = <NODE_PRIVATE_KEY>

# Connect to hub
[Peer]
PublicKey = <HETZNER_GW_PUBLIC_KEY>
Endpoint = <HETZNER_GW_PUBLIC_IP>:51820
AllowedIPs = 10.100.0.0/16
PersistentKeepalive = 25
```

### 5. Start WireGuard

```bash
# Enable and start
sudo systemctl enable wg-quick@wg0
sudo systemctl start wg-quick@wg0

# Verify connection
sudo wg show

# Test connectivity
ping 10.100.0.1  # Hub
ping 10.100.1.2  # Other Hetzner node
ping 10.100.2.1  # AWS node
```

## Terraform Integration

### Hetzner Gateway Provisioning

```hcl
# terraform/hetzner/wireguard.tf

resource "hcloud_server" "wg_gateway" {
  name        = "wrinklefree-wg-gateway"
  server_type = "cx22"  # Small, cheap gateway
  image       = "ubuntu-22.04"
  location    = "fsn1"

  ssh_keys = [hcloud_ssh_key.deploy.id]

  user_data = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y wireguard

    # Generate keys if not exists
    if [ ! -f /etc/wireguard/privatekey ]; then
      wg genkey | tee /etc/wireguard/privatekey | wg pubkey > /etc/wireguard/publickey
      chmod 600 /etc/wireguard/privatekey
    fi

    # Config will be templated separately
    systemctl enable wg-quick@wg0
  EOF

  labels = {
    role = "wireguard-gateway"
  }
}

resource "hcloud_firewall" "wg_gateway" {
  name = "wireguard-gateway"

  rule {
    direction = "in"
    protocol  = "udp"
    port      = "51820"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    port      = "22"
    source_ips = [var.admin_ip]
  }
}

output "wg_gateway_ip" {
  value = hcloud_server.wg_gateway.ipv4_address
}
```

### AWS Security Group

```hcl
# terraform/aws/wireguard.tf

resource "aws_security_group" "wireguard" {
  name        = "wrinklefree-wireguard"
  description = "WireGuard VPN traffic"
  vpc_id      = var.vpc_id

  # WireGuard UDP
  ingress {
    from_port   = 51820
    to_port     = 51820
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow all traffic from WireGuard network
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["10.100.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "wrinklefree-wireguard"
  }
}
```

## SkyPilot Integration

### Using WireGuard IPs in SSH Node Pools

Once WireGuard is running, register nodes by their WireGuard IPs:

```yaml
# ~/.sky/ssh_node_pools.yaml

# Hetzner nodes via WireGuard
hetzner-base:
  user: root
  identity_file: ~/.ssh/hetzner_ed25519
  hosts:
    - ip: 10.100.1.1  # WireGuard IP, not public IP
    - ip: 10.100.1.2
    - ip: 10.100.1.3

# AWS nodes (if persistent) via WireGuard
aws-reserved:
  user: ubuntu
  identity_file: ~/.ssh/aws_key
  hosts:
    - ip: 10.100.2.1
    - ip: 10.100.2.2
```

### SkyServe with WireGuard Networking

```yaml
# skypilot/service-wireguard.yaml

service:
  readiness_probe: /health
  replica_policy:
    min_replicas: 3
    max_replicas: 10

resources:
  ports: 8080
  cpus: 16+
  memory: 256+

envs:
  # Use WireGuard network for inter-node communication
  CLUSTER_NETWORK: "10.100.0.0/16"

setup: |
  # Verify WireGuard connectivity
  ping -c 1 10.100.0.1 || echo "Warning: WireGuard hub not reachable"

run: |
  python -m inference_server --host 0.0.0.0 --port 8080
```

## Load Balancer Configuration

### Cloudflare with WireGuard Origins

Cloudflare connects to your WireGuard gateway, which routes to inference nodes:

```
Internet → Cloudflare → WG Gateway (public IP) → WG Network → Inference Nodes
```

**Cloudflare Origin Configuration:**
- Origin: `https://<WG_GATEWAY_PUBLIC_IP>:443`
- Health check: `GET /health` via gateway

**Gateway Nginx Config:**

```nginx
# /etc/nginx/sites-available/inference-lb

upstream inference_nodes {
    least_conn;

    # Hetzner nodes (WireGuard IPs)
    server 10.100.1.1:8080 weight=10;
    server 10.100.1.2:8080 weight=10;
    server 10.100.1.3:8080 weight=10;

    # AWS burst nodes (WireGuard IPs)
    server 10.100.2.1:8080 weight=5 backup;
    server 10.100.2.2:8080 weight=5 backup;
}

server {
    listen 443 ssl http2;
    server_name inference.example.com;

    ssl_certificate /etc/letsencrypt/live/inference.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/inference.example.com/privkey.pem;

    location / {
        proxy_pass http://inference_nodes;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 5s;
        proxy_read_timeout 120s;
    }

    location /health {
        proxy_pass http://inference_nodes/health;
        proxy_connect_timeout 2s;
        proxy_read_timeout 5s;
    }
}
```

## Dynamic Node Registration

### Auto-Register AWS Spot Instances

When AWS spot instances launch, they should auto-register with WireGuard:

```bash
#!/bin/bash
# /opt/wireguard-register.sh (run on AWS instance boot)

set -e

WG_HUB_IP="${WG_HUB_IP:-<HETZNER_GW_PUBLIC_IP>}"
WG_HUB_PUBKEY="${WG_HUB_PUBKEY:-<HETZNER_GW_PUBLIC_KEY>}"

# Get instance metadata
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
PRIVATE_IP=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)

# Derive WireGuard IP from instance ID (simple hash)
WG_SUFFIX=$(echo "$INSTANCE_ID" | md5sum | head -c 4)
WG_IP="10.100.2.$((16#${WG_SUFFIX} % 250 + 1))"

# Generate keys
wg genkey | tee /etc/wireguard/privatekey | wg pubkey > /etc/wireguard/publickey
chmod 600 /etc/wireguard/privatekey

PRIVATE_KEY=$(cat /etc/wireguard/privatekey)
PUBLIC_KEY=$(cat /etc/wireguard/publickey)

# Create config
cat > /etc/wireguard/wg0.conf <<EOF
[Interface]
Address = ${WG_IP}/32
PrivateKey = ${PRIVATE_KEY}

[Peer]
PublicKey = ${WG_HUB_PUBKEY}
Endpoint = ${WG_HUB_IP}:51820
AllowedIPs = 10.100.0.0/16
PersistentKeepalive = 25
EOF

# Start WireGuard
systemctl start wg-quick@wg0

# Register with hub (call API or use Consul/etcd)
curl -X POST "http://${WG_HUB_IP}:8500/v1/agent/service/register" \
  -H "Content-Type: application/json" \
  -d "{
    \"ID\": \"inference-${INSTANCE_ID}\",
    \"Name\": \"inference\",
    \"Address\": \"${WG_IP}\",
    \"Port\": 8080,
    \"Tags\": [\"aws\", \"spot\"],
    \"Check\": {
      \"HTTP\": \"http://${WG_IP}:8080/health\",
      \"Interval\": \"10s\"
    }
  }"

echo "Registered as ${WG_IP}"
```

## Monitoring WireGuard

### Prometheus Metrics

```yaml
# prometheus/wireguard-exporter.yaml

apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: wireguard-exporter
spec:
  selector:
    matchLabels:
      app: wireguard-exporter
  template:
    spec:
      containers:
        - name: exporter
          image: mindflavor/prometheus-wireguard-exporter
          ports:
            - containerPort: 9586
          securityContext:
            capabilities:
              add: ["NET_ADMIN"]
          volumeMounts:
            - name: wireguard
              mountPath: /etc/wireguard
              readOnly: true
      volumes:
        - name: wireguard
          hostPath:
            path: /etc/wireguard
```

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `wireguard_peer_last_handshake_seconds` | Time since last handshake | > 180s |
| `wireguard_peer_transfer_bytes_total` | Data transferred | Anomaly detection |
| `wireguard_peer_allowed_ips_count` | Number of allowed IPs | Unexpected changes |

### Health Check Script

```bash
#!/bin/bash
# /opt/wireguard-health.sh

# Check hub connectivity
if ! ping -c 1 -W 2 10.100.0.1 > /dev/null 2>&1; then
    echo "ERROR: Cannot reach WireGuard hub"
    exit 1
fi

# Check peer handshakes (should be within 3 minutes)
STALE_PEERS=$(wg show wg0 latest-handshakes | awk -v now=$(date +%s) '$2 < now - 180 {print $1}')
if [ -n "$STALE_PEERS" ]; then
    echo "WARNING: Stale peers: $STALE_PEERS"
    exit 1
fi

# Check interface is up
if ! ip link show wg0 | grep -q "state UP"; then
    echo "ERROR: WireGuard interface down"
    exit 1
fi

echo "OK"
exit 0
```

## Troubleshooting

### Connection Issues

```bash
# Check WireGuard status
sudo wg show

# Check interface
ip addr show wg0

# Test connectivity
ping -c 3 10.100.0.1

# Check firewall
sudo iptables -L -n | grep 51820

# Debug with tcpdump
sudo tcpdump -i eth0 udp port 51820

# Check logs
sudo journalctl -u wg-quick@wg0 -f
```

### Common Problems

| Symptom | Cause | Solution |
|---------|-------|----------|
| No handshake | Firewall blocking UDP 51820 | Open port on both ends |
| Handshake but no traffic | AllowedIPs misconfigured | Check IP ranges |
| Intermittent drops | NAT timeout | Add PersistentKeepalive |
| High latency | Routing through hub | Consider mesh topology |

### Key Rotation

```bash
# Generate new keys
wg genkey | tee /etc/wireguard/privatekey.new | wg pubkey > /etc/wireguard/publickey.new

# Update config (requires updating all peers)
# Then atomic swap:
sudo wg set wg0 private-key /etc/wireguard/privatekey.new

# Distribute new public key to all peers
# Reload peer configs
```

## Security Best Practices

1. **Key Management**: Use HashiCorp Vault or AWS Secrets Manager for key storage
2. **Network Segmentation**: Separate WireGuard network from management network
3. **Firewall Rules**: Only allow WireGuard traffic from known IPs where possible
4. **Audit Logging**: Log all peer connections and disconnections
5. **Regular Rotation**: Rotate keys quarterly or after any compromise
6. **Minimum Privileges**: Inference nodes only need routes to load balancer and each other
