# Tutorial: Adding Hetzner

Set up Hetzner dedicated servers as your cost-effective base layer.

**Time:** ~45 minutes
**Cost:** ~$50-150/month (server rental)
**Requirements:** Hetzner account, SSH key

## What You'll Learn

- How to order a Hetzner dedicated server
- How to configure SSH access
- How to register the server with SkyPilot
- How to verify everything works

## Why Hetzner?

### Cost Comparison

| Provider | 256GB RAM Server | Monthly Cost |
|----------|------------------|--------------|
| AWS r7a.8xlarge | On-demand | ~$1,080/month |
| AWS r7a.8xlarge | Spot (avg) | ~$400/month |
| Hetzner AX102 | Dedicated | ~$150/month |

**That's 3-7x cheaper** for always-on base capacity!

### When to Use Hetzner

```
Your traffic pattern:
│
├── Predictable baseline (e.g., 100 req/sec constant)
│   → Use Hetzner for this part (cheap, fixed cost)
│
└── Variable spikes (e.g., bursts to 500 req/sec)
    → Use cloud spot instances (elastic, pay-per-use)
```

## Prerequisites

1. Complete [First Cloud Deployment](02-first-cloud-deployment.md) first
2. Have a Hetzner account at [hetzner.com](https://www.hetzner.com)
3. Have payment method configured in Hetzner

## Step 1: Order a Dedicated Server

### Choose Your Server

Go to [Hetzner Dedicated Servers](https://www.hetzner.com/dedicated-rootserver).

**Recommended for inference:**

| Server | CPUs | RAM | Price | Best For |
|--------|------|-----|-------|----------|
| AX42 | 8 cores | 64GB | ~$50/mo | Small models (<7B) |
| AX52 | 12 cores | 128GB | ~$80/mo | Medium models (7-13B) |
| AX102 | 24 cores | 256GB | ~$150/mo | Large models (30B+) |

### Order Process

1. **Select server** at hetzner.com/dedicated-rootserver
2. **Choose OS:** Ubuntu 22.04 LTS (recommended)
3. **Select datacenter:**
   - FSN1 (Falkenstein) - Germany
   - NBG1 (Nuremberg) - Germany
   - HEL1 (Helsinki) - Finland

   *Tip: Choose based on where your users are*

4. **Submit order**
   - Takes 1-24 hours to provision
   - You'll receive an email with server IP and root password

### Wait for Provisioning

Check your email for:
```
Subject: Your dedicated root server is ready

Server IP: 123.45.67.89
Root password: xxxxxxxx
```

## Step 2: Initial Server Setup

### First Login

```bash
# SSH into your new server
ssh root@YOUR_SERVER_IP
# Enter the password from the email

# You'll be prompted to change the password
# Choose a strong password!
```

### Generate SSH Key (on your local machine)

```bash
# Generate a new SSH key for Hetzner
ssh-keygen -t ed25519 -f ~/.ssh/hetzner_ed25519 -C "hetzner-deployer"

# Don't set a passphrase for automation (or use ssh-agent)
# Press Enter twice for no passphrase
```

### Copy SSH Key to Server

```bash
# Copy your public key to the server
ssh-copy-id -i ~/.ssh/hetzner_ed25519 root@YOUR_SERVER_IP

# Test passwordless login
ssh -i ~/.ssh/hetzner_ed25519 root@YOUR_SERVER_IP
# Should log in without password prompt!
```

### Secure the Server (Recommended)

```bash
# On the server:

# Update packages
apt update && apt upgrade -y

# Disable password authentication (SSH key only)
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# Set up basic firewall
ufw allow ssh
ufw allow 8080/tcp  # Inference API port
ufw enable
```

## Step 3: Install Server Dependencies

The server needs some packages for SkyPilot and inference.

```bash
# On the server:

# Install Python
apt install -y python3 python3-pip python3-venv

# Install Docker (for containerized inference)
curl -fsSL https://get.docker.com | bash
systemctl enable docker
systemctl start docker

# Verify installations
python3 --version  # Should be 3.10+
docker --version   # Should be 24.x+
```

## Step 4: Update Your .env File

On your **local machine**, update `.env` with server details:

```bash
# Edit your .env file
nano .env
```

Add or update:

```bash
# Hetzner Configuration
HETZNER_SERVER_IPS=YOUR_SERVER_IP
HETZNER_SSH_KEY_PATH=~/.ssh/hetzner_ed25519
HETZNER_SSH_USER=root
```

For multiple servers:
```bash
HETZNER_SERVER_IPS=10.0.1.100,10.0.1.101,10.0.1.102
```

## Step 5: Register with SkyPilot

### Run Setup Script

```bash
# This configures SkyPilot SSH Node Pool
source scripts/setup-env.sh
```

The script will:
1. Load your credentials from `.env`
2. Create `~/.sky/ssh_node_pools.yaml`
3. Run `sky check` to verify

### Verify SSH Node Pool

```bash
# Check SkyPilot sees your servers
sky check

# Should show:
# SSH: enabled
#   hetzner-base: 1 node(s)
```

### Initialize the Node Pool

```bash
# This installs SkyPilot runtime on your Hetzner server
sky ssh up

# This takes 5-10 minutes the first time
# SkyPilot installs its agent and dependencies
```

**Expected output:**
```
Initializing SSH node pool 'hetzner-base'...
  Node 123.45.67.89:
    Installing dependencies...
    Installing SkyPilot runtime...
    Verifying connection...

Node pool 'hetzner-base' ready with 1 node(s)
```

### Verify Node is Available

```bash
# Check node status
sky ssh status

# Should show:
# NODE POOL      NODES   STATUS
# hetzner-base   1       READY
```

## Step 6: Test Deployment to Hetzner

Let's deploy a test service specifically to Hetzner.

### Create Hetzner-Only Config

```bash
cat > skypilot/service-hetzner.yaml << 'EOF'
# Service that only runs on Hetzner
service:
  readiness_probe:
    path: /health
    initial_delay_seconds: 60

  replica_policy:
    min_replicas: 1
    max_replicas: 1  # Just testing

resources:
  ports: 8080
  cpus: 4+
  memory: 32+
  # Force deployment to SSH node pool (Hetzner)
  cloud: ssh

file_mounts:
  /models:
    source: ./models/test
    mode: COPY

envs:
  MODEL_PATH: /models/smollm2-135m.gguf
  PORT: "8080"

run: |
  # Start inference server
  cd /models
  python3 -m http.server 8080 &
  sleep 5
  # Simple health endpoint
  while true; do
    echo '{"status": "healthy"}' | nc -l -p 8080 -q 1
  done
EOF
```

### Deploy to Hetzner

```bash
# Deploy specifically to Hetzner
sky serve up skypilot/service-hetzner.yaml --name hetzner-test
```

### Verify Deployment

```bash
# Check status
sky serve status hetzner-test --all

# Should show replica running on SSH node pool
```

### Test the Endpoint

```bash
# Get endpoint URL
sky serve status hetzner-test

# Test health
curl https://hetzner-test-xxx.sky.serve/health
```

### Clean Up Test

```bash
sky serve down hetzner-test
```

## Step 7: Understand How SkyServe Uses Hetzner

When you deploy with SkyServe, it automatically considers Hetzner:

```
sky serve up skypilot/service.yaml
         │
         ▼
    SkyServe evaluates all available resources:
         │
         ├── Hetzner (SSH Node Pool)
         │   Cost: $0/hr (already paid monthly)
         │   Priority: HIGH (cheapest option)
         │
         ├── AWS Spot
         │   Cost: ~$0.10-0.50/hr
         │   Priority: MEDIUM
         │
         └── AWS On-Demand
             Cost: ~$0.50-2.00/hr
             Priority: LOW (most expensive)
         │
         ▼
    SkyServe chooses cheapest option that has capacity
```

### Priority Order

SkyServe automatically prefers cheaper options:

1. **First:** SSH Node Pool (Hetzner) - $0 marginal cost
2. **Then:** Spot instances - cheap but interruptible
3. **Last:** On-demand - expensive but guaranteed

## Step 8: Adding More Servers (Optional)

### Order Additional Servers

Order more servers from Hetzner as needed.

### Update .env

```bash
# Add new IPs
HETZNER_SERVER_IPS=10.0.1.100,10.0.1.101,10.0.1.102
```

### Re-run Setup

```bash
# Reload configuration
source scripts/setup-env.sh

# Initialize new nodes
sky ssh up

# Verify all nodes
sky ssh status
# Should show: hetzner-base   3   READY
```

## Troubleshooting

### "Connection refused" to server

```bash
# Check SSH works manually
ssh -i ~/.ssh/hetzner_ed25519 root@YOUR_SERVER_IP -v

# If firewall blocking:
# Use Hetzner Robot panel to access rescue console
# Then: ufw allow ssh
```

### "sky ssh up" hangs

```bash
# Try with verbose output
sky ssh up --verbose

# Common causes:
# - Firewall blocking
# - Wrong SSH key path
# - Server not fully booted yet
```

### "Node not available" after setup

```bash
# Re-initialize the node pool
sky ssh down
sky ssh up

# Check node health
sky ssh status
```

### Server keeps getting OOM killed

Your model is too large for the server's RAM.

```bash
# Check server memory
ssh root@YOUR_SERVER_IP free -h

# Options:
# 1. Use a smaller model
# 2. Upgrade to bigger server
# 3. Enable swap (slower but works):
ssh root@YOUR_SERVER_IP "fallocate -l 32G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile"
```

## Cost Analysis

### Break-Even Calculation

When does Hetzner make sense vs. spot instances?

```
Hetzner AX102: $150/month fixed

AWS r7a.4xlarge spot: ~$0.15/hour
Break-even: $150 / $0.15 = 1000 hours

Hours per month: 720

If you need the server 720+ hours/month:
  → Hetzner is cheaper

If you need it < 720 hours/month:
  → Spot instances might be cheaper
```

**Rule of thumb:** If you need always-on capacity, Hetzner wins.

## What's Next?

Now that you have Hetzner set up:

- **Full hybrid setup** → [Hybrid Setup](04-hybrid-setup.md) (Hetzner + cloud spillover)
- **Custom domain** → [Custom Domain](05-custom-domain.md)
- **Monitoring** → [Monitoring](06-monitoring.md)

## Summary

You've learned how to:

1. Order a Hetzner dedicated server
2. Configure SSH access with key-based auth
3. Install required dependencies
4. Register the server as a SkyPilot SSH Node Pool
5. Deploy services to Hetzner
6. Understand cost implications

**Key commands:**
```bash
# Initialize Hetzner nodes
source scripts/setup-env.sh
sky ssh up

# Check node status
sky ssh status

# Deploy (SkyServe auto-selects Hetzner when available)
sky serve up skypilot/service.yaml --name my-model

# Tear down node pool
sky ssh down
```

**Cost tip:** Hetzner servers are paid monthly - keep them utilized!
