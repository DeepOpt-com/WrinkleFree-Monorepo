# Troubleshooting Guide

Common issues and how to fix them.

## Quick Diagnostics

Run these commands first to understand what's wrong:

```bash
# Check cloud credentials
sky check

# Check service status
sky serve status

# Check specific service
sky serve status my-service --all

# View logs
sky serve logs my-service

# View controller logs
sky serve logs my-service --controller
```

---

## Setup Issues

### "sky: command not found"

**Cause:** SkyPilot not installed or not in PATH.

**Fix:**
```bash
# Install with uv
uv sync

# Or install directly
pip install "skypilot[aws,gcp]"

# Verify
which sky
sky --version
```

### "sky check shows no clouds enabled"

**Cause:** Cloud credentials not configured.

**Fix:**
```bash
# Make sure .env exists
ls -la .env

# Load environment
source scripts/setup-env.sh

# Or manually set credentials
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Check again
sky check
```

### "Permission denied" when running setup script

**Cause:** Script not executable.

**Fix:**
```bash
chmod +x scripts/setup-env.sh
source scripts/setup-env.sh
```

### "SSH key not found"

**Cause:** SSH key doesn't exist at the specified path.

**Fix:**
```bash
# Generate a new SSH key
ssh-keygen -t ed25519 -f ~/.ssh/hetzner_ed25519

# Make sure it's in your .env
echo "HETZNER_SSH_KEY_PATH=~/.ssh/hetzner_ed25519" >> .env

# Copy to your server
ssh-copy-id -i ~/.ssh/hetzner_ed25519 root@YOUR_SERVER_IP
```

---

## SSH Node Pool Issues

### "sky ssh up" hangs or times out

**Cause:** Can't connect to your servers.

**Fix:**
1. Verify you can SSH manually:
   ```bash
   ssh -i ~/.ssh/hetzner_ed25519 root@YOUR_SERVER_IP
   ```

2. Check firewall allows SSH (port 22):
   ```bash
   # On the server
   sudo ufw status
   sudo ufw allow ssh
   ```

3. Check your IP hasn't changed:
   ```bash
   cat ~/.sky/ssh_node_pools.yaml
   ```

### "Host key verification failed"

**Cause:** Server's SSH fingerprint changed (reinstall, etc.).

**Fix:**
```bash
# Remove old key
ssh-keygen -R YOUR_SERVER_IP

# Reconnect (will prompt to accept new key)
ssh -i ~/.ssh/hetzner_ed25519 root@YOUR_SERVER_IP
```

### "sky check ssh" shows nodes as unavailable

**Cause:** Nodes not properly initialized.

**Fix:**
```bash
# Tear down and reinitialize
sky ssh down
sky ssh up

# Check logs for errors
sky ssh up --verbose
```

---

## Service Deployment Issues

### "sky serve up" fails immediately

**Cause:** Usually a YAML syntax error or missing required fields.

**Fix:**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('skypilot/service.yaml'))"

# Check required fields exist:
# - service.readiness_probe
# - resources.ports
```

### Service stuck in "PENDING" state

**Cause:** Can't find resources matching your requirements.

**Check:**
```bash
sky serve status my-service --all
sky serve logs my-service --controller
```

**Common causes and fixes:**

| Cause | Fix |
|-------|-----|
| Insufficient quota | Request limit increase from cloud provider |
| Instance type unavailable | Try different region or instance type |
| Spot capacity unavailable | Set `use_spot: false` or wait |

**Quick fix - reduce requirements:**
```yaml
resources:
  cpus: 4+        # Lower from 16+
  memory: 32+     # Lower from 128+
  use_spot: false # Use on-demand instead
```

### Service stuck in "PROVISIONING" for long time

**Cause:** Large setup script or slow model download.

**Fix:**
```bash
# Check what's happening
sky serve logs my-service --replica-id 0

# Common issues:
# - Model download slow: Use faster storage (S3/GCS)
# - Compilation slow: Use pre-built containers
```

### Replicas not becoming READY

**Cause:** Health check failing.

**Check:**
```bash
# View replica logs
sky serve logs my-service --replica-id 0

# Look for:
# - Port mismatch (service not on expected port)
# - Model loading errors
# - Out of memory
```

**Common fixes:**

1. **Port mismatch:**
   ```yaml
   resources:
     ports: 8080  # Must match what your service listens on

   envs:
     PORT: "8080"  # Make sure these match!
   ```

2. **Not enough time to load model:**
   ```yaml
   readiness_probe:
     initial_delay_seconds: 300  # Increase from 120
   ```

3. **Out of memory:**
   ```yaml
   resources:
     memory: 256+  # Increase memory requirement
   ```

---

## Runtime Issues

### High latency / slow responses

**Cause:** Replicas overloaded.

**Check:**
```bash
sky serve status my-service --all
# Look at QPS per replica
```

**Fix:**
```bash
# Scale up
sky serve update my-service --min-replicas 5

# Or adjust target QPS (lower = more replicas)
# Edit service.yaml:
replica_policy:
  target_qps_per_replica: 3.0  # Lower from 5.0
```

### "Connection refused" when calling endpoint

**Cause:** Service not ready or wrong endpoint.

**Fix:**
```bash
# Get correct endpoint
sky serve status my-service

# Check replicas are READY
sky serve status my-service --all

# If no replicas ready, check logs
sky serve logs my-service --replica-id 0
```

### Spot instance interruptions

**Cause:** Cloud provider reclaimed spot capacity.

**This is normal!** SkyServe handles it automatically.

**Check:**
```bash
# See if replicas are being replaced
sky serve status my-service --all

# You should see new replicas starting
```

**If it's happening too often:**
```yaml
# Option 1: Use on-demand for base capacity
resources:
  use_spot: false

# Option 2: Use fallback strategy
spot_recovery: FAILOVER  # Try different regions
```

### Out of memory errors

**Cause:** Model too large for instance.

**Check:**
```bash
sky serve logs my-service --replica-id 0
# Look for "OOM", "Killed", "Cannot allocate memory"
```

**Fix:**
```yaml
resources:
  memory: 256+  # Increase memory requirement

# Or use memory locking
envs:
  MLOCK: "true"  # Prevent swapping
```

---

## Model Issues

### "Model file not found"

**Cause:** File not mounted or wrong path.

**Check your service.yaml:**
```yaml
file_mounts:
  /models:
    source: ./models  # Does this directory exist?
    mode: COPY

envs:
  MODEL_PATH: /models/model.gguf  # Does this file exist in ./models?
```

**Fix:**
```bash
# Check model exists locally
ls -la models/

# Make sure it's a valid GGUF file
file models/model.gguf
```

### "Invalid model format"

**Cause:** Model file corrupted or wrong format.

**Fix:**
```bash
# Verify file integrity
md5sum models/model.gguf  # Compare with expected hash

# Re-download if corrupted
./scripts/download_test_model.sh
```

### Model loading very slow

**Cause:** Downloading on every replica.

**Fix - use cloud storage:**
```yaml
file_mounts:
  /models:
    source: s3://your-bucket/models/  # Much faster!
    mode: COPY
```

---

## Networking Issues

### Can't reach endpoint from outside

**Cause:** Firewall blocking traffic.

**For SkyServe endpoints:**
- SkyServe endpoints should work automatically
- Check if your network allows outbound HTTPS

**For direct instance access:**
```bash
# Check security group/firewall
# AWS:
aws ec2 describe-security-groups

# GCP:
gcloud compute firewall-rules list
```

### High latency between regions

**Cause:** Traffic routing through distant regions.

**Fix:**
```yaml
resources:
  region: us-east-1  # Pin to specific region close to users
```

### WireGuard tunnel not working

**Check:**
```bash
# On each node
sudo wg show

# Should see:
# - interface: wg0
# - peer with recent handshake
```

**Common issues:**

1. **Firewall blocking UDP 51820:**
   ```bash
   sudo ufw allow 51820/udp
   ```

2. **Wrong public key:**
   ```bash
   # Regenerate and redistribute keys
   wg genkey | tee privatekey | wg pubkey > publickey
   ```

3. **NAT traversal issues:**
   ```ini
   # Add to [Peer] section
   PersistentKeepalive = 25
   ```

---

## Cost Issues

### Unexpected high costs

**Check what's running:**
```bash
# List all services
sky serve status

# List all clusters
sky status

# Tear down everything
sky serve down --all
sky down --all
```

**Common causes:**

| Cause | Fix |
|-------|-----|
| Forgot to tear down | `sky serve down my-service` |
| min_replicas too high | Reduce min_replicas in YAML |
| Using on-demand instead of spot | Set `use_spot: true` |
| Large disk size | Reduce `disk_size` in YAML |

### Spot savings not as expected

**Check:**
```bash
sky serve status my-service --all
# See if replicas are on spot or on-demand
```

**Fix:**
```yaml
resources:
  use_spot: true
  spot_recovery: FAILOVER  # Use other regions if spot unavailable
```

---

## Getting More Help

### Collect Debug Information

Before asking for help, gather this info:

```bash
# 1. SkyPilot version
sky --version

# 2. Cloud status
sky check

# 3. Service status
sky serve status my-service --all

# 4. Recent logs
sky serve logs my-service > service_logs.txt
sky serve logs my-service --controller > controller_logs.txt

# 5. Your service YAML (remove secrets!)
cat skypilot/service.yaml
```

### Resources

- **SkyPilot Documentation:** https://docs.skypilot.co
- **SkyPilot GitHub Issues:** https://github.com/skypilot-org/skypilot/issues
- **SkyPilot Slack:** https://slack.skypilot.co
- **This Project:** https://github.com/your-org/WrinkleFree-Deployer/issues
