# Tutorial: First Cloud Deployment

Deploy your model to the cloud with SkyServe.

**Time:** ~30 minutes
**Cost:** ~$0.50-2.00 (can be stopped anytime)
**Requirements:** AWS or GCP account with credentials configured

## What You'll Learn

- How to configure cloud credentials
- How to deploy with SkyServe
- How to monitor your deployment
- How to tear down when done

## Prerequisites

Complete [Local Testing](01-local-testing.md) first to make sure your setup works.

## Step 1: Configure Credentials

### Create .env File

```bash
# Copy the template
cp .env.example .env

# Open in your editor
nano .env  # or vim, code, etc.
```

### Add AWS Credentials

Get your credentials from AWS Console → IAM → Security Credentials → Access Keys.

```bash
# In .env file:
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
```

### Or Add GCP Credentials

1. Go to GCP Console → IAM → Service Accounts
2. Create a new service account with "Compute Admin" role
3. Download the JSON key file

```bash
# In .env file:
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account.json
```

### Load Credentials

```bash
# Run the setup script
source scripts/setup-env.sh

# Verify
sky check
```

You should see:
```
Checking credentials...
  ✓ AWS credentials found

AWS: enabled
  ...
```

## Step 2: Prepare Your Model

Make sure you have a model file:

```bash
# Check model exists
ls -la models/

# For testing, use the small model from local testing
ls models/test/smollm2-135m.gguf
```

## Step 3: Review the Service Configuration

Look at `skypilot/service.yaml`:

```bash
cat skypilot/service.yaml
```

Key settings to understand:

```yaml
# How SkyServe manages your deployment
service:
  readiness_probe:
    path: /health           # URL to check if ready
    initial_delay_seconds: 120  # Wait before first check

  replica_policy:
    min_replicas: 1         # Always have at least 1
    max_replicas: 3         # Scale up to 3 if needed
    target_qps_per_replica: 5.0  # Target requests/second

# What resources each replica needs
resources:
  ports: 8080
  cpus: 4+                  # At least 4 CPUs
  memory: 16+               # At least 16GB RAM
  use_spot: true            # Use spot instances (cheaper!)
```

### For a Cheaper First Test

Edit `skypilot/service.yaml` to use minimal resources:

```yaml
resources:
  cpus: 2+                  # Reduce CPU requirement
  memory: 8+                # Reduce memory requirement
  use_spot: true

replica_policy:
  min_replicas: 1           # Just one replica for testing
  max_replicas: 1
```

## Step 4: Deploy!

```bash
# Deploy the service
sky serve up skypilot/service.yaml --name my-first-model

# This will:
# 1. Find the cheapest instance matching your requirements
# 2. Create the instance
# 3. Copy your model files
# 4. Run setup script
# 5. Start the inference server
# 6. Return an endpoint URL
```

**Expected output:**
```
Launching service 'my-first-model'...
Finding the cheapest resources...
  Using AWS us-east-1, r7a.large ($0.18/hr spot)
Provisioning...
  Instance i-0123456789abcdef launched
Setting up...
  Running setup script...
Starting service...
  Waiting for health check...

Service 'my-first-model' is READY!
Endpoint: https://my-first-model-abc123.sky.serve
```

## Step 5: Wait and Monitor

The deployment takes 5-10 minutes. Watch the progress:

```bash
# Check status (refresh every 10 seconds)
watch -n 10 sky serve status my-first-model

# Or check once
sky serve status my-first-model --all
```

**Status progression:**
```
PENDING → PROVISIONING → STARTING → READY
```

If stuck in any state, check logs:

```bash
# View replica logs
sky serve logs my-first-model --replica-id 0

# View controller logs
sky serve logs my-first-model --controller
```

## Step 6: Test Your Deployment

Get the endpoint:

```bash
sky serve status my-first-model
# Note the "Endpoint:" URL
```

Test it:

```bash
# Health check
curl https://my-first-model-abc123.sky.serve/health

# Inference request
curl https://my-first-model-abc123.sky.serve/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, I am an AI assistant. How can I help you today?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Expected response:**
```json
{
  "id": "completion-xxx",
  "object": "text_completion",
  "choices": [{
    "text": " I can help you with a variety of tasks...",
    "index": 0,
    "finish_reason": "length"
  }]
}
```

## Step 7: Understand Your Costs

Check what you're paying:

```bash
# See active resources
sky status

# Example output:
# CLUSTER        CLOUD    RESOURCES                 STATUS   COST
# sky-serve-xxx  aws      r7a.large (spot)          UP       $0.05/hr
```

**Cost breakdown:**
- Controller: ~$0.01/hr (small instance)
- Replica: ~$0.05-0.20/hr (depends on instance type)
- Data transfer: ~$0.09/GB (only when sending responses)

**For this tutorial:** ~$0.10-0.50 total if you complete in <1 hour.

## Step 8: Clean Up (Important!)

**Don't forget this step** - instances cost money!

```bash
# Tear down the service
sky serve down my-first-model

# Confirm
sky serve status
# Should show: No services running

# Double-check no clusters running
sky status
# Should show: No clusters running
```

## Troubleshooting

### "No resources available"

```bash
# Try a different region
# Edit service.yaml:
resources:
  region: us-west-2  # Try different region
```

### "Quota exceeded"

You need to request a limit increase from AWS/GCP:
- AWS: EC2 → Limits → Request limit increase
- GCP: IAM → Quotas → Request increase

### "Health check failing"

```bash
# Check replica logs
sky serve logs my-first-model --replica-id 0

# Common issues:
# - Model not found: Check file_mounts in service.yaml
# - Out of memory: Increase memory requirement
# - Wrong port: Check PORT environment variable
```

### "Service taking too long"

```bash
# Check what's happening
sky serve logs my-first-model --replica-id 0

# If stuck on model download, consider:
# 1. Using a smaller model for testing
# 2. Pre-uploading model to S3/GCS
```

## What's Next?

Now that you have a cloud deployment working:

- **Want cheaper?** → [Adding Hetzner](03-adding-hetzner.md) (up to 7x cheaper)
- **Want to scale?** → Edit `max_replicas` in service.yaml
- **Want your own domain?** → [Custom Domain](05-custom-domain.md)

## Summary

You've learned how to:

1. ✅ Configure cloud credentials
2. ✅ Deploy with SkyServe
3. ✅ Monitor deployment status
4. ✅ Test your endpoint
5. ✅ Understand costs
6. ✅ Clean up resources

**Key commands to remember:**
```bash
# Deploy
sky serve up skypilot/service.yaml --name my-model

# Check status
sky serve status my-model --all

# View logs
sky serve logs my-model --replica-id 0

# Tear down (DON'T FORGET!)
sky serve down my-model
```

**Cost tip:** Always run `sky serve down` when done testing!
