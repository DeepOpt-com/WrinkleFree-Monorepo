# Tutorial: Custom Domain

Use your own domain (like `api.yourcompany.com`) instead of the default SkyServe URL.

**Time:** ~20 minutes
**Cost:** Domain registration (~$10/year) + Cloudflare (free tier)
**Requirements:** A domain name, Cloudflare account

## What You'll Learn

- How to set up Cloudflare for your domain
- How to point your domain to SkyServe
- How to add SSL/TLS encryption
- How to configure caching and security

## Why Use a Custom Domain?

### Benefits

| Feature | SkyServe URL | Custom Domain |
|---------|--------------|---------------|
| URL | `https://my-model-abc123.sky.serve` | `https://api.yourcompany.com` |
| Branding | Generic | Professional |
| SSL | Automatic | Automatic (via Cloudflare) |
| Caching | None | Configurable |
| DDoS protection | Basic | Cloudflare protection |
| Rate limiting | None | Configurable |

### Architecture

```
Users
  │
  ▼
api.yourcompany.com (Cloudflare)
  │
  ├── SSL termination
  ├── DDoS protection
  ├── Caching (optional)
  │
  ▼
https://my-model-xxx.sky.serve (SkyServe)
  │
  ▼
Your replicas (Hetzner + Cloud)
```

## Prerequisites

1. A running SkyServe deployment
2. A domain name (purchase from Namecheap, Google Domains, etc.)
3. Cloudflare account (free tier works)

## Step 1: Set Up Cloudflare

### Create Cloudflare Account

1. Go to [cloudflare.com](https://cloudflare.com)
2. Sign up for a free account
3. Click "Add a Site"

### Add Your Domain

1. Enter your domain (e.g., `yourcompany.com`)
2. Select the **Free** plan
3. Cloudflare will scan your existing DNS records

### Update Nameservers

Cloudflare will give you two nameservers like:
```
chad.ns.cloudflare.com
diana.ns.cloudflare.com
```

Go to your domain registrar and update the nameservers:
- **Namecheap:** Domain List → Manage → Nameservers → Custom DNS
- **Google Domains:** DNS → Custom name servers
- **GoDaddy:** DNS Management → Nameservers → Change

**Wait 10-60 minutes** for DNS propagation.

### Verify Setup

```bash
# Check nameservers have propagated
dig NS yourcompany.com

# Should show cloudflare.com nameservers
```

## Step 2: Get Your SkyServe Endpoint

```bash
# Get your current SkyServe endpoint
sky serve status my-model

# Note the endpoint URL:
# Endpoint: https://my-model-abc123.sky.serve
```

## Step 3: Create DNS Record

### Option A: CNAME Record (Recommended)

In Cloudflare Dashboard:

1. Go to **DNS** → **Records**
2. Click **Add record**
3. Configure:
   - **Type:** CNAME
   - **Name:** `api` (creates `api.yourcompany.com`)
   - **Target:** `my-model-abc123.sky.serve` (without https://)
   - **Proxy status:** **Proxied** (orange cloud)
   - **TTL:** Auto

4. Click **Save**

### Option B: Full Domain

To use `yourcompany.com` (no subdomain):

1. **Type:** CNAME
2. **Name:** `@` (root domain)
3. **Target:** `my-model-abc123.sky.serve`
4. **Proxy status:** Proxied

## Step 4: Configure SSL/TLS

### Enable Full SSL

In Cloudflare Dashboard:

1. Go to **SSL/TLS** → **Overview**
2. Select **Full (strict)**

This ensures:
- Encryption between users and Cloudflare
- Encryption between Cloudflare and SkyServe

### Enable Always Use HTTPS

1. Go to **SSL/TLS** → **Edge Certificates**
2. Enable **Always Use HTTPS**

## Step 5: Test Your Custom Domain

### Verify DNS

```bash
# Check DNS resolution
dig api.yourcompany.com

# Should show Cloudflare IPs
```

### Test Endpoint

```bash
# Health check
curl https://api.yourcompany.com/health

# Should return: {"status": "healthy"}

# Test inference
curl https://api.yourcompany.com/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20}'
```

## Step 6: Configure Caching (Optional)

### Why Cache?

For repeated identical requests, caching can:
- Reduce latency significantly
- Lower your inference costs
- Handle traffic spikes better

**Note:** Only cache if your API has deterministic outputs (same input = same output).

### Create Cache Rule

1. Go to **Caching** → **Cache Rules**
2. Click **Create rule**
3. Configure:

```
Rule name: Cache completions

When incoming requests match:
  Field: URI Path
  Operator: contains
  Value: /v1/completions

Then:
  Cache eligibility: Eligible for cache
  Edge TTL: Use cache-control header, or 1 hour default
```

### Add Cache Headers in Your App

Modify your inference server to return cache headers:

```python
# In your server code
@app.route('/v1/completions', methods=['POST'])
def completions():
    # ... your inference logic ...

    response = make_response(result)

    # Cache for 1 hour if temperature=0 (deterministic)
    if request.json.get('temperature', 1) == 0:
        response.headers['Cache-Control'] = 'public, max-age=3600'

    return response
```

## Step 7: Add Security Features

### Enable Rate Limiting

Protect against abuse:

1. Go to **Security** → **WAF** → **Rate limiting rules**
2. Click **Create rule**
3. Configure:

```
Rule name: API rate limit

When incoming requests match:
  Field: URI Path
  Operator: starts with
  Value: /v1/

Rate limit:
  Requests: 100
  Period: 1 minute

Action: Block
```

### Enable Bot Protection

1. Go to **Security** → **Bots**
2. Enable **Bot Fight Mode** (free tier)

### Add IP Access Rules (Optional)

If you want to restrict API access:

1. Go to **Security** → **WAF** → **Tools**
2. Add IP Access Rules:
   - Allow your app servers
   - Block everything else

## Step 8: Set Up Multiple Domains (Optional)

### Development/Staging Environments

Create separate subdomains for each environment:

| Environment | Domain | SkyServe Service |
|-------------|--------|------------------|
| Production | `api.yourcompany.com` | `production` |
| Staging | `api-staging.yourcompany.com` | `staging` |
| Development | `api-dev.yourcompany.com` | `dev` |

### Configure Each

```bash
# Deploy staging
sky serve up skypilot/service.yaml --name staging

# Get endpoint
sky serve status staging
# Endpoint: https://staging-xyz789.sky.serve
```

In Cloudflare, add:
- **Name:** `api-staging`
- **Target:** `staging-xyz789.sky.serve`

## Step 9: Updating When Endpoints Change

SkyServe endpoints can change when you:
- Tear down and recreate a service
- Update certain configurations

### Automated DNS Updates

Create a script to update Cloudflare DNS:

```bash
cat > scripts/update-dns.sh << 'EOF'
#!/bin/bash
# Update Cloudflare DNS when SkyServe endpoint changes

# Configuration
CLOUDFLARE_TOKEN="${CLOUDFLARE_API_TOKEN}"
ZONE_ID="${CLOUDFLARE_ZONE_ID}"
RECORD_NAME="api.yourcompany.com"
SERVICE_NAME="${1:-production}"

# Get current SkyServe endpoint
ENDPOINT=$(sky serve status $SERVICE_NAME | grep "Endpoint:" | awk '{print $2}' | sed 's|https://||')

if [ -z "$ENDPOINT" ]; then
    echo "Error: Could not get endpoint for service $SERVICE_NAME"
    exit 1
fi

echo "Updating DNS for $RECORD_NAME -> $ENDPOINT"

# Get record ID
RECORD_ID=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records?name=$RECORD_NAME" \
    -H "Authorization: Bearer $CLOUDFLARE_TOKEN" \
    -H "Content-Type: application/json" | jq -r '.result[0].id')

# Update record
curl -s -X PUT "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records/$RECORD_ID" \
    -H "Authorization: Bearer $CLOUDFLARE_TOKEN" \
    -H "Content-Type: application/json" \
    --data "{
        \"type\": \"CNAME\",
        \"name\": \"$RECORD_NAME\",
        \"content\": \"$ENDPOINT\",
        \"proxied\": true
    }"

echo "DNS updated successfully!"
EOF
chmod +x scripts/update-dns.sh
```

### Add to .env

```bash
# Cloudflare Configuration
CLOUDFLARE_API_TOKEN=your-api-token
CLOUDFLARE_ZONE_ID=your-zone-id
```

### Get Cloudflare Credentials

1. **Zone ID:** Dashboard → Your domain → Overview → (scroll down) → Zone ID
2. **API Token:** My Profile → API Tokens → Create Token → Edit zone DNS template

## Troubleshooting

### "DNS record not found" error

```bash
# Check DNS propagation
dig api.yourcompany.com

# If no results, wait longer for propagation
# Can take up to 48 hours (usually 10-60 minutes)
```

### "SSL handshake failed"

```bash
# Make sure SSL mode is "Full" not "Flexible"
# Cloudflare Dashboard → SSL/TLS → Overview → Full (strict)
```

### "522 Connection timed out"

The origin (SkyServe) isn't responding:

```bash
# Check service is running
sky serve status my-model --all

# Check replicas are healthy
sky serve logs my-model --replica-id 0
```

### "Error 1000: DNS resolution error"

CNAME target is wrong:

```bash
# Make sure target doesn't include https://
# Correct: my-model-abc123.sky.serve
# Wrong: https://my-model-abc123.sky.serve
```

### Custom domain works but is slow

Enable Cloudflare optimizations:

1. **Speed** → **Optimization** → Enable all
2. **Caching** → **Configuration** → Browser Cache TTL: 1 day
3. **Network** → Enable HTTP/3 and 0-RTT

## Summary

You've learned how to:

1. Set up Cloudflare for your domain
2. Create DNS records pointing to SkyServe
3. Configure SSL/TLS encryption
4. Add caching and security features
5. Handle endpoint updates

**Your new architecture:**
```
Users
  │
  ▼
https://api.yourcompany.com
  │ (Cloudflare: SSL, DDoS, caching)
  ▼
https://my-model-xxx.sky.serve
  │ (SkyServe: load balancing)
  ▼
Hetzner + Cloud replicas
```

**Key URLs:**
- **Cloudflare Dashboard:** https://dash.cloudflare.com
- **Your API:** https://api.yourcompany.com

## What's Next?

- **Set up monitoring** → [Monitoring](06-monitoring.md)
- **Configure alerts** → Use Cloudflare Analytics + custom monitoring
- **Add authentication** → Consider API keys or OAuth
