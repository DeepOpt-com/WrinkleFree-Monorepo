# Tutorial: Monitoring

Set up Prometheus and Grafana to monitor your inference service.

**Time:** ~30 minutes
**Cost:** Free (self-hosted) or ~$15/month (Grafana Cloud)
**Requirements:** Docker installed, running SkyServe deployment

## What You'll Learn

- How to collect metrics from your inference service
- How to set up Prometheus for metrics storage
- How to visualize with Grafana dashboards
- How to configure alerts

## Why Monitor?

### What Can Go Wrong

| Problem | Symptoms | Detection |
|---------|----------|-----------|
| High latency | Slow responses | Latency metrics |
| Overloaded replicas | Timeouts, errors | QPS + CPU metrics |
| Memory leak | OOM crashes | Memory metrics |
| Spot interruptions | Capacity drops | Replica count |
| Model errors | Wrong outputs | Error rate |

### Key Metrics to Track

```
                    Request Flow
                         │
  ┌──────────────────────┼──────────────────────┐
  │                      │                      │
  ▼                      ▼                      ▼
Request Rate         Latency              Error Rate
(QPS)               (P50, P99)            (5xx, timeouts)
  │                      │                      │
  └──────────────────────┼──────────────────────┘
                         │
                         ▼
              Resource Utilization
              (CPU, Memory, GPU)
                         │
                         ▼
                  Replica Health
              (count, status, restarts)
```

## Step 1: Expose Metrics from Your Service

### Add Prometheus Metrics to Inference Server

Your inference server should expose metrics at `/metrics`. Here's how to add it:

```python
# metrics.py - Add to your inference server
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time

# Define metrics
REQUEST_COUNT = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'inference_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

TOKENS_GENERATED = Counter(
    'inference_tokens_generated_total',
    'Total tokens generated'
)

MODEL_LOADED = Gauge(
    'inference_model_loaded',
    'Whether the model is loaded (1=yes, 0=no)'
)

ACTIVE_REQUESTS = Gauge(
    'inference_active_requests',
    'Currently processing requests'
)

# Middleware to track requests
class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        start_time = time.time()
        ACTIVE_REQUESTS.inc()

        def custom_start_response(status, headers, exc_info=None):
            status_code = status.split()[0]
            REQUEST_COUNT.labels(endpoint=path, status=status_code).inc()
            return start_response(status, headers, exc_info)

        try:
            return self.app(environ, custom_start_response)
        finally:
            ACTIVE_REQUESTS.dec()
            REQUEST_LATENCY.labels(endpoint=path).observe(time.time() - start_time)

# Metrics endpoint
def metrics_handler():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
```

### Add to Your Server

```python
# server.py
from flask import Flask
from metrics import MetricsMiddleware, metrics_handler, MODEL_LOADED

app = Flask(__name__)
app.wsgi_app = MetricsMiddleware(app.wsgi_app)

@app.route('/metrics')
def metrics():
    return metrics_handler()

# After model loads
MODEL_LOADED.set(1)
```

### Verify Metrics Endpoint

```bash
# Test locally
curl http://localhost:8080/metrics

# Should output Prometheus format:
# # HELP inference_requests_total Total inference requests
# # TYPE inference_requests_total counter
# inference_requests_total{endpoint="/v1/completions",status="200"} 42
# ...
```

## Step 2: Deploy Monitoring Stack

### Create Docker Compose for Monitoring

```bash
cat > docker-compose.monitoring.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.2.0
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
    depends_on:
      - prometheus

  alertmanager:
    image: prom/alertmanager:v0.26.0
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
EOF
```

### Create Prometheus Configuration

```bash
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards

cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - /etc/prometheus/rules/*.yml

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # SkyServe endpoint (via your custom domain or direct)
  - job_name: 'inference-service'
    metrics_path: /metrics
    static_configs:
      - targets:
        # Add your SkyServe endpoint or individual replica IPs
        - 'api.yourcompany.com:443'
    scheme: https
    tls_config:
      insecure_skip_verify: true  # Remove in production

  # Hetzner nodes (direct scraping)
  - job_name: 'hetzner-nodes'
    static_configs:
      - targets:
        - '10.0.1.100:8080'  # Replace with your Hetzner IPs
        - '10.0.1.101:8080'
    # Only if using WireGuard
    # scheme: http
EOF
```

### Create Alertmanager Configuration

```bash
cat > monitoring/alertmanager.yml << 'EOF'
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'

  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'

receivers:
  - name: 'default'
    # Add your notification config

  - name: 'critical-alerts'
    # Slack example:
    # slack_configs:
    #   - api_url: 'https://hooks.slack.com/services/xxx'
    #     channel: '#alerts'
    #     send_resolved: true

    # Email example:
    # email_configs:
    #   - to: 'oncall@yourcompany.com'
    #     from: 'alertmanager@yourcompany.com'
    #     smarthost: 'smtp.gmail.com:587'
    #     auth_username: 'your-email@gmail.com'
    #     auth_password: 'your-app-password'
EOF
```

### Create Grafana Datasource

```bash
cat > monitoring/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
EOF
```

## Step 3: Start Monitoring Stack

```bash
# Start the monitoring stack
docker compose -f docker-compose.monitoring.yml up -d

# Verify all containers are running
docker compose -f docker-compose.monitoring.yml ps

# Check Prometheus is scraping
curl http://localhost:9090/api/v1/targets
```

### Access Dashboards

- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)
- **Alertmanager:** http://localhost:9093

## Step 4: Create Grafana Dashboard

### Import Pre-Built Dashboard

1. Go to Grafana (http://localhost:3000)
2. Login (admin/admin, change password)
3. Click **+** → **Import**
4. Paste this dashboard JSON:

```json
{
  "dashboard": {
    "title": "Inference Service Dashboard",
    "panels": [
      {
        "title": "Request Rate (QPS)",
        "type": "timeseries",
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
        "targets": [{
          "expr": "rate(inference_requests_total[1m])",
          "legendFormat": "{{endpoint}} - {{status}}"
        }]
      },
      {
        "title": "Latency (P99)",
        "type": "timeseries",
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
        "targets": [{
          "expr": "histogram_quantile(0.99, rate(inference_request_duration_seconds_bucket[5m]))",
          "legendFormat": "P99 Latency"
        }]
      },
      {
        "title": "Active Requests",
        "type": "gauge",
        "gridPos": {"x": 0, "y": 8, "w": 6, "h": 6},
        "targets": [{
          "expr": "sum(inference_active_requests)"
        }]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "gridPos": {"x": 6, "y": 8, "w": 6, "h": 6},
        "targets": [{
          "expr": "sum(rate(inference_requests_total{status=~\"5..\"}[5m])) / sum(rate(inference_requests_total[5m])) * 100",
          "legendFormat": "Error %"
        }]
      },
      {
        "title": "Tokens Generated",
        "type": "stat",
        "gridPos": {"x": 12, "y": 8, "w": 6, "h": 6},
        "targets": [{
          "expr": "sum(increase(inference_tokens_generated_total[1h]))",
          "legendFormat": "Tokens/hour"
        }]
      },
      {
        "title": "Model Status",
        "type": "stat",
        "gridPos": {"x": 18, "y": 8, "w": 6, "h": 6},
        "targets": [{
          "expr": "inference_model_loaded",
          "legendFormat": "Model Loaded"
        }],
        "fieldConfig": {
          "defaults": {
            "mappings": [
              {"type": "value", "options": {"0": {"text": "DOWN", "color": "red"}}},
              {"type": "value", "options": {"1": {"text": "UP", "color": "green"}}}
            ]
          }
        }
      }
    ]
  }
}
```

### Create Custom Dashboard

Or build your own:

1. Click **+** → **Dashboard** → **Add visualization**
2. Select **Prometheus** as data source
3. Add queries:

**Request Rate:**
```promql
rate(inference_requests_total[1m])
```

**P99 Latency:**
```promql
histogram_quantile(0.99, rate(inference_request_duration_seconds_bucket[5m]))
```

**Error Rate:**
```promql
sum(rate(inference_requests_total{status=~"5.."}[5m])) /
sum(rate(inference_requests_total[5m])) * 100
```

## Step 5: Configure Alerts

### Create Alert Rules

```bash
mkdir -p monitoring/rules

cat > monitoring/rules/inference-alerts.yml << 'EOF'
groups:
  - name: inference-alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          sum(rate(inference_requests_total{status=~"5.."}[5m])) /
          sum(rate(inference_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99, rate(inference_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P99 latency"
          description: "P99 latency is {{ $value | humanizeDuration }} (threshold: 5s)"

      # No requests (service might be down)
      - alert: NoRequests
        expr: |
          sum(rate(inference_requests_total[5m])) == 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "No requests received"
          description: "No inference requests in the last 10 minutes"

      # Model not loaded
      - alert: ModelNotLoaded
        expr: inference_model_loaded == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Model not loaded"
          description: "Inference model is not loaded on one or more replicas"

      # Low capacity (all replicas busy)
      - alert: HighUtilization
        expr: |
          sum(inference_active_requests) /
          count(inference_model_loaded) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High replica utilization"
          description: "{{ $value | humanizePercentage }} of capacity in use"
EOF
```

### Update Prometheus Config

Add rules to Prometheus:

```yaml
# Add to monitoring/prometheus.yml
rule_files:
  - /etc/prometheus/rules/*.yml
```

Mount rules directory:

```yaml
# Update docker-compose.monitoring.yml
prometheus:
  volumes:
    - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    - ./monitoring/rules:/etc/prometheus/rules  # Add this
    - prometheus_data:/prometheus
```

### Restart Prometheus

```bash
docker compose -f docker-compose.monitoring.yml restart prometheus

# Verify rules loaded
curl http://localhost:9090/api/v1/rules
```

## Step 6: Monitor SkyServe Status

### Create SkyServe Exporter

Script to export SkyServe metrics:

```bash
cat > scripts/skyserve-exporter.py << 'EOF'
#!/usr/bin/env python3
"""Exports SkyServe metrics to Prometheus format."""

import subprocess
import json
import re
from flask import Flask
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metrics
REPLICA_COUNT = Gauge('skyserve_replicas_total', 'Total replicas', ['service', 'status'])
REPLICA_CLOUD = Gauge('skyserve_replicas_by_cloud', 'Replicas by cloud', ['service', 'cloud'])
SERVICE_STATUS = Gauge('skyserve_service_status', 'Service status (1=ready)', ['service'])

def get_skyserve_status(service_name):
    """Get SkyServe status via CLI."""
    try:
        result = subprocess.run(
            ['sky', 'serve', 'status', service_name, '--all'],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout
    except Exception as e:
        print(f"Error getting status: {e}")
        return ""

def parse_status(output, service_name):
    """Parse SkyServe status output and update metrics."""
    # Count replicas by status
    status_counts = {'READY': 0, 'STARTING': 0, 'FAILED': 0}
    cloud_counts = {'ssh': 0, 'aws': 0, 'gcp': 0}

    for line in output.split('\n'):
        if 'READY' in line:
            status_counts['READY'] += 1
        elif 'STARTING' in line:
            status_counts['STARTING'] += 1
        elif 'FAILED' in line:
            status_counts['FAILED'] += 1

        if 'ssh' in line.lower():
            cloud_counts['ssh'] += 1
        elif 'aws' in line.lower():
            cloud_counts['aws'] += 1
        elif 'gcp' in line.lower():
            cloud_counts['gcp'] += 1

    # Update metrics
    for status, count in status_counts.items():
        REPLICA_COUNT.labels(service=service_name, status=status).set(count)

    for cloud, count in cloud_counts.items():
        REPLICA_CLOUD.labels(service=service_name, cloud=cloud).set(count)

    # Service is ready if at least one replica is ready
    SERVICE_STATUS.labels(service=service_name).set(1 if status_counts['READY'] > 0 else 0)

@app.route('/metrics')
def metrics():
    # Update metrics before serving
    for service in ['production']:  # Add your service names
        output = get_skyserve_status(service)
        parse_status(output, service)

    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9091)
EOF
chmod +x scripts/skyserve-exporter.py
```

### Add to Prometheus

```yaml
# Add to monitoring/prometheus.yml
scrape_configs:
  - job_name: 'skyserve-exporter'
    static_configs:
      - targets: ['localhost:9091']
```

### Run Exporter

```bash
# Run in background
python3 scripts/skyserve-exporter.py &

# Or add to docker-compose
```

## Step 7: Set Up Notifications

### Slack Notifications

1. Create Slack webhook: https://api.slack.com/messaging/webhooks
2. Update alertmanager.yml:

```yaml
receivers:
  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/xxx/yyy/zzz'
        channel: '#inference-alerts'
        send_resolved: true
        title: '{{ .Status | toUpper }}: {{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'
```

### PagerDuty Notifications

```yaml
receivers:
  - name: 'critical-alerts'
    pagerduty_configs:
      - service_key: 'your-pagerduty-service-key'
        severity: critical
```

### Email Notifications

```yaml
receivers:
  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@yourcompany.com'
        from: 'alerts@yourcompany.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'alerts@yourcompany.com'
        auth_password: 'app-password'
```

## Step 8: Cost Monitoring

### Track Cloud Spend

Add cost estimates to your dashboard:

```promql
# Estimated hourly cost (example rates)
sum(skyserve_replicas_by_cloud{cloud="aws"}) * 0.15 +
sum(skyserve_replicas_by_cloud{cloud="gcp"}) * 0.12 +
sum(skyserve_replicas_by_cloud{cloud="ssh"}) * 0  # Hetzner is fixed cost
```

### Set Cost Alerts

```yaml
# Add to inference-alerts.yml
- alert: HighCloudSpend
  expr: |
    sum(skyserve_replicas_by_cloud{cloud=~"aws|gcp"}) > 10
  for: 30m
  labels:
    severity: warning
  annotations:
    summary: "High cloud replica count"
    description: "{{ $value }} cloud replicas running for 30+ minutes"
```

## Summary

You've learned how to:

1. Add Prometheus metrics to your inference server
2. Deploy Prometheus + Grafana monitoring stack
3. Create dashboards for key metrics
4. Configure alerting rules
5. Monitor SkyServe status
6. Set up notifications

**Monitoring Stack:**
```
Your Service → Prometheus → Grafana (visualize)
                    ↓
              Alertmanager → Slack/Email/PagerDuty
```

**Key URLs:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Alertmanager: http://localhost:9093

**Key Commands:**
```bash
# Start monitoring
docker compose -f docker-compose.monitoring.yml up -d

# Check alerts
curl http://localhost:9090/api/v1/alerts

# Test alert
curl -XPOST http://localhost:9093/api/v1/alerts -d '[
  {"labels":{"alertname":"Test","severity":"critical"}}
]'
```

## What's Next?

- **Add more metrics** - GPU usage, token throughput, cache hit rates
- **Set up log aggregation** - Loki + Grafana for centralized logs
- **Create SLOs** - Define and track service level objectives
- **Set up on-call rotation** - PagerDuty or Opsgenie integration
