# Terraform Infrastructure

Infrastructure as Code for WrinkleFree hybrid cloud deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WrinkleFree Infrastructure                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │    Hetzner       │  │      GCP         │  │      AWS         │   │
│  │  (Base Layer)    │  │   (Storage)      │  │   (Storage)      │   │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤   │
│  │ • Dedicated      │  │ • GCS Storage    │  │ • S3 Checkpoints │   │
│  │   Servers        │  │ • VPC Network    │  │ • S3 Models      │   │
│  │ • Cloud VMs      │  │ • Firewall       │  │ • IAM Roles      │   │
│  │ • vSwitch Net    │  │ • IAM            │  │                  │   │
│  │ • Firewall       │  │                  │  │                  │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    SkyPilot Orchestration                      │  │
│  │  • SSH Node Pools (Hetzner) • Spot VM Management (all clouds)  │  │
│  │  • Checkpoint Storage (S3/GCS) • Auto-failover                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Modules

### Hetzner (`terraform/hetzner/`)

Base infrastructure layer with fixed monthly cost.

| Resource | Description |
|----------|-------------|
| `hcloud_ssh_key` | SSH key for all servers |
| `hcloud_network` | Private vSwitch network |
| `hcloud_network_subnet` | Inference subnet |
| `hcloud_firewall` | SSH + inference port rules |
| `hcloud_server` | Optional cloud VMs |

**Outputs for SkyPilot:**
```bash
terraform output -raw ssh_node_pools_config >> ~/.sky/ssh_node_pools.yaml
```

### GCP (`terraform/gcp/`)

Storage and networking for SkyPilot-managed GCP workloads.

| Resource | Description |
|----------|-------------|
| `google_compute_network` | VPC network |
| `google_compute_firewall` | SSH, inference, health rules |
| `google_storage_bucket` | Checkpoint + model storage |
| `google_service_account` | SkyPilot IAM |

**Note:** Spot VMs are provisioned directly by SkyPilot, not Terraform.

### AWS (`terraform/aws/`)

Checkpoint and model storage for training.

| Resource | Description |
|----------|-------------|
| `aws_s3_bucket` | Checkpoint storage (versioned) |
| `aws_s3_bucket` | Model storage |
| `aws_s3_bucket_lifecycle_configuration` | Glacier archival |
| `aws_iam_role` | SkyPilot instance role |
| `aws_iam_instance_profile` | EC2 instance profile |

## Quick Start

### 1. Configure Credentials

```bash
# Hetzner
export HCLOUD_TOKEN="your-hetzner-cloud-token"

# GCP
gcloud auth application-default login
export GOOGLE_PROJECT="your-gcp-project"

# AWS
aws configure
```

### 2. Deploy Infrastructure

```bash
# Hetzner (base layer)
cd terraform/hetzner
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars
terraform init && terraform apply

# GCP (burst layer + storage)
cd ../gcp
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars
terraform init && terraform apply

# AWS (checkpoint storage)
cd ../aws
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars
terraform init && terraform apply
```

### 3. Configure SkyPilot

```bash
# Get Hetzner SSH node pool config
terraform -chdir=terraform/hetzner output -raw ssh_node_pools_config >> ~/.sky/ssh_node_pools.yaml

# Get environment variables
terraform -chdir=terraform/aws output skypilot_env_vars
terraform -chdir=terraform/gcp output skypilot_env_vars

# Verify SkyPilot
sky check
```

### 4. Launch Training

```bash
# Set checkpoint storage
export CHECKPOINT_BUCKET=$(terraform -chdir=terraform/aws output -raw checkpoint_bucket_name)
export CHECKPOINT_STORE=s3

# Launch training job
sky jobs launch skypilot/train.yaml -e MODEL=qwen3_4b -e STAGE=2
```

## Testing

### Static Analysis (No Deploy)

```bash
# TFLint
tflint --init
tflint hetzner/ gcp/ aws/

# Checkov security scan
checkov -d . --config-file ../.checkov.yaml

# OPA policy check
terraform plan -out=tfplan.binary
terraform show -json tfplan.binary > tfplan.json
opa eval --data policies/ --input tfplan.json "data.terraform.security.deny"
```

### Terraform Native Tests

```bash
# Run all tests (Terraform 1.6+)
terraform -chdir=hetzner test
terraform -chdir=gcp test
terraform -chdir=aws test

# Verbose output
terraform -chdir=gcp test -verbose
```

### LocalStack Integration

```bash
# Start emulators
docker compose -f ../docker-compose.localstack.yml up -d

# Run integration tests
uv run pytest ../tests/integration/ -v -m localstack

# Cleanup
docker compose -f ../docker-compose.localstack.yml down -v
```

## Environments

Use the `environments/` directory for multi-environment deployments:

```
terraform/environments/
├── dev/
│   ├── main.tf          # Uses modules with dev settings
│   ├── terraform.tfvars
│   └── backend.tf       # Local state
├── staging/
│   └── ...
└── prod/
    ├── main.tf
    ├── terraform.tfvars
    └── backend.tf       # S3 state with locking
```

## Security

### Best Practices Implemented

- **No hardcoded secrets**: All sensitive values via variables
- **S3 encryption**: AES-256 server-side encryption
- **Public access blocked**: All S3 buckets block public access
- **Firewall rules**: SSH restricted, inference via Cloudflare only
- **IAM least privilege**: SkyPilot role has minimal permissions
- **Versioning**: Checkpoints and models versioned for recovery

### Checkov Checks

Key security checks enforced:
- `CKV_AWS_24`: No SSH from 0.0.0.0/0 in prod
- `CKV_GCP_2`: Firewall ingress restrictions
- `CKV_DOCKER_2`: Docker user not root

## Cost Optimization

### Hetzner (Fixed Monthly)

| Server | Specs | Price |
|--------|-------|-------|
| AX42 | Ryzen 9, 128GB | €54/mo |
| AX102 | EPYC 48C, 256GB | €119/mo |
| AX162 | EPYC 96C, 384GB | €179/mo |

### GCP (Pay-per-use Spot via SkyPilot)

| Instance | Specs | Spot Price |
|----------|-------|------------|
| n2-highmem-16 | 16 vCPU, 128GB | ~$0.15/hr |
| n2-highmem-32 | 32 vCPU, 256GB | ~$0.30/hr |

**Note:** Spot VMs are managed by SkyPilot, not Terraform.

### AWS (Storage)

| Storage Class | Price |
|--------------|-------|
| S3 Standard | $0.023/GB/mo |
| S3 Glacier | $0.004/GB/mo |

**Lifecycle**: Checkpoints move to Glacier after 30 days.

## File Structure

```
terraform/
├── README.md                    # This file
├── .tflint.hcl                  # TFLint configuration
│
├── hetzner/
│   ├── providers.tf
│   ├── variables.tf
│   ├── main.tf
│   ├── outputs.tf
│   ├── versions.tf
│   ├── terraform.tfvars.example
│   └── tests/
│       ├── unit.tftest.hcl
│       └── setup/main.tf
│
├── gcp/
│   ├── providers.tf
│   ├── variables.tf
│   ├── networking.tf
│   ├── storage.tf
│   ├── outputs.tf
│   ├── versions.tf
│   ├── terraform.tfvars.example
│   └── tests/unit.tftest.hcl
│
├── aws/
│   ├── providers.tf
│   ├── variables.tf
│   ├── main.tf
│   ├── outputs.tf
│   ├── versions.tf
│   ├── terraform.tfvars.example
│   └── tests/unit.tftest.hcl
│
└── policies/
    ├── security.rego
    └── cost.rego
```
