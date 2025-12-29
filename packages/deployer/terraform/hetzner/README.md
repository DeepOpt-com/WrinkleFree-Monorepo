# Terraform: Hetzner Dedicated Servers

This directory contains Terraform configuration for provisioning Hetzner bare metal servers.

## Prerequisites

- [Terraform](https://terraform.io/) >= 1.5
- Hetzner Robot API credentials
- SSH key pair for server access

## Planned Structure

```
terraform/hetzner/
├── README.md           # This file
├── main.tf             # Main configuration
├── variables.tf        # Input variables
├── outputs.tf          # Output values (IPs for SkyPilot)
├── providers.tf        # Provider configuration
└── terraform.tfvars.example  # Example variables
```

## Configuration (TODO)

### Provider Setup

```hcl
# providers.tf
terraform {
  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "~> 1.45"
    }
  }
}

provider "hcloud" {
  token = var.hetzner_token
}
```

### Server Configuration

```hcl
# main.tf
resource "hcloud_server" "inference" {
  count       = var.node_count
  name        = "wrinklefree-inference-${count.index}"
  server_type = var.server_type  # ax102, ax162, etc.
  image       = "ubuntu-22.04"
  location    = var.location     # fsn1, nbg1, hel1

  ssh_keys = [hcloud_ssh_key.deploy.id]

  labels = {
    role    = "inference"
    project = "wrinklefree"
  }
}

resource "hcloud_ssh_key" "deploy" {
  name       = "wrinklefree-deploy"
  public_key = file(var.ssh_public_key_path)
}
```

### Variables

```hcl
# variables.tf
variable "hetzner_token" {
  description = "Hetzner Cloud API token"
  type        = string
  sensitive   = true
}

variable "node_count" {
  description = "Number of inference nodes"
  type        = number
  default     = 3
}

variable "server_type" {
  description = "Hetzner server type"
  type        = string
  default     = "ax102"  # AMD EPYC, 256GB RAM
}

variable "location" {
  description = "Hetzner datacenter"
  type        = string
  default     = "fsn1"  # Falkenstein
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key"
  type        = string
  default     = "~/.ssh/hetzner_ed25519.pub"
}
```

### Outputs

```hcl
# outputs.tf
output "server_ips" {
  description = "IP addresses for SkyPilot ssh_node_pools.yaml"
  value       = hcloud_server.inference[*].ipv4_address
}

output "ssh_node_pools_config" {
  description = "Ready-to-use SkyPilot config snippet"
  value = yamlencode({
    "hetzner-base" = {
      user          = "root"
      identity_file = "~/.ssh/hetzner_ed25519"
      hosts         = [for s in hcloud_server.inference : { ip = s.ipv4_address }]
    }
  })
}
```

## Usage (TODO)

```bash
# Initialize
cd terraform/hetzner
terraform init

# Configure
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

# Plan
terraform plan

# Apply
terraform apply

# Get IPs for SkyPilot
terraform output -raw ssh_node_pools_config >> ~/.sky/ssh_node_pools.yaml
```

## Hetzner Dedicated Servers (Robot API)

For dedicated servers (not cloud), use the Robot API instead:

```bash
# Robot API requires different authentication
# See: https://robot.hetzner.com/doc/webservice/en.html
```

## Recommended Server Types

| Type | CPU | RAM | Use Case |
|------|-----|-----|----------|
| AX102 | AMD EPYC 9454P (48C) | 256GB DDR5 | Production |
| AX162 | AMD EPYC 9654 (96C) | 384GB DDR5 | High-throughput |
| AX42 | AMD Ryzen 9 5950X | 128GB DDR4 | Development |

## Network Setup

For optimal performance, configure vSwitch:

```hcl
resource "hcloud_network" "inference" {
  name     = "wrinklefree-inference"
  ip_range = "10.0.0.0/16"
}

resource "hcloud_network_subnet" "inference" {
  network_id   = hcloud_network.inference.id
  type         = "cloud"
  network_zone = "eu-central"
  ip_range     = "10.0.1.0/24"
}
```

## Security Notes

- Store `terraform.tfvars` securely (add to `.gitignore`)
- Use Terraform Cloud or encrypted state for production
- Restrict firewall to Cloudflare IPs only
