# =============================================================================
# Authentication
# =============================================================================
variable "hetzner_cloud_token" {
  description = "Hetzner Cloud API token"
  type        = string
  sensitive   = true
}

# =============================================================================
# Environment
# =============================================================================
variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "wrinklefree"
}

# =============================================================================
# Server Configuration
# =============================================================================
variable "node_count" {
  description = "Number of Hetzner Cloud inference nodes (0 for dedicated-only setup)"
  type        = number
  default     = 0

  validation {
    condition     = var.node_count >= 0 && var.node_count <= 20
    error_message = "Node count must be between 0 and 20."
  }
}

variable "server_type" {
  description = "Hetzner Cloud server type"
  type        = string
  default     = "cx52" # 16 vCPU, 32GB RAM
}

variable "location" {
  description = "Hetzner datacenter location"
  type        = string
  default     = "fsn1"

  validation {
    condition     = contains(["fsn1", "nbg1", "hel1"], var.location)
    error_message = "Location must be fsn1, nbg1, or hel1."
  }
}

variable "image" {
  description = "Server OS image"
  type        = string
  default     = "ubuntu-22.04"
}

# =============================================================================
# SSH Configuration
# =============================================================================
variable "ssh_public_key_path" {
  description = "Path to SSH public key file"
  type        = string
  default     = "~/.ssh/hetzner_ed25519.pub"
}

variable "ssh_user" {
  description = "SSH user for server access"
  type        = string
  default     = "root"
}

# =============================================================================
# Networking
# =============================================================================
variable "network_cidr" {
  description = "CIDR block for private network"
  type        = string
  default     = "10.0.0.0/16"
}

variable "subnet_cidr" {
  description = "CIDR block for inference subnet"
  type        = string
  default     = "10.0.1.0/24"
}

# =============================================================================
# Firewall
# =============================================================================
variable "admin_ip" {
  description = "Admin IP for SSH access (CIDR notation)"
  type        = string
  default     = "0.0.0.0/0" # Should be restricted in production
}

variable "inference_port" {
  description = "Port for inference API"
  type        = number
  default     = 8080
}

variable "cloudflare_ips" {
  description = "Cloudflare IP ranges for origin access"
  type        = list(string)
  default = [
    "173.245.48.0/20",
    "103.21.244.0/22",
    "103.22.200.0/22",
    "103.31.4.0/22",
    "141.101.64.0/18",
    "108.162.192.0/18",
    "190.93.240.0/20",
    "188.114.96.0/20",
    "197.234.240.0/22",
    "198.41.128.0/17",
    "162.158.0.0/15",
    "104.16.0.0/13",
    "104.24.0.0/14",
    "172.64.0.0/13",
    "131.0.72.0/22"
  ]
}

# =============================================================================
# Labels/Tags
# =============================================================================
variable "labels" {
  description = "Additional labels to apply to resources"
  type        = map(string)
  default     = {}
}
