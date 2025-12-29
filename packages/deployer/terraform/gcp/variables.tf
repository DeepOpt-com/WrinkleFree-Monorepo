# =============================================================================
# Authentication
# =============================================================================
variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
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
# SSH Configuration
# =============================================================================
variable "ssh_public_key" {
  description = "SSH public key content"
  type        = string
}

variable "ssh_user" {
  description = "SSH username"
  type        = string
  default     = "wrinklefree"
}

# =============================================================================
# Networking
# =============================================================================
variable "network_cidr" {
  description = "CIDR block for the subnet"
  type        = string
  default     = "10.200.0.0/24"
}

variable "inference_port" {
  description = "Port for inference API"
  type        = number
  default     = 8080
}

# =============================================================================
# Cloudflare IPs
# =============================================================================
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
# GCS Checkpoint Storage
# =============================================================================
variable "checkpoint_bucket_name" {
  description = "GCS bucket name for training checkpoints (must be globally unique)"
  type        = string
  default     = ""
}

variable "model_bucket_name" {
  description = "GCS bucket name for trained models (optional)"
  type        = string
  default     = ""
}

variable "enable_versioning" {
  description = "Enable versioning on checkpoint bucket"
  type        = bool
  default     = true
}

variable "lifecycle_nearline_days" {
  description = "Days before moving old checkpoints to Nearline storage (0 = disabled)"
  type        = number
  default     = 30
}

variable "lifecycle_expire_days" {
  description = "Days before expiring old checkpoints (0 = disabled)"
  type        = number
  default     = 90
}

# =============================================================================
# Build Cache Storage (for BitNet.cpp artifacts)
# =============================================================================
variable "build_cache_bucket_name" {
  description = "GCS bucket name for BitNet.cpp build cache (must be globally unique)"
  type        = string
  default     = ""
}

variable "build_cache_expire_days" {
  description = "Days before expiring cached build artifacts (0 = disabled)"
  type        = number
  default     = 30
}

# =============================================================================
# Labels
# =============================================================================
variable "labels" {
  description = "Additional labels for resources"
  type        = map(string)
  default     = {}
}
