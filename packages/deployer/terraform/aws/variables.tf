# =============================================================================
# Authentication
# =============================================================================
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
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
# S3 Checkpoint Storage
# =============================================================================
variable "checkpoint_bucket_name" {
  description = "S3 bucket name for training checkpoints (must be globally unique)"
  type        = string
  default     = ""

  validation {
    condition     = var.checkpoint_bucket_name == "" || can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.checkpoint_bucket_name))
    error_message = "Bucket name must be valid S3 bucket name format."
  }
}

variable "enable_versioning" {
  description = "Enable versioning on checkpoint bucket"
  type        = bool
  default     = true
}

variable "lifecycle_glacier_days" {
  description = "Days before moving old checkpoints to Glacier (0 = disabled)"
  type        = number
  default     = 30
}

variable "lifecycle_expire_days" {
  description = "Days before expiring old checkpoints (0 = disabled)"
  type        = number
  default     = 90
}

# =============================================================================
# Access Control
# =============================================================================
variable "allowed_account_ids" {
  description = "AWS account IDs allowed to access checkpoints"
  type        = list(string)
  default     = []
}

variable "allowed_vpc_endpoints" {
  description = "VPC endpoint IDs allowed to access bucket"
  type        = list(string)
  default     = []
}

# =============================================================================
# Model Storage
# =============================================================================
variable "model_bucket_name" {
  description = "S3 bucket name for trained models (optional, uses checkpoint bucket if empty)"
  type        = string
  default     = ""
}

# =============================================================================
# Tags
# =============================================================================
variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}
