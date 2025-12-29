# =============================================================================
# Network
# =============================================================================
output "network_id" {
  description = "GCP VPC network ID"
  value       = google_compute_network.inference.id
}

output "network_name" {
  description = "GCP VPC network name"
  value       = google_compute_network.inference.name
}

output "subnet_id" {
  description = "GCP subnet ID"
  value       = google_compute_subnetwork.inference.id
}

output "subnet_cidr" {
  description = "GCP subnet CIDR"
  value       = google_compute_subnetwork.inference.ip_cidr_range
}

# =============================================================================
# SkyPilot Integration
# =============================================================================
output "skypilot_config_hint" {
  description = "Hint for SkyPilot GCP configuration"
  value       = "GCP spot instances are automatically managed by SkyPilot. Ensure gcloud is configured with: gcloud auth application-default login"
}

# =============================================================================
# GCS Buckets
# =============================================================================
output "checkpoint_bucket_name" {
  description = "Name of the GCS bucket for training checkpoints"
  value       = google_storage_bucket.checkpoints.name
}

output "checkpoint_bucket_url" {
  description = "URL of the checkpoint bucket"
  value       = google_storage_bucket.checkpoints.url
}

output "model_bucket_name" {
  description = "Name of the GCS bucket for trained models"
  value       = google_storage_bucket.models.name
}

output "model_bucket_url" {
  description = "URL of the models bucket"
  value       = google_storage_bucket.models.url
}

output "build_cache_bucket_name" {
  description = "Name of the GCS bucket for BitNet.cpp build cache"
  value       = google_storage_bucket.build_cache.name
}

output "build_cache_bucket_url" {
  description = "URL of the build cache bucket (gs://...)"
  value       = google_storage_bucket.build_cache.url
}

# =============================================================================
# Service Account
# =============================================================================
output "skypilot_service_account_email" {
  description = "Email of the SkyPilot service account"
  value       = google_service_account.skypilot.email
}

# =============================================================================
# SkyPilot Environment Variables
# =============================================================================
output "skypilot_env_vars" {
  description = "Environment variables to set for SkyPilot training jobs"
  value = {
    CHECKPOINT_BUCKET  = google_storage_bucket.checkpoints.name
    CHECKPOINT_STORE   = "gcs"
    MODEL_BUCKET       = google_storage_bucket.models.name
    GCS_CACHE_BUCKET   = google_storage_bucket.build_cache.url
    BUILD_CACHE_BUCKET = google_storage_bucket.build_cache.name
    GCP_PROJECT        = var.gcp_project_id
    GCP_REGION         = var.gcp_region
  }
}

output "usage_instructions" {
  description = "Instructions for using the checkpoint storage"
  value       = <<-EOT
    # Set environment variables for SkyPilot:
    export CHECKPOINT_BUCKET=${google_storage_bucket.checkpoints.name}
    export CHECKPOINT_STORE=gcs

    # Launch training job:
    sky jobs launch train.yaml -e MODEL=qwen3_4b -e STAGE=2

    # List checkpoints:
    gsutil ls gs://${google_storage_bucket.checkpoints.name}/

    # Download checkpoint:
    gsutil -m cp -r gs://${google_storage_bucket.checkpoints.name}/qwen3_4b/stage2/ ./checkpoints/
  EOT
}
