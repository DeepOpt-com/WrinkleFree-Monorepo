# =============================================================================
# GCS Bucket for Training Checkpoints
# =============================================================================

locals {
  checkpoint_bucket  = var.checkpoint_bucket_name != "" ? var.checkpoint_bucket_name : "${var.project_name}-checkpoints-${var.environment}"
  model_bucket       = var.model_bucket_name != "" ? var.model_bucket_name : "${var.project_name}-models-${var.environment}"
  build_cache_bucket = var.build_cache_bucket_name != "" ? var.build_cache_bucket_name : "${var.project_name}-build-cache-${var.environment}"
}

resource "google_storage_bucket" "checkpoints" {
  name          = local.checkpoint_bucket
  location      = var.gcp_region
  force_destroy = var.environment != "prod" # Only allow force destroy in non-prod

  # Uniform bucket-level access (recommended)
  uniform_bucket_level_access = true

  # Versioning for checkpoint recovery
  versioning {
    enabled = var.enable_versioning
  }

  # Lifecycle rules for cost optimization
  dynamic "lifecycle_rule" {
    for_each = var.lifecycle_nearline_days > 0 ? [1] : []
    content {
      action {
        type          = "SetStorageClass"
        storage_class = "NEARLINE"
      }
      condition {
        age = var.lifecycle_nearline_days
      }
    }
  }

  dynamic "lifecycle_rule" {
    for_each = var.lifecycle_expire_days > 0 ? [1] : []
    content {
      action {
        type = "Delete"
      }
      condition {
        age = var.lifecycle_expire_days
      }
    }
  }

  # Delete old versions after 7 days
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      num_newer_versions = 3
      with_state         = "ARCHIVED"
    }
  }

  labels = merge(local.common_labels, {
    type = "checkpoint-storage"
  })
}

# =============================================================================
# GCS Bucket for Trained Models
# =============================================================================

resource "google_storage_bucket" "models" {
  name          = local.model_bucket
  location      = var.gcp_region
  force_destroy = false # Never force destroy models

  uniform_bucket_level_access = true

  versioning {
    enabled = true # Always version models
  }

  labels = merge(local.common_labels, {
    type = "model-storage"
  })
}

# =============================================================================
# GCS Bucket for BitNet.cpp Build Cache
# =============================================================================

resource "google_storage_bucket" "build_cache" {
  name          = local.build_cache_bucket
  location      = var.gcp_region
  force_destroy = true # Build cache is ephemeral

  uniform_bucket_level_access = true

  # No versioning needed for build cache
  versioning {
    enabled = false
  }

  # Lifecycle rules - auto-expire old builds
  dynamic "lifecycle_rule" {
    for_each = var.build_cache_expire_days > 0 ? [1] : []
    content {
      action {
        type = "Delete"
      }
      condition {
        age = var.build_cache_expire_days
      }
    }
  }

  labels = merge(local.common_labels, {
    type    = "build-cache"
    purpose = "bitnet-cpp-artifacts"
  })
}

# =============================================================================
# IAM for SkyPilot Access
# =============================================================================

# Service account for SkyPilot instances
resource "google_service_account" "skypilot" {
  account_id   = "${var.project_name}-${var.environment}-skypilot"
  display_name = "SkyPilot Service Account for ${var.project_name}"
}

# Grant access to checkpoint bucket
resource "google_storage_bucket_iam_member" "checkpoints_admin" {
  bucket = google_storage_bucket.checkpoints.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.skypilot.email}"
}

# Grant access to models bucket (read + write)
resource "google_storage_bucket_iam_member" "models_viewer" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.skypilot.email}"
}

resource "google_storage_bucket_iam_member" "models_creator" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.skypilot.email}"
}

# Grant access to build cache bucket (read + write for build artifacts)
resource "google_storage_bucket_iam_member" "build_cache_admin" {
  bucket = google_storage_bucket.build_cache.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.skypilot.email}"
}

# Allow SkyPilot to use this service account
resource "google_service_account_iam_member" "skypilot_user" {
  service_account_id = google_service_account.skypilot.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.skypilot.email}"
}
