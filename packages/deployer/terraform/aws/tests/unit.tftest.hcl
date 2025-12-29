# Terraform native tests for AWS module
# Run with: terraform test

# =============================================================================
# Test Variables
# =============================================================================
variables {
  aws_region             = "us-east-1"
  environment            = "dev"
  project_name           = "wrinklefree-test"
  checkpoint_bucket_name = ""
  enable_versioning      = true
  lifecycle_glacier_days = 30
  lifecycle_expire_days  = 90
}

# =============================================================================
# S3 Bucket Tests
# =============================================================================

run "checkpoint_bucket_created" {
  command = plan

  assert {
    condition     = aws_s3_bucket.checkpoints.bucket == "wrinklefree-test-checkpoints-dev"
    error_message = "Checkpoint bucket should be auto-generated with correct name"
  }
}

run "model_bucket_created" {
  command = plan

  assert {
    condition     = aws_s3_bucket.models.bucket == "wrinklefree-test-models-dev"
    error_message = "Model bucket should be auto-generated with correct name"
  }
}

run "versioning_enabled" {
  command = plan

  assert {
    condition     = aws_s3_bucket_versioning.checkpoints.versioning_configuration[0].status == "Enabled"
    error_message = "Versioning should be enabled on checkpoint bucket"
  }
}

run "encryption_enabled" {
  command = plan

  assert {
    condition     = aws_s3_bucket_server_side_encryption_configuration.checkpoints.rule[0].apply_server_side_encryption_by_default[0].sse_algorithm == "AES256"
    error_message = "Server-side encryption should be enabled"
  }
}

run "public_access_blocked" {
  command = plan

  assert {
    condition     = aws_s3_bucket_public_access_block.checkpoints.block_public_acls == true
    error_message = "Public ACLs should be blocked"
  }

  assert {
    condition     = aws_s3_bucket_public_access_block.checkpoints.block_public_policy == true
    error_message = "Public policy should be blocked"
  }
}

# =============================================================================
# Lifecycle Tests
# =============================================================================

run "lifecycle_rule_created" {
  command = plan

  assert {
    condition     = length(aws_s3_bucket_lifecycle_configuration.checkpoints) == 1
    error_message = "Lifecycle configuration should be created when glacier/expire days > 0"
  }
}

run "no_lifecycle_when_disabled" {
  command = plan

  variables {
    lifecycle_glacier_days = 0
    lifecycle_expire_days  = 0
  }

  assert {
    condition     = length(aws_s3_bucket_lifecycle_configuration.checkpoints) == 0
    error_message = "Lifecycle configuration should not be created when disabled"
  }
}

# =============================================================================
# IAM Tests
# =============================================================================

run "iam_policy_created" {
  command = plan

  assert {
    condition     = aws_iam_policy.skypilot_s3_access.name == "wrinklefree-test-dev-skypilot-s3"
    error_message = "IAM policy should have correct name"
  }
}

run "iam_role_created" {
  command = plan

  assert {
    condition     = aws_iam_role.skypilot_instance.name == "wrinklefree-test-dev-skypilot-instance"
    error_message = "IAM role should have correct name"
  }
}

run "instance_profile_created" {
  command = plan

  assert {
    condition     = aws_iam_instance_profile.skypilot.name == "wrinklefree-test-dev-skypilot"
    error_message = "Instance profile should have correct name"
  }
}

# =============================================================================
# Tag Tests
# =============================================================================

run "tags_applied" {
  command = plan

  assert {
    condition     = aws_s3_bucket.checkpoints.tags["Project"] == "wrinklefree-test"
    error_message = "Project tag should be applied"
  }

  assert {
    condition     = aws_s3_bucket.checkpoints.tags["Environment"] == "dev"
    error_message = "Environment tag should be applied"
  }

  assert {
    condition     = aws_s3_bucket.checkpoints.tags["ManagedBy"] == "terraform"
    error_message = "ManagedBy tag should be applied"
  }
}

# =============================================================================
# Custom Bucket Name Tests
# =============================================================================

run "custom_bucket_name" {
  command = plan

  variables {
    checkpoint_bucket_name = "my-custom-checkpoints"
    model_bucket_name      = "my-custom-models"
  }

  assert {
    condition     = aws_s3_bucket.checkpoints.bucket == "my-custom-checkpoints"
    error_message = "Custom checkpoint bucket name should be used"
  }

  assert {
    condition     = aws_s3_bucket.models.bucket == "my-custom-models"
    error_message = "Custom model bucket name should be used"
  }
}
