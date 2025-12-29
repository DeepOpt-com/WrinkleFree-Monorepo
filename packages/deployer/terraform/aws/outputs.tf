# =============================================================================
# S3 Buckets
# =============================================================================
output "checkpoint_bucket_name" {
  description = "Name of the S3 bucket for training checkpoints"
  value       = aws_s3_bucket.checkpoints.id
}

output "checkpoint_bucket_arn" {
  description = "ARN of the checkpoint bucket"
  value       = aws_s3_bucket.checkpoints.arn
}

output "model_bucket_name" {
  description = "Name of the S3 bucket for trained models"
  value       = aws_s3_bucket.models.id
}

output "model_bucket_arn" {
  description = "ARN of the models bucket"
  value       = aws_s3_bucket.models.arn
}

# =============================================================================
# SkyPilot Integration
# =============================================================================
output "skypilot_env_vars" {
  description = "Environment variables to set for SkyPilot training jobs"
  value = {
    CHECKPOINT_BUCKET = aws_s3_bucket.checkpoints.id
    CHECKPOINT_STORE  = "s3"
    MODEL_BUCKET      = aws_s3_bucket.models.id
    AWS_REGION        = var.aws_region
  }
}

output "skypilot_instance_profile" {
  description = "Instance profile ARN for SkyPilot EC2 instances"
  value       = aws_iam_instance_profile.skypilot.arn
}

output "skypilot_instance_profile_name" {
  description = "Instance profile name for SkyPilot EC2 instances"
  value       = aws_iam_instance_profile.skypilot.name
}

# =============================================================================
# IAM
# =============================================================================
output "skypilot_role_arn" {
  description = "IAM role ARN for SkyPilot instances"
  value       = aws_iam_role.skypilot_instance.arn
}

output "s3_policy_arn" {
  description = "IAM policy ARN for S3 access"
  value       = aws_iam_policy.skypilot_s3_access.arn
}

# =============================================================================
# Usage Instructions
# =============================================================================
output "usage_instructions" {
  description = "Instructions for using the checkpoint storage"
  value       = <<-EOT
    # Set environment variables for SkyPilot:
    export CHECKPOINT_BUCKET=${aws_s3_bucket.checkpoints.id}
    export CHECKPOINT_STORE=s3

    # Launch training job:
    sky jobs launch train.yaml -e MODEL=qwen3_4b -e STAGE=2

    # List checkpoints:
    aws s3 ls s3://${aws_s3_bucket.checkpoints.id}/

    # Download checkpoint:
    aws s3 sync s3://${aws_s3_bucket.checkpoints.id}/qwen3_4b/stage2/ ./checkpoints/
  EOT
}
