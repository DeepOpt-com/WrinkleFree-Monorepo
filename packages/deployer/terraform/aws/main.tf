locals {
  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    },
    var.tags
  )

  # Generate bucket name if not provided
  checkpoint_bucket = var.checkpoint_bucket_name != "" ? var.checkpoint_bucket_name : "${var.project_name}-checkpoints-${var.environment}"
  model_bucket      = var.model_bucket_name != "" ? var.model_bucket_name : "${var.project_name}-models-${var.environment}"
}

# =============================================================================
# S3 Bucket for Training Checkpoints
# =============================================================================
resource "aws_s3_bucket" "checkpoints" {
  bucket = local.checkpoint_bucket

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-checkpoints"
    Type = "checkpoint-storage"
  })
}

resource "aws_s3_bucket_versioning" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "checkpoints" {
  count  = var.lifecycle_glacier_days > 0 || var.lifecycle_expire_days > 0 ? 1 : 0
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    id     = "checkpoint-lifecycle"
    status = "Enabled"

    filter {
      prefix = ""
    }

    # Move to Glacier after specified days
    dynamic "transition" {
      for_each = var.lifecycle_glacier_days > 0 ? [1] : []
      content {
        days          = var.lifecycle_glacier_days
        storage_class = "GLACIER"
      }
    }

    # Expire after specified days
    dynamic "expiration" {
      for_each = var.lifecycle_expire_days > 0 ? [1] : []
      content {
        days = var.lifecycle_expire_days
      }
    }

    # Clean up old versions
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

# =============================================================================
# S3 Bucket for Trained Models
# =============================================================================
resource "aws_s3_bucket" "models" {
  bucket = local.model_bucket

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-models"
    Type = "model-storage"
  })
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# =============================================================================
# IAM Policy for SkyPilot Access
# =============================================================================
resource "aws_iam_policy" "skypilot_s3_access" {
  name        = "${var.project_name}-${var.environment}-skypilot-s3"
  description = "Allow SkyPilot to access checkpoint and model buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListBuckets"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.checkpoints.arn,
          aws_s3_bucket.models.arn
        ]
      },
      {
        Sid    = "ReadWriteCheckpoints"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:GetObjectVersion"
        ]
        Resource = [
          "${aws_s3_bucket.checkpoints.arn}/*"
        ]
      },
      {
        Sid    = "ReadModels"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion"
        ]
        Resource = [
          "${aws_s3_bucket.models.arn}/*"
        ]
      },
      {
        Sid    = "WriteModels"
        Effect = "Allow"
        Action = [
          "s3:PutObject"
        ]
        Resource = [
          "${aws_s3_bucket.models.arn}/*"
        ]
      }
    ]
  })

  tags = local.common_tags
}

# =============================================================================
# IAM Role for EC2 Instances (SkyPilot)
# =============================================================================
resource "aws_iam_role" "skypilot_instance" {
  name = "${var.project_name}-${var.environment}-skypilot-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "skypilot_s3" {
  role       = aws_iam_role.skypilot_instance.name
  policy_arn = aws_iam_policy.skypilot_s3_access.arn
}

resource "aws_iam_instance_profile" "skypilot" {
  name = "${var.project_name}-${var.environment}-skypilot"
  role = aws_iam_role.skypilot_instance.name

  tags = local.common_tags
}
