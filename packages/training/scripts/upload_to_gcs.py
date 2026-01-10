#!/usr/bin/env python3
"""Upload checkpoints to GCS bucket.

FAILS LOUDLY on any error - checkpoints are critical.

Environment variables:
    CHECKPOINT_DIR: Local directory to upload (default: /tmp/checkpoints)
    GCS_BUCKET: GCS bucket name (default: wrinklefree-checkpoints)
    GCS_PREFIX: GCS path prefix (default: checkpoints/smoke-test)
    GOOGLE_CLOUD_PROJECT: GCP project ID (optional)
"""
import os
import sys
from pathlib import Path

from wf_train.utils.gcs_client import (
    GCSError,
    GCSUploadError,
    get_gcs_client,
    upload_blob,
)


def upload_directory_to_gcs(
    local_dir: str,
    bucket_name: str,
    gcs_prefix: str,
    project: str | None = None,
) -> None:
    """Upload all files from local_dir to GCS bucket under gcs_prefix.

    FAILS LOUDLY on any error - checkpoints are critical.

    Args:
        local_dir: Local directory to upload
        bucket_name: GCS bucket name
        gcs_prefix: GCS path prefix
        project: GCP project ID (optional)

    Raises:
        FileNotFoundError: If local directory doesn't exist
        GCSError: If any upload fails
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    # Validate GCS access upfront (fail fast)
    print(f"Validating GCS access to gs://{bucket_name}/...")
    _, bucket = get_gcs_client(project=project, bucket_name=bucket_name, validate=True)
    print("GCS access validated")

    file_count = 0
    errors = []

    for root, dirs, files in os.walk(local_dir):
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(local_dir)
            gcs_path = f"{gcs_prefix}/{rel_path}"

            try:
                print(f"Uploading {file_path} -> gs://{bucket_name}/{gcs_path}")
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(str(file_path), timeout=600)
                file_count += 1
            except Exception as e:
                error_msg = f"Failed to upload {file_path}: {e}"
                print(f"ERROR: {error_msg}", file=sys.stderr)
                errors.append(error_msg)

    if errors:
        raise GCSUploadError(
            f"Failed to upload {len(errors)} files:\n" + "\n".join(errors)
        )

    print(f"\nUploaded {file_count} files to gs://{bucket_name}/{gcs_prefix}")
    print(
        f"View: https://console.cloud.google.com/storage/browser/{bucket_name}/{gcs_prefix}"
    )


if __name__ == "__main__":
    local_dir = os.environ.get("CHECKPOINT_DIR", "/tmp/checkpoints")
    bucket_name = os.environ.get("GCS_BUCKET", "wrinklefree-checkpoints")
    gcs_prefix = os.environ.get("GCS_PREFIX", "checkpoints/smoke-test")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")

    try:
        upload_directory_to_gcs(local_dir, bucket_name, gcs_prefix, project)
    except (FileNotFoundError, GCSError) as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
