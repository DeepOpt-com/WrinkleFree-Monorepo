#!/usr/bin/env python3
"""Upload evaluation results to GCS bucket.

Usage:
    OUTPUT_DIR=/tmp/eval_results GCS_BUCKET=wrinklefree-results GCS_PREFIX=eval/run1 \
        python scripts/upload_results.py
"""
import os
import sys
from pathlib import Path

from google.cloud import storage


def upload_directory_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str):
    """Upload all files from local_dir to GCS bucket under gcs_prefix."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"Error: {local_dir} does not exist")
        sys.exit(1)

    file_count = 0
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(local_dir)
            gcs_path = f"{gcs_prefix}/{rel_path}"

            print(f"Uploading {file_path} -> gs://{bucket_name}/{gcs_path}")
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(file_path))
            file_count += 1

    print(f"\nUploaded {file_count} files to gs://{bucket_name}/{gcs_prefix}")
    print(f"View: https://console.cloud.google.com/storage/browser/{bucket_name}/{gcs_prefix}")


if __name__ == "__main__":
    local_dir = os.environ.get("OUTPUT_DIR", "/tmp/eval_results")
    bucket_name = os.environ.get("GCS_BUCKET", "wrinklefree-results")
    gcs_prefix = os.environ.get("GCS_PREFIX", "eval-results")

    upload_directory_to_gcs(local_dir, bucket_name, gcs_prefix)
