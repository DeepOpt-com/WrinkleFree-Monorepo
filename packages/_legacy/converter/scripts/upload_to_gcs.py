#!/usr/bin/env python
"""Upload output directory to GCS bucket."""

import argparse
import os
from pathlib import Path

from google.cloud import storage


def upload_to_gcs(output_dir: str, bucket_name: str, gcs_prefix: str):
    """Upload all files from output_dir to GCS bucket under gcs_prefix."""
    output_path = Path(os.path.expanduser(output_dir))

    if not output_path.exists():
        print(f"Error: Output directory {output_path} does not exist")
        return False

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    uploaded = 0
    for path in output_path.rglob("*"):
        if path.is_file():
            blob_name = f"{gcs_prefix}/{path.relative_to(output_path)}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(path))
            print(f"Uploaded: gs://{bucket_name}/{blob_name}")
            uploaded += 1

    print(f"\nAll {uploaded} files uploaded to gs://{bucket_name}/{gcs_prefix}/")
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload output to GCS")
    parser.add_argument("--output-dir", required=True, help="Local output directory")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--prefix", required=True, help="GCS prefix (folder path)")
    args = parser.parse_args()

    success = upload_to_gcs(args.output_dir, args.bucket, args.prefix)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
