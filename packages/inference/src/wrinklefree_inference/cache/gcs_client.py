"""GCS client for model cache operations."""

import logging
import os
from pathlib import Path
from typing import Optional

from google.cloud import storage

logger = logging.getLogger(__name__)

GCS_BUCKET = "wrinklefree-models"
GCS_CACHE_PREFIX = "cache"


class GCSModelCache:
    """GCS-backed cache for packed BitNet models."""

    def __init__(
        self,
        bucket_name: str = GCS_BUCKET,
        cache_prefix: str = GCS_CACHE_PREFIX,
        local_cache_dir: Optional[str] = None,
    ):
        self.bucket_name = bucket_name
        self.cache_prefix = cache_prefix
        self.local_cache_dir = Path(
            local_cache_dir or os.path.expanduser("~/.cache/wrinklefree/models")
        )
        self._client: Optional[storage.Client] = None

    @property
    def client(self) -> storage.Client:
        if self._client is None:
            self._client = storage.Client(project="wrinklefree-481904")
        return self._client

    @property
    def bucket(self) -> storage.Bucket:
        return self.client.bucket(self.bucket_name)

    def cache_key_to_gcs_path(self, cache_key: str) -> str:
        """Convert cache key to GCS path."""
        return f"{self.cache_prefix}/{cache_key}/"

    def cache_key_to_local_path(self, cache_key: str) -> Path:
        """Convert cache key to local cache path."""
        return self.local_cache_dir / cache_key

    def exists(self, cache_key: str) -> bool:
        """Check if cached model exists in GCS."""
        gcs_path = self.cache_key_to_gcs_path(cache_key)
        # Check for config.json as marker file
        blob = self.bucket.blob(f"{gcs_path}config.json")
        return blob.exists()

    def download(self, cache_key: str) -> Path:
        """Download cached model from GCS to local cache.

        Returns local path to downloaded model.
        """
        local_path = self.cache_key_to_local_path(cache_key)

        # Check if already in local cache
        if (local_path / "config.json").exists():
            logger.info(f"Using local cache: {local_path}")
            return local_path

        gcs_path = self.cache_key_to_gcs_path(cache_key)
        logger.info(f"Downloading from gs://{self.bucket_name}/{gcs_path}")

        local_path.mkdir(parents=True, exist_ok=True)

        # List and download all blobs
        blobs = self.client.list_blobs(self.bucket_name, prefix=gcs_path)
        for blob in blobs:
            relative_name = blob.name[len(gcs_path) :]
            if not relative_name:
                continue
            local_file = local_path / relative_name
            local_file.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file))
            logger.debug(f"Downloaded: {relative_name}")

        return local_path

    def upload(self, local_model_path: Path, cache_key: str) -> str:
        """Upload packed model to GCS cache.

        Returns GCS URI (gs://...).
        """
        gcs_path = self.cache_key_to_gcs_path(cache_key)
        logger.info(f"Uploading to gs://{self.bucket_name}/{gcs_path}")

        for local_file in local_model_path.rglob("*"):
            if local_file.is_file():
                relative_name = local_file.relative_to(local_model_path)
                blob = self.bucket.blob(f"{gcs_path}{relative_name}")
                blob.upload_from_filename(str(local_file))
                logger.debug(f"Uploaded: {relative_name}")

        return f"gs://{self.bucket_name}/{gcs_path}"
