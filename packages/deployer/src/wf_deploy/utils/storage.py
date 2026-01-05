"""Storage path utilities for WrinkleFree Deployer.

Handles parsing and validation of storage paths across different backends:
- Local paths
- S3 (s3://)
- GCS (gs://)
- HuggingFace Hub (hf://)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class StorageType(Enum):
    """Storage backend types."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gs"
    HUGGINGFACE = "hf"
    R2 = "r2"  # Cloudflare R2 (S3-compatible)
    AZURE = "azure"


@dataclass
class StoragePath:
    """Parsed storage path with bucket/container and key."""

    storage_type: StorageType
    bucket: Optional[str]
    key: str
    original: str

    @property
    def is_remote(self) -> bool:
        """Check if this is a remote storage path."""
        return self.storage_type != StorageType.LOCAL

    @property
    def is_local(self) -> bool:
        """Check if this is a local path."""
        return self.storage_type == StorageType.LOCAL

    def to_uri(self) -> str:
        """Convert back to URI string."""
        if self.storage_type == StorageType.LOCAL:
            return self.key
        elif self.storage_type == StorageType.S3:
            return f"s3://{self.bucket}/{self.key}"
        elif self.storage_type == StorageType.GCS:
            return f"gs://{self.bucket}/{self.key}"
        elif self.storage_type == StorageType.HUGGINGFACE:
            return f"hf://{self.key}"
        elif self.storage_type == StorageType.AZURE:
            return f"azure://{self.bucket}/{self.key}"
        else:
            return self.original


def parse_storage_path(path: str) -> StoragePath:
    """Parse a storage path into its components.

    Args:
        path: Storage path (local, s3://, gs://, hf://, etc.)

    Returns:
        StoragePath with parsed components.

    Examples:
        >>> parse_storage_path("s3://my-bucket/models/model.gguf")
        StoragePath(storage_type=StorageType.S3, bucket="my-bucket", key="models/model.gguf", ...)

        >>> parse_storage_path("gs://my-bucket/models/model.gguf")
        StoragePath(storage_type=StorageType.GCS, bucket="my-bucket", key="models/model.gguf", ...)

        >>> parse_storage_path("hf://HuggingFaceTB/SmolLM2-135M-Instruct")
        StoragePath(storage_type=StorageType.HUGGINGFACE, bucket=None, key="HuggingFaceTB/SmolLM2-135M-Instruct", ...)

        >>> parse_storage_path("/path/to/model.gguf")
        StoragePath(storage_type=StorageType.LOCAL, bucket=None, key="/path/to/model.gguf", ...)
    """
    path = path.strip()

    # S3
    if path.startswith("s3://"):
        parts = path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return StoragePath(StorageType.S3, bucket, key, path)

    # GCS
    if path.startswith("gs://"):
        parts = path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return StoragePath(StorageType.GCS, bucket, key, path)

    # HuggingFace Hub
    if path.startswith("hf://"):
        key = path[5:]
        return StoragePath(StorageType.HUGGINGFACE, None, key, path)

    # Azure Blob
    if path.startswith("azure://"):
        parts = path[8:].split("/", 1)
        container = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return StoragePath(StorageType.AZURE, container, key, path)

    # R2 (Cloudflare, S3-compatible)
    if path.startswith("r2://"):
        parts = path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return StoragePath(StorageType.R2, bucket, key, path)

    # Local path
    return StoragePath(StorageType.LOCAL, None, path, path)


def validate_model_path(path: str) -> bool:
    """Validate a model path exists or is a valid remote path.

    Args:
        path: Model path to validate.

    Returns:
        True if path is valid (remote or existing local).

    Raises:
        ValueError: If local path doesn't exist.
    """
    parsed = parse_storage_path(path)

    if parsed.is_remote:
        # Remote paths are assumed valid (checked at runtime by SkyPilot)
        return True

    # Local path must exist
    if not Path(parsed.key).exists():
        raise ValueError(f"Local model path does not exist: {parsed.key}")

    return True
