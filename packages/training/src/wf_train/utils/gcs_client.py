"""Centralized GCS client with fail-loud error handling.

This module provides a single source of truth for all GCS operations.
ALL GCS operations MUST fail loudly - no silent returns.

Usage:
    from wf_train.utils.gcs_client import get_gcs_client, with_gcs_retry, GCSError

    client, bucket = get_gcs_client(project="wrinklefree-481904", bucket_name="wrinklefree-checkpoints")

    @with_gcs_retry
    def upload_something():
        blob = bucket.blob("path/to/file")
        blob.upload_from_filename("local/path")
"""

from __future__ import annotations

import functools
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from google.cloud.storage import Bucket, Client

logger = logging.getLogger(__name__)

# Default GCS settings
DEFAULT_PROJECT = "wrinklefree-481904"
DEFAULT_BUCKET = "wrinklefree-checkpoints"
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 30  # seconds


# =============================================================================
# Exceptions - All GCS failures MUST raise these
# =============================================================================


class GCSError(Exception):
    """Base exception for all GCS operations.

    NEVER catch this and return None/False silently.
    Let it propagate to crash training if GCS is required.
    """

    pass


class GCSAuthError(GCSError):
    """Raised when GCS authentication fails."""

    pass


class GCSDownloadError(GCSError):
    """Raised when GCS download fails after retries."""

    pass


class GCSUploadError(GCSError):
    """Raised when GCS upload fails after retries."""

    pass


class GCSNotFoundError(GCSError):
    """Raised when a GCS object or bucket is not found."""

    pass


# =============================================================================
# Retry Decorator
# =============================================================================

T = TypeVar("T")


def with_gcs_retry(
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    delay: int = DEFAULT_RETRY_DELAY,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for GCS operations with retry logic.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        delay: Delay between retries in seconds (default: 30)
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function with retry logic

    Example:
        @with_gcs_retry(max_attempts=3, delay=30)
        def upload_checkpoint():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt == max_attempts:
                        logger.error(
                            f"GCS operation {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    logger.warning(
                        f"GCS operation {func.__name__} attempt {attempt}/{max_attempts} failed: {e}\n"
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)

            # Should never reach here, but satisfy type checker
            raise last_error or GCSError("Unknown error")

        return wrapper

    return decorator


# =============================================================================
# Client Management
# =============================================================================

# Global client cache to avoid repeated auth
_client_cache: dict[str, tuple[Any, Any]] = {}


def get_gcs_client(
    project: str | None = None,
    bucket_name: str | None = None,
    validate: bool = True,
) -> tuple["Client", "Bucket"]:
    """Get or create a GCS client and bucket reference.

    FAILS LOUDLY if authentication fails or bucket is not accessible.

    Args:
        project: GCP project ID (default: from env or DEFAULT_PROJECT)
        bucket_name: GCS bucket name (default: DEFAULT_BUCKET)
        validate: If True, validate bucket access on first call

    Returns:
        Tuple of (storage.Client, storage.Bucket)

    Raises:
        GCSAuthError: If authentication fails
        GCSNotFoundError: If bucket does not exist
        GCSError: For other GCS errors
    """
    # Get project from env or default
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
    bucket_name = bucket_name or DEFAULT_BUCKET

    cache_key = f"{project}:{bucket_name}"

    # Return cached client if available
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    try:
        from google.cloud import storage
        from google.auth import exceptions as auth_exceptions
    except ImportError as e:
        raise GCSAuthError(
            "google-cloud-storage package not installed.\n"
            "Install with: pip install google-cloud-storage\n"
            f"Error: {e}"
        ) from e

    # Check credentials file if specified
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and not Path(creds_path).exists():
        raise GCSAuthError(
            f"GOOGLE_APPLICATION_CREDENTIALS file not found: {creds_path}\n"
            "Either:\n"
            "  1. Set the correct path to your service account JSON\n"
            "  2. Unset GOOGLE_APPLICATION_CREDENTIALS to use default credentials"
        )

    try:
        # Create client with project
        client = storage.Client(project=project)

        # Get bucket reference
        bucket = client.bucket(bucket_name)

        # Validate bucket access if requested
        if validate:
            if not bucket.exists():
                raise GCSNotFoundError(
                    f"GCS bucket not found or not accessible: gs://{bucket_name}/\n"
                    f"Project: {project}\n"
                    "Check:\n"
                    "  1. Bucket name is correct\n"
                    "  2. You have storage.buckets.get permission\n"
                    "  3. The project ID matches the bucket's project"
                )

        # Cache for future use
        _client_cache[cache_key] = (client, bucket)

        logger.info(f"GCS client initialized: project={project}, bucket={bucket_name}")
        return client, bucket

    except auth_exceptions.DefaultCredentialsError as e:
        raise GCSAuthError(
            f"GCS authentication failed: {e}\n"
            "Either:\n"
            "  1. Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON\n"
            "  2. Run 'gcloud auth application-default login'\n"
            "  3. Use a service account on GCP infrastructure"
        ) from e

    except Exception as e:
        if isinstance(e, GCSError):
            raise
        raise GCSError(f"GCS client initialization failed: {e}") from e


def clear_client_cache() -> None:
    """Clear the client cache. Useful for testing."""
    _client_cache.clear()


# =============================================================================
# High-Level Operations
# =============================================================================


@with_gcs_retry()
def download_blob(
    bucket_name: str,
    blob_path: str,
    local_path: Path,
    project: str | None = None,
) -> Path:
    """Download a blob from GCS.

    FAILS LOUDLY if download fails after retries.

    Args:
        bucket_name: GCS bucket name
        blob_path: Path to blob in bucket
        local_path: Local path to download to
        project: GCP project ID

    Returns:
        Path to downloaded file

    Raises:
        GCSNotFoundError: If blob does not exist
        GCSDownloadError: If download fails after retries
    """
    _, bucket = get_gcs_client(project=project, bucket_name=bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise GCSNotFoundError(
            f"GCS blob not found: gs://{bucket_name}/{blob_path}\n"
            "The checkpoint may not exist or the path is incorrect."
        )

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        logger.info(f"Downloaded gs://{bucket_name}/{blob_path} to {local_path}")
        return local_path
    except Exception as e:
        raise GCSDownloadError(
            f"Failed to download gs://{bucket_name}/{blob_path}: {e}"
        ) from e


@with_gcs_retry()
def upload_blob(
    bucket_name: str,
    blob_path: str,
    local_path: Path,
    project: str | None = None,
    timeout: int = 600,
) -> str:
    """Upload a file to GCS.

    FAILS LOUDLY if upload fails after retries.

    Args:
        bucket_name: GCS bucket name
        blob_path: Path to blob in bucket
        local_path: Local path to upload from
        project: GCP project ID
        timeout: Upload timeout in seconds (default: 600)

    Returns:
        GCS URI of uploaded blob (gs://bucket/path)

    Raises:
        FileNotFoundError: If local file does not exist
        GCSUploadError: If upload fails after retries
    """
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    _, bucket = get_gcs_client(project=project, bucket_name=bucket_name)
    blob = bucket.blob(blob_path)

    try:
        blob.upload_from_filename(str(local_path), timeout=timeout)
        gcs_uri = f"gs://{bucket_name}/{blob_path}"
        logger.info(f"Uploaded {local_path} to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        raise GCSUploadError(
            f"Failed to upload {local_path} to gs://{bucket_name}/{blob_path}: {e}"
        ) from e


@with_gcs_retry()
def download_directory(
    bucket_name: str,
    prefix: str,
    local_dir: Path,
    project: str | None = None,
) -> Path:
    """Download all blobs with a given prefix from GCS.

    FAILS LOUDLY if download fails after retries.

    Args:
        bucket_name: GCS bucket name
        prefix: Prefix of blobs to download
        local_dir: Local directory to download to
        project: GCP project ID

    Returns:
        Path to local directory containing downloaded files

    Raises:
        GCSNotFoundError: If no blobs found with prefix
        GCSDownloadError: If download fails after retries
    """
    client, bucket = get_gcs_client(project=project, bucket_name=bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise GCSNotFoundError(
            f"No blobs found with prefix: gs://{bucket_name}/{prefix}\n"
            "The checkpoint directory may not exist."
        )

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        for blob in blobs:
            # Skip directory markers
            if blob.name.endswith("/"):
                continue

            # Compute relative path
            rel_path = blob.name[len(prefix) :].lstrip("/")
            if not rel_path:
                rel_path = Path(blob.name).name

            local_path = local_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            blob.download_to_filename(str(local_path))
            logger.debug(f"Downloaded {blob.name} to {local_path}")

        logger.info(
            f"Downloaded {len(blobs)} files from gs://{bucket_name}/{prefix} to {local_dir}"
        )
        return local_dir

    except Exception as e:
        raise GCSDownloadError(
            f"Failed to download gs://{bucket_name}/{prefix}: {e}"
        ) from e


def blob_exists(
    bucket_name: str,
    blob_path: str,
    project: str | None = None,
) -> bool:
    """Check if a blob exists in GCS.

    Args:
        bucket_name: GCS bucket name
        blob_path: Path to blob in bucket
        project: GCP project ID

    Returns:
        True if blob exists, False otherwise

    Raises:
        GCSAuthError: If authentication fails
    """
    _, bucket = get_gcs_client(project=project, bucket_name=bucket_name)
    blob = bucket.blob(blob_path)
    return blob.exists()


def validate_gcs_access(
    project: str | None = None,
    bucket_name: str | None = None,
) -> None:
    """Validate GCS access at startup.

    Call this early in training to fail fast if GCS is misconfigured.

    Raises:
        GCSAuthError: If authentication fails
        GCSNotFoundError: If bucket does not exist
        GCSError: For other GCS errors
    """
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
    bucket_name = bucket_name or DEFAULT_BUCKET

    logger.info(f"Validating GCS access: project={project}, bucket={bucket_name}")

    # This will raise if anything is wrong
    get_gcs_client(project=project, bucket_name=bucket_name, validate=True)

    logger.info("GCS access validated successfully")
