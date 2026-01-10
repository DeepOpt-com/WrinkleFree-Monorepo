"""Run manager for GCS-based checkpoint and metadata management.

Manages the lifecycle of training runs including:
- Status tracking (RUNNING, COMPLETED, INTERRUPTED, FAILED)
- Checkpoint upload/download to GCS
- Automatic resume detection

IMPORTANT: This module FAILS LOUDLY on all GCS errors.
If GCS is enabled but credentials are missing or operations fail, training will ABORT.
This is intentional - silent failures lead to lost work.

Example:
    >>> from wf_train.utils.run_manager import RunManager, RunStatus
    >>> manager = RunManager(
    ...     fingerprint="abc123...",
    ...     gcs_bucket="wrinklefree-checkpoints",
    ...     audit_logger=AuditLogger(),
    ... )
    >>> should_resume, ckpt_path, wandb_id = manager.check_and_resume()
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from wf_train.utils.gcs_client import (
    GCSAuthError,
    GCSDownloadError,
    GCSError,
    GCSNotFoundError,
    GCSUploadError,
    get_gcs_client,
    with_gcs_retry,
)

if TYPE_CHECKING:
    from google.cloud.storage import Bucket, Client

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    """Status of a training run."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"


# Keep CredentialsError for backward compatibility
CredentialsError = GCSAuthError


class RunManager:
    """Manages training run lifecycle with GCS integration.

    Handles:
    - Run status tracking via metadata.json in GCS
    - Checkpoint upload/download
    - Automatic resume detection

    FAILS LOUDLY on all GCS errors - no silent degradation.

    Attributes:
        fingerprint: SHA256 hash identifying this run
        gcs_bucket: GCS bucket name
        gcs_prefix: Prefix path in bucket
        audit_logger: Logger for warnings and errors
    """

    def __init__(
        self,
        fingerprint: str,
        gcs_bucket: str,
        audit_logger: Any,
        fingerprint_metadata: dict[str, Any] | None = None,
        gcs_prefix: str = "experiments",
        gcs_project: str | None = None,
        local_cache_dir: Path | None = None,
        rank: int = 0,
        skip_gcs: bool = False,
    ):
        """Initialize run manager.

        Args:
            fingerprint: SHA256 hash identifying this run
            gcs_bucket: GCS bucket name
            audit_logger: AuditLogger instance for warnings
            fingerprint_metadata: Metadata from fingerprint generation
            gcs_prefix: Prefix path in bucket (default: "experiments")
            gcs_project: GCP project ID (required for user credentials)
            local_cache_dir: Local directory for caching checkpoints
            rank: Process rank (only rank 0 does GCS operations)
            skip_gcs: If True, skip GCS operations entirely

        Raises:
            GCSAuthError: If GCS is enabled but credentials missing
        """
        self.fingerprint = fingerprint
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.gcs_project = gcs_project
        self.audit_logger = audit_logger
        self.fingerprint_metadata = fingerprint_metadata or {}
        self.rank = rank
        self.skip_gcs = skip_gcs

        # Local cache for downloaded checkpoints
        self.local_cache_dir = local_cache_dir or Path(tempfile.gettempdir())
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # GCS paths
        self.gcs_base_path = f"{gcs_prefix}/{fingerprint}"
        self.metadata_blob_path = f"{self.gcs_base_path}/metadata.json"
        self.checkpoint_blob_path = f"{self.gcs_base_path}/checkpoints/last.pt"
        self.config_blob_path = f"{self.gcs_base_path}/config.json"

        # Initialize GCS client (FAIL LOUDLY if credentials missing)
        self._client: Client | None = None
        self._bucket: Bucket | None = None
        if not skip_gcs and rank == 0:
            self._init_gcs_client()

    def _init_gcs_client(self) -> None:
        """Initialize GCS client using centralized client.

        FAILS LOUDLY if credentials are missing or invalid.
        This is intentional - silent failures lead to lost work.

        Raises:
            GCSAuthError: If credentials missing or invalid
        """
        try:
            self._client, self._bucket = get_gcs_client(
                project=self.gcs_project,
                bucket_name=self.gcs_bucket,
                validate=True,
            )
        except GCSError as e:
            # Log to audit logger for tracking
            self.audit_logger.log_credentials_missing(
                service="GCS",
                error=str(e),
                fingerprint=self.fingerprint,
            )
            raise

    def get_run_status(self) -> dict[str, Any] | None:
        """Get current run status from GCS.

        Returns:
            Metadata dict if run exists, None if run doesn't exist

        Raises:
            GCSError: If GCS operation fails (not just missing)
        """
        if self.skip_gcs or self.rank != 0:
            return None

        try:
            blob = self._bucket.blob(self.metadata_blob_path)
            if not blob.exists():
                return None

            content = blob.download_as_text()
            return json.loads(content)

        except Exception as e:
            # Log and re-raise - FAIL LOUDLY
            self.audit_logger.log_gcs_error(
                operation="get_run_status",
                error=str(e),
                fingerprint=self.fingerprint,
            )
            raise GCSError(f"Failed to get run status: {e}") from e

    def check_and_resume(self) -> tuple[bool, Path | None, str | None]:
        """Check if we should resume from an existing run.

        Returns:
            Tuple of (should_resume, checkpoint_path, wandb_run_id)
            - should_resume: True if we should resume
            - checkpoint_path: Local path to downloaded checkpoint
            - wandb_run_id: WandB run ID to resume (for continuous plots)

        Raises:
            GCSError: If GCS operations fail
        """
        if self.skip_gcs or self.rank != 0:
            return (False, None, None)

        metadata = self.get_run_status()
        if metadata is None:
            return (False, None, None)

        status = metadata.get("status")
        wandb_run_id = metadata.get("wandb_run_id")

        # Already completed - don't resume
        if status == RunStatus.COMPLETED:
            print(f"Run {self.fingerprint[:8]} already COMPLETED")
            return (False, None, wandb_run_id)

        # Running or Interrupted - try to resume
        if status in (RunStatus.RUNNING, RunStatus.INTERRUPTED):
            print(f"Found existing run in state: {status}")

            try:
                checkpoint_path = self.download_checkpoint()
                print(f"Downloaded checkpoint to: {checkpoint_path}")
                return (True, checkpoint_path, wandb_run_id)
            except GCSNotFoundError:
                print("No checkpoint found, starting fresh")
                return (False, None, None)

        # Failed - could resume or start fresh (start fresh for now)
        if status == RunStatus.FAILED:
            print("Previous run FAILED, starting fresh")
            return (False, None, None)

        return (False, None, None)

    @with_gcs_retry()
    def download_checkpoint(self, checkpoint_type: str = "last") -> Path:
        """Download checkpoint from GCS.

        FAILS LOUDLY if download fails.

        Args:
            checkpoint_type: Type of checkpoint ("last" or "best")

        Returns:
            Local path to downloaded checkpoint

        Raises:
            GCSNotFoundError: If checkpoint doesn't exist
            GCSDownloadError: If download fails after retries
        """
        if self.skip_gcs or self.rank != 0:
            raise GCSError("GCS is disabled or not rank 0")

        blob_path = f"{self.gcs_base_path}/checkpoints/{checkpoint_type}.pt"

        try:
            blob = self._bucket.blob(blob_path)
            if not blob.exists():
                raise GCSNotFoundError(
                    f"Checkpoint not found: gs://{self.gcs_bucket}/{blob_path}"
                )

            # Download to local cache
            local_path = self.local_cache_dir / f"{self.fingerprint[:8]}_{checkpoint_type}.pt"
            blob.download_to_filename(str(local_path))

            # Verify checkpoint
            if not self._verify_checkpoint(local_path):
                self.audit_logger.log_checkpoint_corrupted(
                    checkpoint_path=blob_path,
                    error="Failed to load checkpoint file",
                    fingerprint=self.fingerprint,
                )
                local_path.unlink(missing_ok=True)
                raise GCSDownloadError(
                    f"Checkpoint corrupted: gs://{self.gcs_bucket}/{blob_path}"
                )

            return local_path

        except (GCSNotFoundError, GCSDownloadError):
            raise
        except Exception as e:
            self.audit_logger.log_gcs_error(
                operation="download_checkpoint",
                error=str(e),
                fingerprint=self.fingerprint,
            )
            raise GCSDownloadError(f"Failed to download checkpoint: {e}") from e

    def _verify_checkpoint(self, path: Path) -> bool:
        """Verify checkpoint file is valid.

        Args:
            path: Path to checkpoint file

        Returns:
            True if checkpoint is valid
        """
        try:
            # Try to load checkpoint
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            # Check for required keys
            if not isinstance(checkpoint, dict):
                return False

            # At minimum, should have model_state_dict or be a state dict itself
            return True

        except Exception:
            return False

    @with_gcs_retry()
    def update_status(
        self,
        status: RunStatus,
        global_step: int | None = None,
        epoch: int | None = None,
        loss: float | None = None,
        wandb_run_id: str | None = None,
        wandb_url: str | None = None,
        error_message: str | None = None,
        **extra_fields: Any,
    ) -> None:
        """Update run status in GCS.

        FAILS LOUDLY if update fails.

        Args:
            status: New run status
            global_step: Current training step
            epoch: Current epoch
            loss: Current loss value
            wandb_run_id: WandB run ID
            wandb_url: WandB run URL (e.g., https://wandb.ai/entity/project/runs/abc123)
            error_message: Error message (for FAILED status)
            **extra_fields: Additional fields to store

        Raises:
            GCSUploadError: If update fails after retries
        """
        if self.skip_gcs or self.rank != 0:
            return

        try:
            # Get existing metadata or create new
            try:
                existing = self.get_run_status() or {}
            except GCSError:
                existing = {}

            # Build updated metadata
            metadata = {
                **existing,
                "fingerprint": self.fingerprint,
                "status": status.value if isinstance(status, RunStatus) else status,
                "last_updated": datetime.now().isoformat(),
                "git_commit": self.fingerprint_metadata.get("git_commit", "unknown"),
                "git_dirty": self.fingerprint_metadata.get("git_dirty", False),
                **extra_fields,
            }

            # Set created_at if new
            if "created_at" not in metadata:
                metadata["created_at"] = datetime.now().isoformat()

            # Update optional fields if provided
            if global_step is not None:
                metadata["global_step"] = global_step
            if epoch is not None:
                metadata["epoch"] = epoch
            if loss is not None:
                metadata["loss"] = float(loss)
            if wandb_run_id is not None:
                metadata["wandb_run_id"] = wandb_run_id
            if wandb_url is not None:
                metadata["wandb_url"] = wandb_url
            if error_message is not None:
                metadata["error_message"] = error_message

            # Upload metadata
            blob = self._bucket.blob(self.metadata_blob_path)
            blob.upload_from_string(
                json.dumps(metadata, indent=2, default=str),
                content_type="application/json",
            )

            logger.debug(f"Updated run status to {status}")

        except Exception as e:
            self.audit_logger.log_gcs_error(
                operation="update_status",
                error=str(e),
                fingerprint=self.fingerprint,
            )
            raise GCSUploadError(f"Failed to update run status: {e}") from e

    @with_gcs_retry()
    def upload_checkpoint(
        self,
        local_path: Path,
        checkpoint_type: str = "last",
        experiment_name: str | None = None,
        stage: str | None = None,
    ) -> str:
        """Upload checkpoint to GCS.

        FAILS LOUDLY if upload fails.

        Args:
            local_path: Path to local checkpoint file
            checkpoint_type: Name for this checkpoint ("final", "step_100", "best")
            experiment_name: If provided, uses checkpoint discovery path format
            stage: Training stage (e.g., "stage1_9", "stage2")

        Returns:
            GCS URI of uploaded checkpoint

        Path formats:
            With experiment_name/stage:
                gs://{bucket}/checkpoints/{experiment_name}/{stage}_checkpoint/checkpoints/{checkpoint_type}/checkpoint.pt
            Without (legacy fingerprint-based):
                gs://{bucket}/{gcs_prefix}/{fingerprint}/checkpoints/{checkpoint_type}.pt

        Raises:
            FileNotFoundError: If local checkpoint doesn't exist
            GCSUploadError: If upload fails after retries
        """
        if self.skip_gcs or self.rank != 0:
            raise GCSError("GCS is disabled or not rank 0")

        if not local_path.exists():
            raise FileNotFoundError(f"Local checkpoint not found: {local_path}")

        try:
            if experiment_name and stage:
                # New format: matches checkpoint discovery pattern
                blob_path = f"checkpoints/{experiment_name}/{stage}_checkpoint/checkpoints/{checkpoint_type}/checkpoint.pt"
            else:
                # Legacy fingerprint-based format
                blob_path = f"{self.gcs_base_path}/checkpoints/{checkpoint_type}.pt"

            blob = self._bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path), timeout=600)

            gcs_uri = f"gs://{self.gcs_bucket}/{blob_path}"
            print(f"Uploaded checkpoint to {gcs_uri}")
            return gcs_uri

        except Exception as e:
            self.audit_logger.log_gcs_error(
                operation="upload_checkpoint",
                error=str(e),
                fingerprint=self.fingerprint,
            )
            raise GCSUploadError(f"Failed to upload checkpoint: {e}") from e

    @with_gcs_retry()
    def upload_config(self, config: dict[str, Any]) -> str:
        """Upload resolved config to GCS for debugging.

        FAILS LOUDLY if upload fails.

        Args:
            config: Resolved config dict

        Returns:
            GCS URI of uploaded config

        Raises:
            GCSUploadError: If upload fails after retries
        """
        if self.skip_gcs or self.rank != 0:
            raise GCSError("GCS is disabled or not rank 0")

        try:
            blob = self._bucket.blob(self.config_blob_path)
            blob.upload_from_string(
                json.dumps(config, indent=2, default=str),
                content_type="application/json",
            )
            return f"gs://{self.gcs_bucket}/{self.config_blob_path}"

        except Exception as e:
            self.audit_logger.log_gcs_error(
                operation="upload_config",
                error=str(e),
                fingerprint=self.fingerprint,
            )
            raise GCSUploadError(f"Failed to upload config: {e}") from e

    def is_completed(self) -> bool:
        """Check if run is already completed.

        Returns:
            True if run status is COMPLETED

        Raises:
            GCSError: If status check fails
        """
        metadata = self.get_run_status()
        if metadata is None:
            return False
        return metadata.get("status") == RunStatus.COMPLETED


def create_run_manager(
    fingerprint: str,
    config: Any,
    audit_logger: Any,
    fingerprint_metadata: dict[str, Any] | None = None,
    rank: int = 0,
    skip_gcs: bool = False,
) -> RunManager | None:
    """Factory function to create RunManager from config.

    Args:
        fingerprint: Run fingerprint
        config: Hydra config with gcs settings
        audit_logger: AuditLogger instance
        fingerprint_metadata: Metadata from fingerprint generation
        rank: Process rank
        skip_gcs: If True, skip GCS entirely

    Returns:
        RunManager instance, or None if GCS disabled

    Raises:
        GCSAuthError: If GCS enabled but credentials missing
    """
    from omegaconf import OmegaConf

    # Get GCS config
    if isinstance(config, dict):
        gcs_config = config.get("gcs", {})
    else:
        gcs_config = OmegaConf.to_container(config.get("gcs", {}), resolve=True)

    # Check if GCS is enabled
    if not gcs_config.get("enabled", True):
        return None

    # Skip if requested
    if skip_gcs:
        return None

    return RunManager(
        fingerprint=fingerprint,
        gcs_bucket=gcs_config.get("bucket", "wrinklefree-checkpoints"),
        audit_logger=audit_logger,
        fingerprint_metadata=fingerprint_metadata,
        gcs_prefix=gcs_config.get("experiment_prefix", "experiments"),
        gcs_project=gcs_config.get("project"),
        rank=rank,
        skip_gcs=False,
    )
