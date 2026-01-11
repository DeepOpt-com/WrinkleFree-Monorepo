"""Audit logging for training runs.

Logs warnings and critical events to a committed folder for permanent audit trail.
These logs are NOT in .gitignore - they provide accountability for training runs.

Example:
    >>> logger = AuditLogger()
    >>> logger.log_warning("dirty_git", {
    ...     "fingerprint": "abc123...",
    ...     "git_commit": "def456...",
    ...     "message": "Training with uncommitted changes"
    ... })
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! WARNING: DIRTY GIT
    !!! fingerprint: abc123...
    !!! git_commit: def456...
    !!! message: Training with uncommitted changes
    !!! Log saved to: training_logs/warnings/
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class AuditLogger:
    """Logger for persistent warnings and audit events.

    Logs are saved to a committed folder (not in .gitignore) for permanent
    audit trail. Each warning is a separate JSON file for easy tracking.

    Attributes:
        log_dir: Directory for warning logs
        max_files: Maximum number of files to keep (oldest deleted)
        enabled: Whether logging is enabled
    """

    # Warning types and their severity
    WARNING_TYPES = {
        "dirty_git": "WARNING",
        "gcs_auth_failed": "CRITICAL",
        "checkpoint_corrupted": "ERROR",
        "resume_config_mismatch": "WARNING",
        "network_error": "ERROR",
        "training_interrupted": "INFO",
        "training_failed": "ERROR",
        "credentials_missing": "CRITICAL",
    }

    def __init__(
        self,
        log_dir: Path | str = "training_logs/warnings",
        max_files: int = 100,
        max_age_days: int = 90,
        enabled: bool = True,
    ):
        """Initialize audit logger.

        Args:
            log_dir: Directory for warning logs (relative to project root)
            max_files: Maximum number of log files to keep
            max_age_days: Delete files older than this
            enabled: Whether logging is enabled
        """
        self.log_dir = Path(log_dir)
        self.max_files = max_files
        self.max_age_days = max_age_days
        self.enabled = enabled

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_warning(
        self,
        warning_type: str,
        details: dict[str, Any],
        also_raise: bool = False,
    ) -> Path | None:
        """Log a warning to the audit folder.

        Args:
            warning_type: Type of warning (e.g., "dirty_git", "gcs_auth_failed")
            details: Dict of warning details
            also_raise: If True, raise an exception after logging

        Returns:
            Path to the log file, or None if logging disabled
        """
        if not self.enabled:
            # Still print banner even if file logging disabled
            self._print_banner(warning_type, details)
            if also_raise:
                raise RuntimeError(f"{warning_type}: {details.get('message', '')}")
            return None

        timestamp = datetime.now()
        fingerprint = details.get("fingerprint", "unknown")[:8]

        # Filename: YYYY-MM-DD_HH-MM-SS_{type}_{fingerprint}.json
        filename = (
            f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
            f"_{warning_type}_{fingerprint}.json"
        )

        # Build log entry
        severity = self.WARNING_TYPES.get(warning_type, "WARNING")
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "type": warning_type,
            "severity": severity,
            **details,
        }

        # Write JSON file (not .log to avoid .gitignore)
        log_path = self.log_dir / filename
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2, default=str)

        # Print LOUD banner
        self._print_banner(warning_type, details, log_path)

        # Rotate old files
        self._rotate_logs()

        if also_raise:
            raise RuntimeError(f"{warning_type}: {details.get('message', '')}")

        return log_path

    def _print_banner(
        self,
        warning_type: str,
        details: dict[str, Any],
        log_path: Path | None = None,
    ) -> None:
        """Print a LOUD warning banner to stderr.

        Args:
            warning_type: Type of warning
            details: Warning details
            log_path: Path to log file (if saved)
        """
        severity = self.WARNING_TYPES.get(warning_type, "WARNING")

        # Use different borders for different severities
        if severity == "CRITICAL":
            border_char = "X"
            prefix = "XXX CRITICAL"
        elif severity == "ERROR":
            border_char = "!"
            prefix = "!!! ERROR"
        else:
            border_char = "!"
            prefix = "!!! WARNING"

        border = border_char * 72

        # Print to stderr for visibility
        print(f"\n{border}", file=sys.stderr)
        print(f"{prefix}: {warning_type.upper().replace('_', ' ')}", file=sys.stderr)

        for key, value in details.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:97] + "..."
            print(f"{prefix.split()[0]} {key}: {str_value}", file=sys.stderr)

        if log_path:
            print(f"{prefix.split()[0]} Log saved to: {log_path}", file=sys.stderr)

        print(f"{border}\n", file=sys.stderr)

    def _rotate_logs(self) -> None:
        """Delete old log files to stay under limits."""
        if not self.log_dir.exists():
            return

        # Get all JSON files sorted by modification time (oldest first)
        log_files = sorted(
            self.log_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )

        # Delete files over max count
        while len(log_files) > self.max_files:
            oldest = log_files.pop(0)
            oldest.unlink()

        # Delete files over max age
        cutoff = datetime.now().timestamp() - (self.max_age_days * 24 * 60 * 60)
        for log_file in log_files:
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()

    def log_dirty_git(
        self,
        fingerprint: str,
        git_commit: str,
        message: str = "Training with uncommitted changes",
    ) -> Path | None:
        """Convenience method for dirty git warnings."""
        return self.log_warning(
            "dirty_git",
            {
                "fingerprint": fingerprint,
                "git_commit": git_commit,
                "message": message,
                "recommendation": "Commit changes before production runs",
            },
        )

    def log_credentials_missing(
        self,
        service: str,
        error: str,
        fingerprint: str | None = None,
    ) -> Path | None:
        """Log missing credentials - CRITICAL error.

        This logs the error LOUDLY but does NOT raise an exception.
        The caller (e.g., RunManager) is responsible for raising the
        appropriate exception type (e.g., CredentialsError).

        Args:
            service: Service name (e.g., "GCS", "WandB")
            error: Error message
            fingerprint: Run fingerprint if available
        """
        return self.log_warning(
            "credentials_missing",
            {
                "service": service,
                "error": error,
                "fingerprint": fingerprint or "unknown",
                "message": f"{service} credentials not found or invalid",
                "action": f"{service} DISABLED - training continues without checkpoint sync. "
                          f"To fix: set up credentials or use gcs.enabled=false",
            },
            also_raise=False,  # Caller raises appropriate exception
        )

    def log_gcs_error(
        self,
        operation: str,
        error: str,
        fingerprint: str | None = None,
    ) -> Path | None:
        """Log GCS operation error."""
        return self.log_warning(
            "gcs_auth_failed",
            {
                "operation": operation,
                "error": error,
                "fingerprint": fingerprint or "unknown",
                "message": f"GCS {operation} failed",
            },
        )

    def log_checkpoint_corrupted(
        self,
        checkpoint_path: str,
        error: str,
        fingerprint: str | None = None,
    ) -> Path | None:
        """Log corrupted checkpoint."""
        return self.log_warning(
            "checkpoint_corrupted",
            {
                "checkpoint_path": checkpoint_path,
                "error": error,
                "fingerprint": fingerprint or "unknown",
                "message": "Checkpoint file is corrupted or invalid",
                "action": "Will start from scratch if no valid checkpoint found",
            },
        )

    def log_training_interrupted(
        self,
        fingerprint: str,
        global_step: int,
        message: str = "Training interrupted by user",
    ) -> Path | None:
        """Log training interruption."""
        return self.log_warning(
            "training_interrupted",
            {
                "fingerprint": fingerprint,
                "global_step": global_step,
                "message": message,
            },
        )

    def log_training_failed(
        self,
        fingerprint: str,
        error: str,
        traceback: str | None = None,
    ) -> Path | None:
        """Log training failure."""
        return self.log_warning(
            "training_failed",
            {
                "fingerprint": fingerprint,
                "error": error,
                "traceback": traceback or "",
                "message": "Training failed with exception",
            },
        )
