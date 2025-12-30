"""Mutagen-based file synchronization."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .config import DEFAULT_REMOTE_DIR, SYNC_EXCLUDES, SSHInfo

# Mutagen session label prefix
SESSION_LABEL: Final[str] = "runpod-dev"


@dataclass
class SyncStatus:
    """Status of a sync session."""

    name: str
    status: str
    local_path: str
    remote_path: str
    conflicts: int = 0
    problems: list[str] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "name": self.name,
            "status": self.status,
            "local_path": self.local_path,
            "remote_path": self.remote_path,
            "conflicts": self.conflicts,
            "problems": self.problems,
        }


class SyncManager:
    """Manage file synchronization with Mutagen."""

    def __init__(self, ssh_info: SSHInfo):
        """Initialize the sync manager.

        Args:
            ssh_info: SSH connection info for the remote pod.
        """
        self.ssh_info = ssh_info
        self._check_mutagen_installed()

    def _check_mutagen_installed(self) -> None:
        """Check if mutagen is installed, exit with instructions if not."""
        if not shutil.which("mutagen"):
            from rich.console import Console
            console = Console(stderr=True)
            console.print("[red]Error:[/red] mutagen is not installed")
            console.print("\nInstall mutagen:")
            console.print("  macOS:  brew install mutagen-io/mutagen/mutagen")
            console.print("  Linux:  Download from https://mutagen.io/documentation/introduction/installation")
            console.print("  Windows: Download from https://mutagen.io/documentation/introduction/installation")
            sys.exit(1)

    def _session_name(self, name: str) -> str:
        """Generate a unique session name."""
        return f"{SESSION_LABEL}-{name}"

    def _build_ignore_args(self) -> list[str]:
        """Build mutagen ignore arguments from exclusion patterns."""
        args = []
        for pattern in SYNC_EXCLUDES:
            args.extend(["--ignore", pattern])
        return args

    def create_session(
        self,
        name: str,
        local_path: str | Path,
        remote_path: str = DEFAULT_REMOTE_DIR,
    ) -> bool:
        """Create a new mutagen sync session.

        Args:
            name: Session name (used for identification).
            local_path: Local directory to sync.
            remote_path: Remote directory path on the pod.

        Returns:
            True if session created successfully.
        """
        local_path = Path(local_path).resolve()
        session_name = self._session_name(name)

        # Build remote URL: user@host:port:path (mutagen SSH URL format)
        # The format is [user@]host[:port]:path
        remote_url = f"{self.ssh_info.user}@{self.ssh_info.host}:{self.ssh_info.port}:{remote_path}"

        # Build command
        cmd = [
            "mutagen", "sync", "create",
            "--name", session_name,
            # Sync mode: two-way-safe prevents destructive operations
            "--sync-mode", "two-way-safe",
            # Watch for changes
            "--watch-mode", "portable",
            # Ignore VCS directories
            "--ignore-vcs",
            # Ignore patterns
            *self._build_ignore_args(),
            # Paths
            str(local_path),
            remote_url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Check if session already exists
            if "already exists" in result.stderr:
                return True
            raise RuntimeError(f"Failed to create sync session: {result.stderr}")

        return True

    def terminate_session(self, name: str) -> bool:
        """Terminate a sync session.

        Args:
            name: Session name to terminate.

        Returns:
            True if terminated successfully.
        """
        session_name = self._session_name(name)

        result = subprocess.run(
            ["mutagen", "sync", "terminate", session_name],
            capture_output=True,
            text=True,
        )

        return result.returncode == 0

    def pause_session(self, name: str) -> bool:
        """Pause a sync session.

        Args:
            name: Session name to pause.

        Returns:
            True if paused successfully.
        """
        session_name = self._session_name(name)

        result = subprocess.run(
            ["mutagen", "sync", "pause", session_name],
            capture_output=True,
            text=True,
        )

        return result.returncode == 0

    def resume_session(self, name: str) -> bool:
        """Resume a paused sync session.

        Args:
            name: Session name to resume.

        Returns:
            True if resumed successfully.
        """
        session_name = self._session_name(name)

        result = subprocess.run(
            ["mutagen", "sync", "resume", session_name],
            capture_output=True,
            text=True,
        )

        return result.returncode == 0

    def flush_session(self, name: str) -> bool:
        """Force immediate sync of a session.

        Args:
            name: Session name to flush.

        Returns:
            True if flushed successfully.
        """
        session_name = self._session_name(name)

        result = subprocess.run(
            ["mutagen", "sync", "flush", session_name],
            capture_output=True,
            text=True,
        )

        return result.returncode == 0

    def get_status(self, name: str) -> SyncStatus | None:
        """Get status of a sync session.

        Args:
            name: Session name to check.

        Returns:
            SyncStatus if session exists, None otherwise.
        """
        session_name = self._session_name(name)

        result = subprocess.run(
            ["mutagen", "sync", "list", "--json"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return None

        try:
            data = json.loads(result.stdout)
            sessions = data.get("sessions", [])

            for session in sessions:
                if session.get("name") == session_name:
                    return SyncStatus(
                        name=name,
                        status=session.get("status", "unknown"),
                        local_path=session.get("alpha", {}).get("path", ""),
                        remote_path=session.get("beta", {}).get("path", ""),
                        conflicts=session.get("conflicts", 0),
                        problems=session.get("problems"),
                    )
        except json.JSONDecodeError:
            pass

        return None

    def list_sessions(self) -> list[SyncStatus]:
        """List all runpod-dev sync sessions.

        Returns:
            List of SyncStatus objects.
        """
        result = subprocess.run(
            ["mutagen", "sync", "list", "--json"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return []

        sessions = []
        try:
            data = json.loads(result.stdout)
            for session in data.get("sessions", []):
                name = session.get("name", "")
                if name.startswith(f"{SESSION_LABEL}-"):
                    # Strip prefix
                    short_name = name[len(SESSION_LABEL) + 1:]
                    sessions.append(SyncStatus(
                        name=short_name,
                        status=session.get("status", "unknown"),
                        local_path=session.get("alpha", {}).get("path", ""),
                        remote_path=session.get("beta", {}).get("path", ""),
                        conflicts=session.get("conflicts", 0),
                        problems=session.get("problems"),
                    ))
        except json.JSONDecodeError:
            pass

        return sessions

    def monitor(self, name: str) -> subprocess.Popen:
        """Start monitoring a sync session (blocking).

        Args:
            name: Session name to monitor.

        Returns:
            Popen process that can be terminated.
        """
        session_name = self._session_name(name)

        return subprocess.Popen(
            ["mutagen", "sync", "monitor", session_name],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )


def terminate_all_sessions() -> int:
    """Terminate all runpod-dev sync sessions.

    Returns:
        Number of sessions terminated.
    """
    result = subprocess.run(
        ["mutagen", "sync", "list", "--json"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return 0

    count = 0
    try:
        data = json.loads(result.stdout)
        for session in data.get("sessions", []):
            name = session.get("name", "")
            if name.startswith(f"{SESSION_LABEL}-"):
                subprocess.run(
                    ["mutagen", "sync", "terminate", name],
                    capture_output=True,
                )
                count += 1
    except json.JSONDecodeError:
        pass

    return count


def rsync_git_files(
    ssh_info: SSHInfo,
    local_path: str | Path,
    remote_path: str = DEFAULT_REMOTE_DIR,
    credentials_path: str | None = None,
    verbose: bool = True,
) -> bool:
    """Sync only git-tracked files using rsync.

    This is the recommended sync method - only syncs files in the git tree,
    avoiding large model files, venvs, and other untracked content.

    Args:
        ssh_info: SSH connection information.
        local_path: Local directory to sync (must be a git repo).
        remote_path: Remote directory path on the pod.
        credentials_path: Optional path to credentials to sync separately.
        verbose: If True, show rsync progress.

    Returns:
        True if sync succeeded.
    """
    local_path = Path(local_path).resolve()

    # Verify it's a git repo
    if not (local_path / ".git").exists():
        raise ValueError(f"{local_path} is not a git repository")

    # Create remote directory
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -p {ssh_info.port}"
    mkdir_result = subprocess.run(
        [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-p", str(ssh_info.port),
            f"{ssh_info.user}@{ssh_info.host}",
            f"mkdir -p {remote_path}",
        ],
        capture_output=True,
    )
    if mkdir_result.returncode != 0:
        return False

    # Get list of git-tracked files
    git_files = subprocess.run(
        ["git", "ls-files"],
        cwd=local_path,
        capture_output=True,
        text=True,
    )
    if git_files.returncode != 0:
        return False

    # Write file list to temp file for rsync --files-from
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(git_files.stdout)
        files_list_path = f.name

    try:
        # Build rsync command
        rsync_cmd = [
            "rsync",
            "-avz",
            "--progress" if verbose else "--quiet",
            "--files-from", files_list_path,
            "-e", f"ssh -o StrictHostKeyChecking=no -p {ssh_info.port}",
            str(local_path) + "/",
            f"{ssh_info.user}@{ssh_info.host}:{remote_path}/",
        ]

        result = subprocess.run(rsync_cmd)
        if result.returncode != 0:
            return False

        # Sync credentials separately if specified
        if credentials_path:
            creds_path = Path(credentials_path)
            if creds_path.exists():
                creds_rsync = [
                    "rsync",
                    "-avz",
                    "--progress" if verbose else "--quiet",
                    "-e", f"ssh -o StrictHostKeyChecking=no -p {ssh_info.port}",
                    str(creds_path) + "/",
                    f"{ssh_info.user}@{ssh_info.host}:{remote_path}/{creds_path.name}/",
                ]
                subprocess.run(creds_rsync)

        return True

    finally:
        # Clean up temp file
        Path(files_list_path).unlink(missing_ok=True)
