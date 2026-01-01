"""Remote environment setup utilities."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable

from .config import DEFAULT_REMOTE_DIR, SSHInfo


def run_ssh_command(
    ssh_info: SSHInfo,
    command: str,
    capture_output: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command on the remote pod via SSH.

    Args:
        ssh_info: SSH connection information.
        command: Command to execute.
        capture_output: Whether to capture stdout/stderr.
        check: Whether to raise on non-zero exit code.

    Returns:
        CompletedProcess result.
    """
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(ssh_info.port),
        f"{ssh_info.user}@{ssh_info.host}",
        command,
    ]

    return subprocess.run(
        ssh_cmd,
        capture_output=capture_output,
        text=True,
        check=check,
    )


def run_ssh_command_interactive(
    ssh_info: SSHInfo,
    command: str,
) -> int:
    """Run a command interactively (output to terminal).

    Args:
        ssh_info: SSH connection information.
        command: Command to execute.

    Returns:
        Exit code.
    """
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(ssh_info.port),
        f"{ssh_info.user}@{ssh_info.host}",
        command,
    ]

    result = subprocess.run(ssh_cmd)
    return result.returncode


def check_command_exists(ssh_info: SSHInfo, command: str) -> bool:
    """Check if a command exists on the remote.

    Args:
        ssh_info: SSH connection information.
        command: Command to check for.

    Returns:
        True if command exists.
    """
    result = run_ssh_command(ssh_info, f"which {command}")
    return result.returncode == 0


def install_dependencies(
    ssh_info: SSHInfo,
    on_progress: Callable[[str], None] | None = None,
) -> bool:
    """Install required dependencies on the remote pod.

    Installs: rsync, tmux, uv

    Args:
        ssh_info: SSH connection information.
        on_progress: Optional callback for progress updates.

    Returns:
        True if all dependencies installed successfully.
    """
    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Install rsync if not present
    if not check_command_exists(ssh_info, "rsync"):
        log("Installing rsync...")
        result = run_ssh_command(
            ssh_info,
            "apt-get update && apt-get install -y rsync",
        )
        if result.returncode != 0:
            return False
    else:
        log("rsync: already installed")

    # Install tmux if not present
    if not check_command_exists(ssh_info, "tmux"):
        log("Installing tmux...")
        result = run_ssh_command(
            ssh_info,
            "apt-get update && apt-get install -y tmux",
        )
        if result.returncode != 0:
            return False
    else:
        log("tmux: already installed")

    # Install uv if not present
    if not check_command_exists(ssh_info, "uv"):
        log("Installing uv...")
        result = run_ssh_command(
            ssh_info,
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
        )
        if result.returncode != 0:
            return False

        # Add to PATH in bashrc
        run_ssh_command(
            ssh_info,
            'grep -q "/.local/bin" ~/.bashrc || echo \'export PATH="$HOME/.local/bin:$PATH"\' >> ~/.bashrc',
        )
    else:
        log("uv: already installed")

    return True


def run_uv_sync(
    ssh_info: SSHInfo,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    all_packages: bool = True,
) -> bool:
    """Run uv sync on the remote pod.

    Args:
        ssh_info: SSH connection information.
        remote_dir: Remote directory containing the project.
        all_packages: If True, sync all workspace packages.

    Returns:
        True if sync succeeded.
    """
    sync_cmd = "uv sync --all-packages" if all_packages else "uv sync"

    command = f"""
    export PATH="$HOME/.local/bin:$PATH"
    cd {remote_dir}
    {sync_cmd}
    """

    # Run interactively so user sees output
    return run_ssh_command_interactive(ssh_info, command) == 0


def create_tmux_session(
    ssh_info: SSHInfo,
    name: str,
    remote_dir: str = DEFAULT_REMOTE_DIR,
) -> bool:
    """Create a named tmux session on the remote pod.

    Args:
        ssh_info: SSH connection information.
        name: Session name.
        remote_dir: Working directory for the session.

    Returns:
        True if session created or already exists.
    """
    # Check if session already exists
    result = run_ssh_command(ssh_info, f"tmux has-session -t {name} 2>/dev/null")
    if result.returncode == 0:
        return True  # Already exists

    # Create new session
    result = run_ssh_command(
        ssh_info,
        f"tmux new-session -d -s {name} -c {remote_dir}",
    )

    return result.returncode == 0


def attach_tmux_session(
    ssh_info: SSHInfo,
    name: str,
) -> int:
    """Attach to a tmux session (interactive).

    Args:
        ssh_info: SSH connection information.
        name: Session name.

    Returns:
        Exit code from tmux.
    """
    ssh_cmd = [
        "ssh",
        "-t",  # Force PTY allocation for tmux
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(ssh_info.port),
        f"{ssh_info.user}@{ssh_info.host}",
        f"tmux attach-session -t {name} || tmux new-session -s {name}",
    ]

    result = subprocess.run(ssh_cmd)
    return result.returncode


def add_host_key(ssh_info: SSHInfo) -> bool:
    """Add host key to known_hosts (for mutagen compatibility).

    Args:
        ssh_info: SSH connection information.

    Returns:
        True if key was added or already exists.
    """
    # Use ssh-keyscan to get the host key
    result = subprocess.run(
        ["ssh-keyscan", "-p", str(ssh_info.port), ssh_info.host],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not result.stdout.strip():
        return False

    # Append to known_hosts if not already there
    known_hosts_path = Path.home() / ".ssh" / "known_hosts"
    known_hosts_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if key already exists
    if known_hosts_path.exists():
        existing = known_hosts_path.read_text()
        # Check if this host:port combination is already in known_hosts
        host_pattern = f"[{ssh_info.host}]:{ssh_info.port}"
        if host_pattern in existing:
            return True

    # Append new key
    with open(known_hosts_path, "a") as f:
        f.write(result.stdout)

    return True


def setup_gcs_credentials(
    ssh_info: SSHInfo,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    credentials_subdir: str = "credentials",
    on_progress: Callable[[str], None] | None = None,
) -> bool:
    """Set up GCS credentials on the remote pod.

    Configures GOOGLE_APPLICATION_CREDENTIALS in .bashrc and activates
    the service account with gcloud if available.

    Args:
        ssh_info: SSH connection information.
        remote_dir: Remote project directory.
        credentials_subdir: Name of credentials subdirectory.
        on_progress: Optional callback for progress updates.

    Returns:
        True if setup succeeded or no credentials found.
    """
    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    creds_dir = f"{remote_dir}/{credentials_subdir}"
    gcs_file = f"{creds_dir}/gcp-service-account.json"

    # Check if GCS credentials file exists
    result = run_ssh_command(ssh_info, f"test -f {gcs_file}")
    if result.returncode != 0:
        log("No GCS credentials found, skipping")
        return True  # No credentials, nothing to do

    log("Setting up GCS credentials...")

    # Add GOOGLE_APPLICATION_CREDENTIALS to .bashrc if not already there
    check_cmd = 'grep -q "GOOGLE_APPLICATION_CREDENTIALS" ~/.bashrc'
    result = run_ssh_command(ssh_info, check_cmd)

    if result.returncode != 0:
        # Not in .bashrc, add it
        add_cmd = f'echo \'export GOOGLE_APPLICATION_CREDENTIALS="{gcs_file}"\' >> ~/.bashrc'
        run_ssh_command(ssh_info, add_cmd)
        log("Added GOOGLE_APPLICATION_CREDENTIALS to .bashrc")
    else:
        log("GOOGLE_APPLICATION_CREDENTIALS already in .bashrc")

    # Also source .env file if it exists (for WANDB_API_KEY, etc.)
    env_file = f"{creds_dir}/.env"
    result = run_ssh_command(ssh_info, f"test -f {env_file}")
    if result.returncode == 0:
        # Add sourcing of .env to .bashrc if not already there
        check_env_cmd = f'grep -q "source.*{credentials_subdir}/.env" ~/.bashrc'
        result = run_ssh_command(ssh_info, check_env_cmd)
        if result.returncode != 0:
            add_env_cmd = f'echo \'[ -f "{env_file}" ] && source "{env_file}"\' >> ~/.bashrc'
            run_ssh_command(ssh_info, add_env_cmd)
            log("Added .env sourcing to .bashrc")

    # Activate service account with gcloud if available (for gsutil)
    if check_command_exists(ssh_info, "gcloud"):
        log("Activating GCS service account with gcloud...")
        activate_cmd = f'gcloud auth activate-service-account --key-file="{gcs_file}" --quiet 2>/dev/null'
        result = run_ssh_command(ssh_info, activate_cmd)
        if result.returncode == 0:
            log("GCS service account activated")
        else:
            log("Warning: gcloud auth failed (gsutil may not work)")

    return True


def wait_for_ssh(
    ssh_info: SSHInfo,
    timeout: int = 120,
    poll_interval: int = 2,
    add_host_keys: bool = True,
) -> bool:
    """Wait for SSH to become available.

    Args:
        ssh_info: SSH connection information.
        timeout: Maximum wait time in seconds.
        poll_interval: Time between checks.
        add_host_keys: If True, automatically add host key to known_hosts.

    Returns:
        True if SSH is available.
    """
    import socket
    import time

    start = time.time()

    while time.time() - start < timeout:
        try:
            sock = socket.create_connection(
                (ssh_info.host, ssh_info.port),
                timeout=5,
            )
            sock.close()

            # Add host key for mutagen compatibility
            if add_host_keys:
                add_host_key(ssh_info)

            # Also verify SSH responds
            result = run_ssh_command(ssh_info, "echo ok", capture_output=True)
            if result.returncode == 0:
                return True

        except (socket.timeout, ConnectionRefusedError, OSError):
            pass

        time.sleep(poll_interval)

    return False
