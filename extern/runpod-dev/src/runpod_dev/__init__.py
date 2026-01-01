"""RunPod development workflow tool.

A Python package for rapid GPU development on RunPod instances.
Provides both a CLI and programmatic API for launching instances,
syncing files, and setting up development environments.

Basic CLI usage:
    $ runpod-dev launch my-dev --gpu H100
    $ runpod-dev sync my-dev
    $ runpod-dev stop my-dev

Programmatic usage:
    >>> from runpod_dev import PodManager, SyncManager
    >>>
    >>> # Launch or connect to an instance
    >>> manager = PodManager()
    >>> pod = manager.create("my-dev", gpu="H100")
    >>> pod = manager.wait_for_ready(pod.id)
    >>>
    >>> # Start file sync
    >>> sync = SyncManager(pod.ssh_info)
    >>> sync.create_session("my-dev", "./local", "/workspace/project")
    >>>
    >>> # Get sync status
    >>> status = sync.get_status("my-dev")
    >>> print(status.status)
    >>>
    >>> # Stop when done
    >>> manager.stop("my-dev")
"""

from .config import (
    DEFAULT_GPU,
    DEFAULT_IMAGE,
    DEFAULT_REMOTE_DIR,
    GPU_TYPES,
    SYNC_EXCLUDES,
    SSHInfo,
    get_api_key,
    get_gpu_type_id,
    list_gpu_types,
)
from .pod import PodInfo, PodManager
from .setup import (
    add_host_key,
    attach_tmux_session,
    create_tmux_session,
    install_dependencies,
    run_ssh_command,
    run_uv_sync,
    setup_gcs_credentials,
    wait_for_ssh,
)
from .sync import SyncManager, SyncStatus, rsync_git_files, terminate_all_sessions

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Config
    "DEFAULT_GPU",
    "DEFAULT_IMAGE",
    "DEFAULT_REMOTE_DIR",
    "GPU_TYPES",
    "SYNC_EXCLUDES",
    "SSHInfo",
    "get_api_key",
    "get_gpu_type_id",
    "list_gpu_types",
    # Pod management
    "PodInfo",
    "PodManager",
    # Sync
    "SyncManager",
    "SyncStatus",
    "rsync_git_files",
    "terminate_all_sessions",
    # Setup
    "add_host_key",
    "attach_tmux_session",
    "create_tmux_session",
    "install_dependencies",
    "run_ssh_command",
    "run_uv_sync",
    "setup_gcs_credentials",
    "wait_for_ssh",
]
