"""Configuration and credential management for runpod-dev."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# GPU type mappings (short name -> RunPod GPU ID)
GPU_TYPES: Final[dict[str, str]] = {
    "H100": "NVIDIA H100 80GB HBM3",
    "H100-SXM": "NVIDIA H100 SXM",
    "A100": "NVIDIA A100 80GB PCIe",
    "A100-SXM": "NVIDIA A100-SXM4-80GB",
    "A40": "NVIDIA A40",
    "L40S": "NVIDIA L40S",
    "RTX4090": "NVIDIA GeForce RTX 4090",
    "RTX3090": "NVIDIA GeForce RTX 3090",
    "RTX6000": "NVIDIA RTX 6000 Ada Generation",
}

# Default configuration
DEFAULT_GPU: Final[str] = "H100"
# Ubuntu 24.04 has Python 3.12, CUDA 12.9, PyTorch 2.6
DEFAULT_IMAGE: Final[str] = "runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2404"
DEFAULT_VOLUME_SIZE: Final[int] = 100  # GB
DEFAULT_CONTAINER_DISK: Final[int] = 50  # GB
DEFAULT_REMOTE_DIR: Final[str] = "/workspace/project"

# Sync exclusion patterns - aggressive to keep sync small
# NOTE: These are used as fallback. Prefer git-only sync mode.
SYNC_EXCLUDES: Final[list[str]] = [
    # Virtual environments (CRITICAL - huge)
    ".venv",
    "venv",
    ".conda",
    "env",

    # Model files (CRITICAL - huge)
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    "*.bin",
    "*.gguf",
    "*.onnx",
    "models/",

    # Data files
    "*.parquet",
    "*.arrow",
    "*.h5",
    "*.hdf5",
    "*.npy",
    "*.npz",

    # Logs and outputs
    "wandb/",
    "checkpoints/",
    "outputs/",
    "logs/",
    "runs/",

    # Cache
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "*.egg-info",

    # Build artifacts
    "dist/",
    "build/",
    "*.egg",

    # Rust build artifacts (CRITICAL - huge)
    "target/",
    "Cargo.lock",

    # C/C++ build artifacts
    "*.o",
    "*.so",
    "*.a",
    "*.dylib",
    "cmake-build-*/",
    "CMakeFiles/",

    # IDE
    ".idea",
    ".vscode",

    # OS
    ".DS_Store",
    "Thumbs.db",

    # Node
    "node_modules/",

    # Credentials (NEVER sync via mutagen - use rsync separately)
    ".env",
    ".env.*",
    "*-credentials.json",
]


@dataclass
class SSHInfo:
    """SSH connection information for a pod."""

    host: str
    port: int
    user: str = "root"

    @property
    def ssh_command(self) -> str:
        """Return the SSH command to connect."""
        return f"ssh -o StrictHostKeyChecking=no -p {self.port} {self.user}@{self.host}"

    @property
    def rsync_target(self) -> str:
        """Return the rsync target string."""
        return f"{self.user}@{self.host}"

    @property
    def ssh_url(self) -> str:
        """Return SSH URL for mutagen."""
        return f"{self.user}@{self.host}:{self.port}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "ssh_command": self.ssh_command,
        }


def get_api_key(verbose: bool = True) -> str:
    """Load RunPod API key from environment or credential files.

    Search order:
    1. RUNPOD_API_KEY environment variable
    2. packages/deployer/credentials/.env (if in WrinkleFree repo)
    3. ~/.config/.env.global
    4. ~/.runpod/api_key

    Args:
        verbose: If True, print helpful error messages on failure.

    Returns:
        The API key string.

    Raises:
        SystemExit: If no API key is found.
    """
    # 1. Check environment variable
    if api_key := os.environ.get("RUNPOD_API_KEY"):
        return api_key

    # 2. Check common credential file locations
    credential_paths = [
        Path("packages/deployer/credentials/.env"),
        Path.home() / ".config" / ".env.global",
        Path.home() / ".runpod" / "api_key",
    ]

    for path in credential_paths:
        if path.exists():
            # Check if it's a plain API key file (no extension, not .env)
            if path.name == "api_key" or (path.suffix == "" and ".env" not in path.name):
                # Plain text file (e.g., ~/.runpod/api_key)
                api_key = path.read_text().strip()
                if api_key and not api_key.startswith("#"):
                    return api_key
            else:
                # .env file - parse it
                try:
                    from dotenv import dotenv_values
                    values = dotenv_values(path)
                    if api_key := values.get("RUNPOD_API_KEY"):
                        return api_key
                except ImportError:
                    # Fall back to simple parsing
                    for line in path.read_text().splitlines():
                        line = line.strip()
                        # Skip comments and empty lines
                        if not line or line.startswith("#"):
                            continue
                        # Handle "export VAR=value" format
                        if line.startswith("export "):
                            line = line[7:]  # Remove "export " prefix
                        if line.startswith("RUNPOD_API_KEY="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")

    # No API key found
    if verbose:
        from rich.console import Console
        console = Console(stderr=True)
        console.print("[red]Error:[/red] RUNPOD_API_KEY not found")
        console.print("\nSet it via one of:")
        console.print("  1. Environment variable: export RUNPOD_API_KEY=your_key")
        console.print("  2. File: echo 'your_key' > ~/.runpod/api_key")
        console.print("  3. .env file with RUNPOD_API_KEY=your_key")
        console.print("\nGet your API key at: https://runpod.io/console/user/settings")

    sys.exit(1)


def get_gpu_type_id(gpu: str) -> str:
    """Convert GPU short name to RunPod GPU type ID.

    Args:
        gpu: Short name (e.g., "H100") or full ID.

    Returns:
        RunPod GPU type ID string.
    """
    return GPU_TYPES.get(gpu.upper(), gpu)


def list_gpu_types() -> list[dict[str, str]]:
    """Return list of available GPU types for display.

    Returns:
        List of dicts with 'name' and 'id' keys.
    """
    return [{"name": name, "id": gpu_id} for name, gpu_id in GPU_TYPES.items()]
