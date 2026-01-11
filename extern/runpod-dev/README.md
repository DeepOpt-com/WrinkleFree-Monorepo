# runpod-dev

Rapid GPU development workflow for RunPod. Launch instances, sync files, and develop remotely with a single command.

## Features

- **One-command launch**: Create or connect to RunPod instances
- **Smart file sync**: Git-tracked files only by default (~90MB vs 16GB)
- **Live sync**: Optional real-time bidirectional sync with [Mutagen](https://mutagen.io)
- **Auto-setup**: Installs tmux, uv, and runs `uv sync`
- **JSON output**: Machine-readable output for AI/programmatic use
- **Reusable library**: Import and use programmatically

## Installation

```bash
pip install runpod-dev
# or
uv add runpod-dev
```

### System Dependencies

Install Mutagen for live file sync:

```bash
# macOS
brew install mutagen-io/mutagen/mutagen

# Linux
curl -fsSL https://github.com/mutagen-io/mutagen/releases/latest/download/mutagen_linux_amd64_v*.tar.gz | tar xz
sudo mv mutagen /usr/local/bin/

# Or download from: https://mutagen.io/documentation/introduction/installation
```

## Setup

Set your RunPod API key:

```bash
export RUNPOD_API_KEY=your_api_key

# Or save to file
echo "your_api_key" > ~/.runpod/api_key

# Or in .env file
echo "RUNPOD_API_KEY=your_api_key" >> .env
```

Get your API key at: https://runpod.io/console/user/settings

## Usage

### CLI

```bash
# Launch an H100 instance (syncs git files, sets up tmux/uv)
runpod-dev launch my-dev

# Launch with specific GPU
runpod-dev launch my-dev --gpu A100

# Launch with custom volume size (GB)
runpod-dev launch my-dev --volume-size 200

# Launch with live file watching
runpod-dev launch my-dev --watch

# Connect to existing instance (print SSH command)
runpod-dev connect my-dev

# Sync files to running instance (git-tracked only, fast)
runpod-dev sync my-dev

# Sync with live watching (for ongoing development)
runpod-dev sync my-dev --watch

# Monitor live sync status
runpod-dev sync my-dev --monitor

# Stop instance (preserves data, can resume later)
runpod-dev stop my-dev

# Terminate instance (deletes permanently)
runpod-dev stop my-dev --terminate

# List all instances
runpod-dev list

# List available GPU types
runpod-dev gpus

# Cleanup orphaned sync sessions
runpod-dev cleanup
```

### JSON Output (for AI/automation)

Add `--json` to any command for machine-readable output:

```bash
runpod-dev launch my-dev --json
```

Output:
```json
{
  "pod": {
    "id": "abc123",
    "name": "my-dev",
    "status": "RUNNING",
    "ssh": {
      "host": "123.45.67.89",
      "port": 22222,
      "ssh_command": "ssh -p 22222 root@123.45.67.89"
    }
  },
  "ssh_command": "ssh -p 22222 root@123.45.67.89",
  "sync_active": true
}
```

### Programmatic API

```python
from runpod_dev import PodManager, SyncManager

# Initialize with API key (or loads from env)
manager = PodManager(api_key="your_key")

# Launch or connect
pod = manager.find_by_name("my-dev")
if not pod:
    pod = manager.create("my-dev", gpu="H100")
    pod = manager.wait_for_ready(pod.id)

# Start file sync
sync = SyncManager(pod.ssh_info)
sync.create_session("my-dev", "./local", "/workspace/project")

# Check sync status
status = sync.get_status("my-dev")
print(f"Sync status: {status.status}")

# Stop when done
manager.stop("my-dev")
```

## Defaults

| Setting | Default |
|---------|---------|
| GPU | H100 |
| Image | `runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2404` |
| Python | 3.12 (Ubuntu 24.04) |
| CUDA | 12.9 |
| PyTorch | 2.6 |
| Volume Size | 100 GB |

## Available GPUs

| Name | RunPod ID |
|------|-----------|
| H100 | NVIDIA H100 80GB HBM3 |
| A100 | NVIDIA A100 80GB PCIe |
| A40 | NVIDIA A40 |
| L40S | NVIDIA L40S |
| RTX4090 | NVIDIA GeForce RTX 4090 |
| RTX6000 | NVIDIA RTX 6000 Ada Generation |

## File Sync

### Default Behavior (Recommended)
By default, `sync` uses `git ls-files` to sync only tracked files. This is fast (~90MB) and excludes large untracked files like models, venvs, and build artifacts.

### Live Sync with Mutagen
Use `--watch` to enable real-time bidirectional sync. Mutagen excludes these patterns:

| Category | Patterns |
|----------|----------|
| **Virtual envs** | `.venv/`, `venv/`, `.conda/` |
| **Models** | `*.pt`, `*.ckpt`, `*.safetensors`, `*.bin`, `*.gguf` |
| **Build artifacts** | `target/` (Rust), `build/`, `dist/`, `CMakeFiles/` |
| **Data files** | `*.parquet`, `*.arrow`, `*.h5`, `*.npy` |
| **Logs** | `wandb/`, `checkpoints/`, `outputs/`, `logs/` |
| **Cache** | `__pycache__/`, `.pytest_cache/`, `.mypy_cache/` |
| **Credentials** | `.env`, `.env.*`, `*-credentials.json` |

## License

MIT
