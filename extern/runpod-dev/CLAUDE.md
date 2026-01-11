# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

`runpod-dev` is a Python CLI/library for rapid GPU development on RunPod instances:
- Launch/connect to RunPod instances (H100 default)
- Live sync files via Mutagen (git-tracked files only)
- Auto-setup tmux and uv on remote instances
- JSON output mode for AI/programmatic interaction

## Architecture

```
src/runpod_dev/
├── __init__.py     # Public API exports (PodManager, SyncManager, etc.)
├── cli.py          # Click CLI entry point (runpod-dev command)
├── config.py       # Configuration, GPU types, API key loading
├── pod.py          # RunPod instance management (create/stop/terminate)
├── sync.py         # Mutagen sync + rsync_git_files()
└── setup.py        # Remote setup (tmux, uv, SSH utilities)
```

## Key Commands

```bash
# Run CLI (from WrinkleFree monorepo)
uv run runpod-dev --help

# Launch instance with sync
uv run runpod-dev launch my-dev --gpu H100

# List instances
uv run runpod-dev list

# Sync git-tracked files only (default, fast ~90MB)
uv run runpod-dev sync my-dev

# Sync + start live watching with mutagen
uv run runpod-dev sync my-dev --watch

# Monitor live sync status
uv run runpod-dev sync my-dev --monitor

# Stop instance
uv run runpod-dev stop my-dev

# Terminate (permanent delete)
uv run runpod-dev stop my-dev --terminate
```

## Best Practices

### Always Use tmux on RunPod
**IMPORTANT**: When running commands on RunPod instances, always use tmux to prevent jobs from dying if SSH disconnects.

```bash
# The launch command automatically creates a tmux session
uv run runpod-dev launch my-dev

# To reattach to an existing tmux session:
ssh -p PORT root@HOST -t "tmux attach -t my-dev || tmux new -s my-dev"

# Or manually inside SSH:
tmux attach -t my-dev   # Reattach
tmux new -s my-dev      # Create new
tmux ls                 # List sessions
```

Long-running training jobs, experiments, and builds should always run inside tmux.

### GCS Credentials Setup
When using `--credentials`, GCS is automatically configured:

```bash
# Launch with credentials
uv run runpod-dev launch my-dev --credentials credentials/

# Or sync to existing instance
uv run runpod-dev sync my-dev --credentials credentials/
```

This will:
1. Sync `gcp-service-account.json` to remote
2. Add `GOOGLE_APPLICATION_CREDENTIALS` to `.bashrc`
3. Source `.env` file in `.bashrc` (for WANDB_API_KEY, etc.)
4. Activate service account with `gcloud` if available (for gsutil)

## Critical Design Decisions

### 1. Two-Phase Sync Strategy
**Default behavior** (`sync` command without `--watch`):
- Uses `rsync_git_files()` - syncs only git-tracked files (~90MB vs 16GB)
- Fast one-time sync using `git ls-files` + rsync `--files-from`
- Location: `sync.py:rsync_git_files()`

**With `--watch` flag**:
- First runs rsync of git files (fast initial sync)
- Then starts Mutagen for live bidirectional watching
- Mutagen uses SYNC_EXCLUDES patterns (target/, .venv/, *.pt, etc.)
- Location: `sync.py:SyncManager.create_session()`

### 2. SYNC_EXCLUDES Patterns (config.py)
Comprehensive exclusions for mutagen to prevent syncing large files:
- `.venv/`, `target/` (Rust), `build/`, `CMakeFiles/`
- `*.pt`, `*.ckpt`, `*.safetensors`, `*.bin` (models)
- `wandb/`, `checkpoints/`, `outputs/`
- `node_modules/`, `__pycache__/`

### 3. API Key Loading
API keys are loaded from (in order):
1. `RUNPOD_API_KEY` environment variable
2. `packages/deployer/credentials/.env`
3. `~/.config/.env.global`
4. `~/.runpod/api_key`

**Important**: `.env` files have empty suffix in pathlib, so we check `".env" in path.name`.

Location: `config.py:get_api_key()`

## Testing Commands

```bash
# List pods (verifies API key works)
uv run runpod-dev list --json

# Check sync status
uv run runpod-dev sync my-dev --status

# Cleanup orphaned sync sessions
uv run runpod-dev cleanup
```

## Common Issues

### "Invalid header value" when running commands
- API key is loading incorrectly (likely reading entire file as key)
- Check `config.py:get_api_key()` for .env parsing logic

### Sync too large (GB instead of KB)
- Ensure using `rsync_git_files()` not raw mutagen
- Verify `git ls-files` returns expected files

### Host key verification failed
- `add_host_key()` should be called before mutagen sync
- Uses `ssh-keyscan` to add keys to known_hosts

## Integration with WrinkleFree

This package is a git submodule at `extern/runpod-dev`.

```bash
# Update submodule
git submodule update --remote extern/runpod-dev

# Push changes (from extern/runpod-dev directory)
cd extern/runpod-dev && git push origin master
```

## Programmatic Usage

```python
from runpod_dev import PodManager, SyncManager, rsync_git_files

# Launch or find pod
manager = PodManager()
pod = manager.find_by_name("my-dev") or manager.create("my-dev", gpu="H100")
pod = manager.wait_for_ready(pod.id)

# Sync git files + credentials
rsync_git_files(pod.ssh_info, "/local/path", credentials_path="credentials/")

# Start live watching
sync = SyncManager(pod.ssh_info)
sync.create_session("my-dev", "/local/path", "/workspace/project")
```
