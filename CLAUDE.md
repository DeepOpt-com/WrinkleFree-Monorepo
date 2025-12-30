# WrinkleFree Monorepo

Monorepo for 1.58-bit quantized LLM research using uv workspaces.

## Package Map

| Package | Type | Purpose |
|---------|------|---------|
| `packages/training` | App | 1.58-bit training pipeline (BitDistill) |
| `packages/cheapertraining` | Lib | Shared data layer & utilities |
| `packages/fairy2` | App | Complex-valued quantization (Fairy2i) |
| `packages/inference` | App | Serving layer (sglang-bitnet) |
| `packages/eval` | App | Model evaluation (lm-eval) |
| `packages/deployer` | App | Cloud deployment (Modal/SkyPilot) |
| `packages/converter` | App | Model format conversion (DLM) |

## Quick Start

```bash
# Install all packages
uv sync --all-packages

# Run training
uv run --package wrinklefree python scripts/train.py model=smollm2_135m training=stage2_pretrain

# Run tests
uv run pytest
```

## Key Commands

| Task | Command |
|------|---------|
| Install all deps | `uv sync --all-packages` |
| Install one package | `uv sync --package wrinklefree` |
| Run in package context | `uv run --package wrinklefree python scripts/train.py` |
| Add dep to package | `cd packages/training && uv add torch` |
| Run all tests | `uv run pytest` |

## Shared Dependencies

`cheapertraining` is the shared library imported by other packages:

```
cheapertraining (library)
    │
    ├──► training (wrinklefree)
    │       Uses: cheapertraining.data, cheapertraining.influence
    │
    └──► fairy2
            Uses: cheapertraining.data
```

**Adding workspace dependencies**:
```toml
# In pyproject.toml
[project]
dependencies = ["cheapertraining"]

[tool.uv.sources]
cheapertraining = { workspace = true }
```

**Important**: Changes to cheapertraining affect training and fairy2 - test both after modifications.

## GCP Configuration

- **Project ID**: `wrinklefree-481904`
- **CPU Quota Limit**: 24 vCPUs per VM family in us-central1 (affects c3d, c2, etc.)
  - Use `c3d-standard-22` (22 vCPUs) instead of larger instances
  - Or request quota increase via GCP Console

## Remote Sync (gpucloud-dev)

The `gpucloud-dev` tool (in `extern/gpucloud-dev/`) handles all sync operations:

```bash
# Sync to RunPod instance
uv run gcd sync my-dev

# Sync to SSH host (uses .sync.conf preset)
uv run gcd sync-ssh desktop

# Smart sync (skips if watch active or no changes)
uv run gcd sync-ssh desktop --smart

# Sync with live watching
uv run gcd sync-ssh desktop --watch

# Check sync status (for AI agents)
uv run gcd status --json
```

### AI Agent Sync Protocol (CRITICAL)

Before running commands on remote servers:

1. **Use `--smart` flag** (or check sync status first):
   ```bash
   uv run gcd sync-ssh desktop --smart
   ```

2. **Decision logic** (handled automatically with `--smart`):
   - If `watch_active=true` → SKIP SYNC (mutagen handles it)
   - If `files_checksum` unchanged → SKIP SYNC
   - Otherwise → Run rsync

3. **Recommended workflow**:
   ```bash
   # Start watch mode once at session start
   uv run gcd sync-ssh desktop --watch

   # Subsequent operations: use --smart
   uv run gcd sync-ssh desktop --smart
   ```

This prevents the costly 90MB rsync on every command.

### SSH Presets

Configure `.sync.conf` for SSH hosts:
```ini
[project]
uv_projects=packages/training packages/inference

[desktop]
host=Desktop
dir=/home/lev/code/WrinkleFree
```

**Presets**: `desktop`, `runpod`, `RTX6000`

## Inference Engine Quick Start

```bash
# On Desktop: Start Streamlit chat interface
ssh Desktop 'cd /home/lev/code/WrinkleFree/packages/inference && \
  uv run streamlit run demo/serve_sglang.py --server.port 7860'

# Access at http://192.168.1.217:7860
```

## SSH Hosts

Desktop IP: `192.168.1.217` (configured in `~/.ssh/config`)

## RunPod Setup (ALWAYS DO THIS)

When starting a new RunPod instance, **always install gcloud**:

```bash
# Install gcloud CLI
curl -sSL https://sdk.cloud.google.com > /tmp/install_gcloud.sh
bash /tmp/install_gcloud.sh --disable-prompts --install-dir=/opt

# Add to PATH
echo 'export PATH=/opt/google-cloud-sdk/bin:$PATH' >> ~/.bashrc
export PATH=/opt/google-cloud-sdk/bin:$PATH

# Verify
gcloud --version
```

## Core Principles

- FAIL LOUDLY INSTEAD OF FALLBACKS
- DO NOT LAUNCH GPU INSTANCES ON GCP - use Nebius and RunPod
- Each package has its own CLAUDE.md with package-specific guidance

## Troubleshooting

### Package not found
```bash
uv sync --all-packages --reinstall
```

### Import errors between packages
Ensure workspace sources are configured:
```toml
[tool.uv.sources]
cheapertraining = { workspace = true }
```

### Submodule issues
```bash
git submodule update --init --recursive
```

## Documentation

- [Quick Start](docs/quick-start.md) - Installation and first steps
- [Architecture](docs/architecture.md) - System design and package relationships
- [Dependencies](docs/dependencies.md) - Dependency graph and version constraints
- [Development](docs/development.md) - Contributing and CI/CD

# SKY Pilot
### NEVER CANCEL JOBS THAT YOU DON'T OWN!!!! (ONLY CANCEL JOBS THAT YOU STARTED YOURSELF)
