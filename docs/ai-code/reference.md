# WrinkleFree AI Reference

Extended documentation for AI assistants. See main `CLAUDE.md` for rules.

## Training Pipeline Overview

WrinkleFree implements **BitDistill** for training 1.58-bit (ternary) LLMs:

```
Stage 1: SubLN Insertion (packages/training)
    │   Convert model: add BitLinear + SubLN layers
    ▼
Stage 1.9: Layer-wise Distillation (packages/training)
    │   Align hidden states with teacher (~100M tokens)
    ▼
Stage 2: Continue Pre-training (packages/training)
    │   QAT with gradual quantization warmup (~10B tokens)
    ▼
Stage 3: Knowledge Distillation (packages/training - objectives)
    │   BitDistill or TCS objectives with teacher guidance
    ▼
LRC: Post-quantization Correction (packages/training)
    │   Low-Rank Correction for error recovery (~50M tokens)
    ▼
Export: Convert to GGUF (see root CLAUDE.md for workflow)
    │
    ▼
Serve: Inference with BitNet.cpp (packages/inference)
```

**Key insight**: All training stages are in `packages/training`. Distillation uses the objectives system.

## Package Map (Full)

| Package | Type | Purpose | Key Entry Point |
|---------|------|---------|-----------------|
| `packages/training` | App | 1.58-bit training (all stages) + distillation + LRC | `scripts/train.py` |
| `packages/architecture` | Lib | BitNet layers (BitLinear, BitLinearLRC, SubLN) & conversion | Import as library |
| `packages/data_handler` | Lib | Data loading & influence functions | Import as library |
| `packages/inference` | App | Model serving (sglang-bitnet) | `demo/serve_sglang.py` |
| `packages/eval` | App | Model evaluation (lm-eval) | `scripts/evaluate.py` |
| `packages/deployer` | App | Cloud deployment (SkyPilot) | `wf` CLI |
| `packages/mobile` | App | Android inference | Android app |

> **Note**: Legacy packages (`distillation`, `converter`, `cheapertraining`) are archived in `packages/_legacy/`.

## Shared Dependencies

`data_handler` is the shared data library, `architecture` provides BitNet components:

```
data_handler (library)
    │
    └──► training (wrinklefree)
            Uses: data_handler.data, data_handler.influence

architecture (library)
    │
    └──► training (wrinklefree)
            Uses: bitnet_arch.layers (BitLinear, BitLinearLRC), bitnet_arch.conversion
```

**Adding workspace dependencies**:
```toml
# In pyproject.toml
[project]
dependencies = ["data-handler", "bitnet-arch"]

[tool.uv.sources]
data-handler = { workspace = true }
bitnet-arch = { workspace = true }
```

**Important**: Changes to data_handler or architecture affect training - test both after modifications.

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

### AI Agent Sync Protocol

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

## SSH Hosts

Desktop IP: `192.168.1.217` (configured in `~/.ssh/config`)

## RunPod Setup

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

## Troubleshooting

### Package not found
```bash
uv sync --all-packages --reinstall
```

### Import errors between packages
Ensure workspace sources are configured:
```toml
[tool.uv.sources]
data-handler = { workspace = true }
bitnet-arch = { workspace = true }
```

### Submodule issues
```bash
git submodule update --init --recursive
```

## Documentation Links

- [Quick Start](../quick-start.md) - Installation and first steps
- [Architecture](../architecture.md) - System design and package relationships
- [Dependencies](../dependencies.md) - Dependency graph and version constraints
- [Development](../development.md) - Contributing and CI/CD
