# WrinkleFree Monorepo

Monorepo for 1.58-bit quantized LLM research using uv workspaces.

## AI Navigation Guide

**Start here to understand the codebase:**

| Question | Where to Look |
|----------|---------------|
| How does 1.58-bit training work? | `packages/training/CLAUDE.md` - Full pipeline explanation |
| How does distillation work? | `packages/distillation/CLAUDE.md` - BitDistill, TCS loss |
| What are BitLinear/SubLN? | `packages/architecture/CLAUDE.md` - Layer implementations |
| How is data loaded/mixed? | `packages/data_handler/CLAUDE.md` - Streaming, packing, influence |
| How to deploy to cloud? | `packages/deployer/CLAUDE.md` - SkyPilot/Modal setup |
| How to serve models? | `packages/inference/CLAUDE.md` - SGLang, BitNet.cpp, Rust gateway |
| How to evaluate models? | `packages/eval/CLAUDE.md` - lm-eval harness, benchmarks |
| How to convert to DLM? | `packages/converter/CLAUDE.md` - Fast-dLLM v2 training |
| System architecture? | `docs/architecture.md` - Package relationships |
| Package dependencies? | `docs/dependencies.md` - Dependency graph |

**Each package has its own CLAUDE.md** with detailed guidance for that specific area.

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
Stage 3: Knowledge Distillation (packages/distillation)
    │   BitDistill or TCS loss with teacher guidance
    ▼
Export: Convert to GGUF/DLM (packages/converter)
    │
    ▼
Serve: Inference with BitNet.cpp (packages/inference)
```

**Key insight**: Stages 1-2 are in `training`, Stage 3+ is in `distillation`.

## Package Map

| Package | Type | Purpose | Key Entry Point |
|---------|------|---------|-----------------|
| `packages/training` | App | 1.58-bit training (Stages 1, 1.9, 2) | `scripts/train.py` |
| `packages/distillation` | App | Knowledge distillation (Stage 3+) | `scripts/distill.py` |
| `packages/architecture` | Lib | BitNet layers & model conversion | Import as library |
| `packages/data_handler` | Lib | Data loading & influence functions | Import as library |
| `packages/inference` | App | Model serving (sglang-bitnet) | `demo/serve_sglang.py` |
| `packages/eval` | App | Model evaluation (lm-eval) | `scripts/evaluate.py` |
| `packages/deployer` | App | Cloud deployment (SkyPilot/Modal) | `wf` CLI |
| `packages/converter` | App | DLM format conversion | `scripts/train_dlm.py` |

## Common Tasks Quick Reference

| Task | Package | Command |
|------|---------|---------|
| Train Stage 2 | training | `uv run --package wrinklefree python scripts/train.py model=smollm2_135m training=stage2_pretrain` |
| Run distillation | distillation | `uv run --package wrinklefree-distillation python scripts/distill.py student.checkpoint_path=...` |
| Deploy to cloud | deployer | `wf train -m smollm2_135m -s 2 --cloud nebius` |
| Run evaluation | eval | `uv run --package wrinklefree_eval python scripts/evaluate.py --model-path ...` |
| Start inference | inference | `uv run streamlit run demo/serve_sglang.py` |

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

`data_handler` is the shared data library, `architecture` provides BitNet components:

```
data_handler (library)
    │
    ├──► training (wrinklefree)
    │       Uses: data_handler.data, data_handler.influence
    │
    └──► distillation (wrinklefree-distillation)
            Uses: data_handler.data, data_handler.influence

architecture (library)
    │
    └──► training (wrinklefree)
            Uses: bitnet_arch.layers, bitnet_arch.conversion
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

**Important**: Changes to data_handler affect training and distillation - test both after modifications.

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
data-handler = { workspace = true }
bitnet-arch = { workspace = true }
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
