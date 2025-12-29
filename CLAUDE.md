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

- `cheapertraining` is imported by: training, fairy2
- Use `workspace = true` in `[tool.uv.sources]` for inter-package deps

## GCP Configuration

- **Project ID**: `wrinklefree-481904`

## Remote Sync

```bash
# Sync to Desktop
./sync.sh --preset desktop --no-watch

# Sync to RunPod
./sync.sh --preset runpod --no-watch
```

**Presets** (in `.sync.conf`): `desktop`, `runpod`, `RTX6000`

## Inference Engine Quick Start

```bash
# On Desktop: Start Streamlit chat interface
ssh Desktop 'cd /home/lev/code/WrinkleFree/packages/inference && \
  uv run streamlit run demo/serve_sglang.py --server.port 7860'

# Access at http://192.168.1.217:7860
```

## SSH Hosts

Desktop IP: `192.168.1.217` (configured in `~/.ssh/config`)

## Core Principles

- FAIL LOUDLY INSTEAD OF FALLBACKS
- DO NOT LAUNCH GPU INSTANCES ON GCP - use Nebius and RunPod
- Each package has its own CLAUDE.md with package-specific guidance
