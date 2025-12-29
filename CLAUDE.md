# WrinkleFree Project Instructions

## Overview

WrinkleFree is a monorepo for 1.58-bit quantized LLM research. It uses `meta` for managing multiple git repositories.

## Subprojects

| Project | Purpose |
|---------|---------|
| `WrinkleFree-1.58Quant` | Training pipeline for 1.58-bit quantization (Stage 1-3) |
| `WrinkleFree-CheaperTraining` | Memory-efficient training experiments |
| `WrinkleFree-Deployer` | Cloud deployment configs (SkyPilot, Modal) |
| `WrinkleFree-Inference-Engine` | Serving layer for 1.58-bit models (BitNet.cpp, sglang-bitnet) |
| `WrinkleFree-Eval` | Evaluation harness for quantized models |
| `WrinkleFree-DLM-Converter` | Model format conversion utilities |
| `extern/BitNet` | Microsoft BitNet.cpp submodule |

## GCP Configuration

- **Project ID**: `wrinklefree-481904`

## Remote Sync

Use `./sync.sh --preset <name>` to sync to remote machines:

```bash
# Sync to Desktop (local LAN)
./sync.sh --preset desktop --no-watch

# Sync to RunPod
./sync.sh --preset runpod --no-watch

# Watch mode (continuous sync)
./sync.sh --preset desktop
```

**Presets** (defined in `.sync.conf`):
- `desktop` - Local Desktop machine at `/home/lev/code/WrinkleFree`
- `runpod` - RunPod instance
- `RTX6000` - RTX 6000 GPU server

**Note**: For large submodules like `extern/BitNet`, use targeted rsync instead:
```bash
# Sync only specific files (faster)
rsync -avz --no-owner --no-group /local/path/ Desktop:/remote/path/
```

## Inference Engine Quick Start

```bash
# On Desktop: Start Streamlit chat interface
ssh Desktop 'cat > /tmp/serve.sh << "SCRIPT"
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine
tmux kill-session -t streamlit 2>/dev/null || true
tmux new-session -d -s streamlit "uv run streamlit run demo/serve_sglang.py --server.port 7860 --server.address 0.0.0.0"
SCRIPT
chmod +x /tmp/serve.sh && /tmp/serve.sh'

# Access at http://192.168.1.217:7860 (Desktop LAN IP)
```

## SSH Hosts

Desktop is configured in `~/.ssh/config`. Current IP: `192.168.1.217`

## Other Notes
- FAIL LOUDLY INSTEAD OF FALLBACKS FOR MOST CODE
- DO NOT LAUNCH GPU INSTANCES ON GCP! Just Nebius and RunPod for now.

