#!/bin/bash
# Common setup script for SkyPilot benchmark/inference jobs
# Source this from your SkyPilot YAML setup section:
#   source /opt/inference/scripts/skypilot_common_setup.sh
#
# Provides:
# - System deps (clang, cmake, git, curl)
# - uv package manager
# - Python packages (huggingface_hub, torch, safetensors, etc.)
# - CPU info logging

set -ex

echo "============================================"
echo "WrinkleFree Common Setup"
echo "============================================"

# === System Info ===
lscpu | grep -E "(Model name|CPU\(s\)|Thread|Core|Socket|Cache)" | head -15 || true
cat /proc/cpuinfo | grep -m1 "flags" | tr ' ' '\n' | grep -E "^avx" | head -10 || true
free -h

# === System Dependencies ===
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y clang cmake git curl python3-pip python3-venv ccache
elif command -v yum &> /dev/null; then
    sudo yum install -y clang cmake git curl python3-pip ccache
fi

# === Install uv ===
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# === Python Packages ===
# Install to current python env (SkyPilot uses conda python by default)
# Using pip directly for conda compatibility
pip install \
    "huggingface_hub[cli]" \
    hf_transfer \
    safetensors \
    torch \
    numpy \
    transformers

echo "Python packages installed"

# Verify
python3 -c "import huggingface_hub; print(f'huggingface_hub {huggingface_hub.__version__}')"
python3 -c "import torch; print(f'torch {torch.__version__}')"

# === Create Common Directories ===
sudo mkdir -p /results /models
sudo chown $(whoami) /results /models 2>/dev/null || true

echo "=== Common Setup Complete ==="
