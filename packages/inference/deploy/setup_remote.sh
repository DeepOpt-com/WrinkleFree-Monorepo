#!/bin/bash
# Remote setup script for BitNet inference on Linode
# This script is executed ON the Linode instance

set -euo pipefail

echo "=== BitNet Inference Remote Setup ==="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

INSTALL_DIR="${INSTALL_DIR:-/opt/wrinklefree}"

# 1. System dependencies
log "Installing system dependencies..."
apt-get update
apt-get install -y \
    build-essential \
    curl \
    git \
    cmake \
    pkg-config \
    libssl-dev \
    python3-dev \
    ufw

# 2. Install uv (Python package manager)
log "Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 3. Install Node.js and pm2
log "Installing Node.js and pm2..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
node --version

if ! command -v pm2 &> /dev/null; then
    npm install -g pm2
fi
pm2 --version

# 4. Create log directory
log "Setting up log directory..."
mkdir -p /var/log/pm2
chmod 755 /var/log/pm2

# 5. Sync Python dependencies
log "Installing Python dependencies..."
cd "$INSTALL_DIR/packages/inference"
uv sync --frozen 2>/dev/null || uv sync

# 6. Create venv and build sgl-kernel with BitNet SIMD kernels
log "Setting up Python venv for native kernels..."
cd "$INSTALL_DIR"
if [ ! -d ".venv" ]; then
    uv venv .venv
fi
source .venv/bin/activate

# CRITICAL: Install CPU-only PyTorch to avoid CMake CUDA errors
# The default torch wheel has CMake files that look for CUDA, causing build failures
# Force-reinstall ensures we get the pure CPU version
log "Installing CPU-only PyTorch (required for sgl-kernel build)..."
uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu

log "Building sgl-kernel with BitNet SIMD kernels..."
cd "$INSTALL_DIR/packages/inference/extern/sglang-bitnet/sgl-kernel"

# Use CPU-only config
if [ -f "pyproject_cpu.toml" ]; then
    cp pyproject_cpu.toml pyproject.toml
fi

# Install build deps and build kernel
export CMAKE_BUILD_PARALLEL_LEVEL=4
export MAKEFLAGS="-j4"
uv pip install scikit-build-core cmake ninja pybind11
uv pip install -e . --no-build-isolation 2>&1 || \
    uv pip install . --no-build-isolation

# Copy .so from site-packages to source dir (required for editable install)
log "Installing kernel library..."
cp $INSTALL_DIR/.venv/lib/python3.*/site-packages/sgl_kernel/common_ops.*.so \
   $INSTALL_DIR/packages/inference/extern/sglang-bitnet/sgl-kernel/python/sgl_kernel/

# Install vllm-cpu-stub (required for sglang)
log "Installing vllm-cpu-stub..."
cd "$INSTALL_DIR/packages/inference/extern/vllm-cpu-stub"
uv pip install -e .

# Install remaining inference deps
log "Installing inference dependencies..."
uv pip install transformers accelerate safetensors huggingface_hub flask streamlit

# Verify kernel installation
log "Verifying BitNet kernel..."
python -c "
from sgl_kernel.quantization import bitnet_check_kernel_available
avail = bitnet_check_kernel_available()
print(f'BitNet SIMD kernel available: {avail}')
if not avail:
    print('WARNING: Native kernel not available, will use slow fallback')
    exit(1)
" || warn "Native kernel build may have failed, will use fallback"

# 7. Download model if not present
MODEL_DIR="$INSTALL_DIR/models/bitnet-b1.58-2B-4T"
if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/config.json" ]; then
    log "Downloading BitNet model from HuggingFace..."
    mkdir -p "$INSTALL_DIR/models"
    cd "$INSTALL_DIR/models"
    uv run huggingface-cli download microsoft/bitnet-b1.58-2B-4T --local-dir bitnet-b1.58-2B-4T
    log "Model downloaded to $MODEL_DIR"
fi

# 8. Convert model to packed .bin format for native server
MODEL_BIN="$INSTALL_DIR/models/bitnet-b1.58-2B-4T.bin"
if [ ! -f "$MODEL_BIN" ]; then
    log "Converting model to native format (~1-2 min)..."
    cd "$INSTALL_DIR/packages/inference"
    uv run python scripts/convert_to_sglkernel.py "$MODEL_DIR" "$MODEL_BIN"
    log "Model converted to $MODEL_BIN"
fi

# 9. Configure UFW firewall
log "Configuring firewall..."
ufw --force enable
ufw allow 22/tcp comment 'SSH'
ufw allow 7860/tcp comment 'Streamlit UI'
ufw allow 30000/tcp comment 'BitNet API'
ufw status

# 10. Start pm2 services
log "Starting pm2 services..."
cd "$INSTALL_DIR/packages/inference"
pm2 start deploy/ecosystem.config.js

# 11. Setup pm2 to start on boot
log "Configuring pm2 startup..."
pm2 startup systemd -u root --hp /root
pm2 save

# 12. Verify services
log "Verifying services..."
sleep 5
pm2 list

# Check if services are running
if pm2 show bitnet-native &>/dev/null && pm2 show streamlit-ui &>/dev/null; then
    log "All services started successfully!"
else
    err "Some services failed to start. Check: pm2 logs"
fi

# Print access info
PUBLIC_IP=$(curl -s ifconfig.me)
echo ""
echo "=== Deployment Complete ==="
echo -e "${GREEN}Streamlit UI:${NC} http://${PUBLIC_IP}:7860"
echo -e "${GREEN}API Endpoint:${NC} http://${PUBLIC_IP}:30000/v1/chat/completions"
echo ""
echo "Useful commands:"
echo "  pm2 status       - Check service status"
echo "  pm2 logs         - View all logs"
echo "  pm2 logs bitnet-native - View inference server logs"
echo "  pm2 restart all  - Restart all services"
