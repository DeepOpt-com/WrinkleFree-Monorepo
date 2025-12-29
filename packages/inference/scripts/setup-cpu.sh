#!/bin/bash
# One-time CPU setup for sglang-bitnet inference
#
# This script sets up the inference engine for CPU-only operation.
# Run from the inference package directory:
#   ./scripts/setup-cpu.sh
#
# After setup, start the server with:
#   ./scripts/serve.sh --backend sglang
#
# Or run individual components:
#   ./scripts/launch_sglang_bitnet.sh  # Server only
#   uv run streamlit run demo/serve_sglang.py  # UI only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_DIR}/../../.venv"
PIP="${VENV_DIR}/bin/pip"

cd "$PROJECT_DIR"

echo "=== WrinkleFree Inference CPU Setup ==="
echo "Project: $PROJECT_DIR"
echo "Venv: $VENV_DIR"
echo ""

# Build throttling
export CMAKE_BUILD_PARALLEL_LEVEL=4
export MAKEFLAGS="-j4"

# Step 1: Initialize submodules
echo "[1/6] Initializing submodules..."
git submodule update --init extern/sglang-bitnet || true

# Step 2: Install CPU-only PyTorch
echo "[2/6] Installing CPU-only PyTorch..."
$PIP install --force-reinstall torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Step 3: Install vllm CPU stub
echo "[3/6] Installing vllm-cpu-stub..."
$PIP install -e extern/vllm-cpu-stub

# Step 4: Install sglang (without GPU deps)
echo "[4/6] Installing sglang from fork (CPU-only)..."
$PIP install -e extern/sglang-bitnet/python --no-deps

# Install CPU-compatible sglang dependencies
$PIP install \
    aiohttp anthropic fastapi uvicorn \
    huggingface_hub hf_transfer gguf \
    einops numpy scipy pillow requests \
    pydantic orjson msgspec packaging \
    transformers accelerate sentencepiece \
    tiktoken tokenizers safetensors \
    psutil pyzmq grpcio \
    pybase64 partial_json_parser openai==2.6.1 IPython \
    compressed-tensors xgrammar==0.1.27

# Step 5: Build sgl-kernel (CPU-only)
echo "[5/6] Building sgl-kernel with BitNet kernels..."
cd extern/sglang-bitnet/sgl-kernel
cp pyproject_cpu.toml pyproject.toml
$PIP install scikit-build-core cmake ninja pybind11
taskset -c 0-7 $PIP install -e . --no-build-isolation 2>&1 || \
    taskset -c 0-7 $PIP install . --no-build-isolation
cd "$PROJECT_DIR"

# Step 6: Copy .so for editable install
echo "[6/6] Copying kernel library..."
cp ${VENV_DIR}/lib/python3.*/site-packages/sgl_kernel/common_ops.*.so \
   extern/sglang-bitnet/sgl-kernel/python/sgl_kernel/

# Verify
echo ""
echo "=== Verifying Installation ==="
${VENV_DIR}/bin/python -c "
from sgl_kernel.quantization import bitnet_check_kernel_available
print(f'BitNet kernel available: {bitnet_check_kernel_available()}')
import sglang
print(f'sglang version: {sglang.__version__}')
import vllm
print(f'vllm-cpu-stub: OK')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Start the server with:"
echo "  ./scripts/serve.sh --backend sglang"
echo ""
echo "Or run components individually:"
echo "  ./scripts/launch_sglang_bitnet.sh   # Server on port 30000"
echo "  uv run streamlit run demo/serve_sglang.py --server.port 7860"
