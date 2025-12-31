#!/bin/bash
# Start BitNet native inference server (uses sgl-kernel SIMD kernels for ~25 tok/s)
cd /opt/wrinklefree
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Disable torch.compile to avoid C++ compilation issues
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Use venv with sgl-kernel (CRITICAL: must use this venv for native kernels)
source /opt/wrinklefree/.venv/bin/activate

# Use pre-converted .bin model (packed format for native kernels)
MODEL_BIN="${BITNET_MODEL:-/opt/wrinklefree/models/bitnet-b1.58-2B-4T.bin}"
TOKENIZER_DIR="${BITNET_TOKENIZER:-/opt/wrinklefree/models/bitnet-b1.58-2B-4T}"

# Convert model if .bin doesn't exist
if [ ! -f "$MODEL_BIN" ]; then
    echo "Converting model to native format..."
    cd /opt/wrinklefree/packages/inference
    python scripts/convert_to_sglkernel.py "$TOKENIZER_DIR" "$MODEL_BIN"
fi

exec python packages/inference/scripts/serve_bitnet_native.py \
    --model "$MODEL_BIN" \
    --tokenizer "$TOKENIZER_DIR" \
    --port 30000
