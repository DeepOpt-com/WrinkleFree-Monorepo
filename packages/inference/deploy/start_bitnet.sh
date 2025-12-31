#!/bin/bash
# Start BitNet Flask inference server (uses transformers directly)
cd /opt/wrinklefree
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Disable torch.compile to avoid C++ compilation issues
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Use local model path (downloaded via huggingface-cli)
MODEL="${BITNET_MODEL:-/opt/wrinklefree/models/bitnet-b1.58-2B-4T}"

exec uv run --package wrinklefree-inference python packages/inference/scripts/serve_bitnet_flask.py \
    --model-path "$MODEL" \
    --port 30000
