#!/bin/bash
# Download SmolLM2-135M for testing (small, fast-loading model)
#
# Usage:
#   ./scripts/download_test_model.sh
#   ./scripts/download_test_model.sh ./custom/path

set -e

MODEL_DIR="${1:-./models/test}"
MODEL_NAME="smollm2-135m.gguf"

echo "Downloading SmolLM2-135M test model to $MODEL_DIR..."

mkdir -p "$MODEL_DIR"

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Download the GGUF model
# Note: You may need to use a different repo or convert from safetensors
# This is a placeholder - update with actual model location
echo "Downloading from HuggingFace..."

# Option 1: Direct download if GGUF available
# huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct-GGUF \
#     smollm2-135m-instruct-q4_k_m.gguf \
#     --local-dir "$MODEL_DIR"

# Option 2: Download safetensors and convert
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct \
    --local-dir "$MODEL_DIR/source"

echo ""
echo "Model downloaded to $MODEL_DIR"
echo ""
echo "If you need to convert to GGUF format:"
echo "  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
echo "  2. Convert: python llama.cpp/convert_hf_to_gguf.py $MODEL_DIR/source"
echo ""
echo "Or use a pre-quantized model from WrinkleFree-1.58Quant:"
echo "  cp ../WrinkleFree-1.58Quant/outputs/model.gguf $MODEL_DIR/$MODEL_NAME"
