#!/bin/bash
set -e

echo "Starting vLLM inference server..."
echo "Model: ${MODEL_PATH}"
echo "Host: ${HOST}:${PORT}"
echo "Max model length: ${MAX_MODEL_LEN}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"

# Run vLLM OpenAI-compatible server
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8080}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    "$@"
