#!/bin/bash
# Start Streamlit chat UI
cd /opt/wrinklefree
export PATH="$HOME/.local/bin:$PATH"
export BITNET_BACKEND=native

# Use venv with all deps
source /opt/wrinklefree/.venv/bin/activate

exec streamlit run packages/inference/demo/serve_sglang.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
