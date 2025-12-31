#!/bin/bash
# Start Streamlit chat UI
cd /opt/wrinklefree
export PATH="$HOME/.local/bin:$PATH"
export BITNET_BACKEND=native
exec uv run --package wrinklefree-inference streamlit run packages/inference/demo/serve_sglang.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
