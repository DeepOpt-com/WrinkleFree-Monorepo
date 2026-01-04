# Archived Demo Files

These files are archived for reference. Use `demo/serve_sglang.py` for production serving.

## Archived Files

| File | Reason |
|------|--------|
| `serve_bitnet_cpp.py` | Uses BitNet.cpp backend (reference only) |
| `serve_streamlit_api.py` | Uses BitNet.cpp sync API (deprecated) |
| `serve_streamlit.py` | Direct transformers loading (OOM on low-memory systems) |
| `serve_bitnet.py` | Transformers-based server (OOM issues) |
| `serve_bitnet_2b.py` | Earlier version of BitNet server |
| `serve_native.py` | Direct native kernel testing |

## Current Recommendation

Use `demo/serve_sglang.py` which connects to SGLang-BitNet server:

```bash
# Start SGLang server
./scripts/launch_sglang_bitnet.sh

# Start Streamlit frontend
uv run streamlit run demo/serve_sglang.py --server.port 7860
```
