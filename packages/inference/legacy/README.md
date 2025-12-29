# Legacy Components

Archived code from earlier development stages. The primary serving stack is now **SGLang-BitNet + Streamlit**.

## Why Archived

These components were part of earlier approaches that have been superseded:
- **BitNet.cpp integration** - Replaced by SGLang-BitNet with native SIMD kernels
- **Custom CLI** - Simplified to shell scripts
- **Custom model implementations** - Using transformers native BitNet support
- **GGUF conversion** - Not needed with HuggingFace direct loading

## Archived Directories

| Directory | Contents |
|-----------|----------|
| `src/` | Legacy Python modules (server, converter, ui, models, native, cli) |
| `scripts/` | BitNet.cpp build/conversion scripts |
| `benchmark/` | Experimental benchmarking code |
| `demo/` | Alternative server implementations |
| `modal/` | Modal cloud deployment experiments |
| `docs/` | BitNet.cpp focused documentation |

## Current Primary Stack

Use the main project's primary serving stack instead:

```bash
# Start SGLang server
./scripts/launch_sglang_bitnet.sh

# Start Streamlit frontend
uv run streamlit run demo/serve_sglang.py --server.port 7860
```

## Reusing Legacy Code

If you need to use any legacy components, you can import them directly:

```python
# Example: Use legacy converter
import sys
sys.path.insert(0, "legacy/src")
from converter.hf_to_gguf import HFToGGUFConverter
```

Note: Legacy code may have broken imports and dependencies.
