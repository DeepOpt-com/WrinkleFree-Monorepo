# Research Notes

This directory contains **archived** historical research notes and optimization analysis
from Dec 2024 - Dec 2025.

**Important**: The sglang-bitnet submodule has been removed from the codebase.
All paths and scripts referencing `extern/sglang-bitnet/` no longer exist.

## Current Inference Path

```bash
# Build wf_server (Pure Rust, ~26 tok/s)
cd rust && cargo build --release --bin wf_server --features native-inference
./target/release/wf_server --model-path ../models/model.gguf --port 30000

# Or dlm_server for DLM models (~60 tok/s)
cargo build --release --bin dlm_server --features llama-inference
```

See [CLAUDE.md](../../CLAUDE.md) for full instructions.

## Contents

| File | Description | Status |
|------|-------------|--------|
| `sglang_optimization_plan.md` | Dec 2025 performance optimization analysis | Archived |
| `notebook.md` | Development notes and experiments | Archived |
