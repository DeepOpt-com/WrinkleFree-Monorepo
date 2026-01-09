# C++ Inference Wrappers

C++ wrapper code for llama.cpp integration, used by `dlm_server` (llama-inference feature).

## Files

| File | Purpose |
|------|---------|
| `llama_engine.cpp` | llama.cpp FFI wrapper for Rust |
| `kv_cache.cpp` | KV cache management |
| `bitnet_batch.cpp` | Batch inference scheduling |
| `*.h` | Header files |

## Usage

These files are compiled automatically by `rust/build.rs` when building with `--features llama-inference`.

Requires llama.cpp to be built first:
```bash
./scripts/setup_llama_cpp.sh
```
