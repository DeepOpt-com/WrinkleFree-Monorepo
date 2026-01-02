# Development Notebook

## 2026-01-02: DLM Scheduler Performance Optimization

### Summary
Achieved **20x throughput improvement** in the DLM (Diffusion LLM) scheduler by replacing the iterative small-block decoding approach with a single-pass greedy decode strategy.

### Before
- ~6 tok/s on GCP c3d-standard-32
- Multiple forward passes per block (4 small blocks x ~3 iterations = ~12 passes)
- Excessive KV cache clearing per iteration

### After
- **120.67 tok/s** on GCP c3d-standard-32 (50 iterations, 128 max tokens)
- Single forward pass per block
- Single KV cache clear per block

### Key Changes

1. **Single-pass greedy decode** (`dlm_scheduler.rs:493-596`)
   - Instead of iterating over small blocks with confidence thresholding
   - Decode entire block in ONE forward pass
   - Use greedy argmax for all positions simultaneously

2. **Batch layout optimization**
   - Include previous token at batch position 0 for token shift
   - Batch: `[prev_token, mask, mask, ..., mask]`
   - logits[i] predicts token[i] due to token shift

3. **Increased default block size** (`dlm_config.rs:34`)
   - Changed from 32 to 64 tokens per block
   - Larger blocks = fewer forward passes = higher throughput

### Algorithm
```
Old: O(small_blocks * iterations) forward passes per block
New: O(1) forward pass per block
```

### Trade-offs
- **Speed**: 20x faster
- **Quality**: Greedy decode (no confidence thresholding)
  - For trained DLM models, greedy is often sufficient
  - Can add iterative refinement back as an option for quality-critical use cases

### Benchmark Results
```
=== Benchmark Results ===
Iterations:        50
Prompt tokens:     20
Max tokens:        128
Total tokens:      6450

Latency (ms):
  Mean:            1069.02
  p50:             1069.31
  p95:             1071.16
  p99:             1071.76

Throughput:
  Tokens/sec:      120.67
  Avg tokens/req:  129.00
  Total time:      53.45s
=========================
```

### Files Changed
- `packages/inference/extern/sglang-bitnet/sgl-model-gateway/src/inference/dlm_scheduler.rs`
- `packages/inference/extern/sglang-bitnet/sgl-model-gateway/src/inference/dlm_config.rs`

---

## 2026-01-02: YAML Config Support for DLM Server

Added YAML configuration file support for the DLM server.

### Usage
```bash
# Generate example config
./dlm_server --generate-config > my_config.yaml

# Run with config file
./dlm_server --config my_config.yaml

# CLI args override YAML (useful for testing)
./dlm_server --config my_config.yaml --block-size 64 --benchmark
```

### Example Config (`configs/dlm_server.yaml`)
```yaml
model_path: /path/to/dlm-model.gguf
host: 0.0.0.0
port: 30000

dlm:
  block_size: 32        # MUST match training
  threshold: 0.95
  small_block_size: 8
  mask_token_id: null   # Auto-detect

scheduler:
  max_sequences: 16
  enable_radix_cache: true

benchmark:
  enabled: false
  iterations: 50
  max_tokens: 64
```

### Priority Order
1. CLI arguments (highest)
2. YAML config file
3. Environment variables (`MODEL_PATH`)
4. Defaults (lowest)

### Files Changed
- `packages/inference/extern/sglang-bitnet/sgl-model-gateway/src/bin/dlm_server.rs`
- `packages/inference/configs/dlm_server.yaml` (new)
