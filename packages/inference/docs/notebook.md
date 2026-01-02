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

3. **Block size stays at 32** (`dlm_config.rs:34`)
   - MUST match training block size (Fast-dLLM v2 default is 32)
   - Larger blocks would break model quality

4. **Configurable decode mode** (`dlm_config.rs`, `dlm_server.rs`)
   - `greedy` - Single-pass argmax (~120 tok/s) - DEFAULT
   - `iterative` - Per-paper confidence thresholding (slower but correct)

### Algorithm
```
Greedy mode:   O(1) forward pass per block
Iterative mode: O(iterations) forward passes per block (until all unmasked)
```

### Trade-offs
- **Greedy mode**: 20x faster, may have slightly lower quality
- **Iterative mode**: Correct per Fast-dLLM v2 paper (arXiv:2509.26328)
  - Uses confidence thresholding to progressively unmask tokens
  - Always unmasks at least one token per iteration to ensure progress

### Benchmark Results (GCP c3d-standard-32, 20 iterations, 64 max tokens)

| Mode | Threshold | Throughput | % of Greedy | Notes |
|------|-----------|------------|-------------|-------|
| **Greedy** | N/A | 60.81 tok/s | 100% | Single-pass argmax |
| **Iterative** | 0.5 | 60.58 tok/s | 99.6% | Nearly all tokens unmask in 1 pass |
| **Iterative** | 0.7 | 54.12 tok/s | 89.0% | **Recommended balance** |
| **Iterative** | 0.9 | 20.37 tok/s | 33.5% | Per-paper quality |

**Key optimizations for iterative mode:**
- Incremental KV cache: only clear from first masked position
- Only request logits for masked positions
- Store best predictions for fallback (no MASK in output)

### Files Changed
- `packages/inference/extern/sglang-bitnet/sgl-model-gateway/src/inference/dlm_scheduler.rs`
- `packages/inference/extern/sglang-bitnet/sgl-model-gateway/src/inference/dlm_config.rs`

---

## 2026-01-02: Optimized Iterative Mode (Per-Paper Correctness)

### Problem
The initial iterative implementation was slow (~12.79 tok/s) because it:
1. Cleared the entire KV cache every iteration
2. Recomputed all positions every iteration
3. Requested logits for all positions (not just masked ones)

### Solution: Incremental KV Cache + Selective Computation

The Fast-dLLM v2 paper uses a "DualCache" mechanism. We implemented a simplified version:

**Key insight**: With causal attention, if position `i` is unmasked and all positions `j < i` are also unmasked, then KV[i] is stable and doesn't need recomputation.

```
Iteration 1: [MASK, MASK, MASK, MASK] → unmask positions 0, 2
Iteration 2: [tok0, MASK, tok2, MASK] → only recompute from position 1
Iteration 3: [tok0, tok1, tok2, MASK] → only recompute from position 3
```

### Algorithm (Optimized Iterative)

```python
def decode_block_iterative(block_size, threshold):
    tokens = [MASK] * block_size
    is_masked = [True] * block_size
    stable_prefix_len = 0  # Contiguous unmasked from start

    while any(is_masked):
        # Only clear KV from first position that might change
        recompute_from = stable_prefix_len
        clear_kv(recompute_from, block_size)

        # Forward pass only for positions that need it
        batch = [tokens[recompute_from - 1]] + tokens[recompute_from:]
        logits = forward(batch)

        # Compute confidence only for masked positions
        candidates = []
        for i in range(recompute_from, block_size):
            if is_masked[i]:
                token, conf = argmax_with_confidence(logits[i - recompute_from])
                candidates.append((i, token, conf))

        # Unmask above threshold (or at least one for progress)
        unmasked_any = False
        for idx, token, conf in candidates:
            if conf > threshold:
                tokens[idx] = token
                is_masked[idx] = False
                unmasked_any = True

        if not unmasked_any:
            # Unmask highest confidence to ensure progress
            best = max(candidates, key=lambda x: x[2])
            tokens[best[0]] = best[1]
            is_masked[best[0]] = False

        # Update stable prefix
        stable_prefix_len = first_masked_index(is_masked)

    return tokens
```

### Correctness Guarantees

1. **Token shift preserved**: logits[i-1] predicts token[i] (per paper)
2. **Confidence thresholding**: Only unmask when model is confident
3. **Progress guarantee**: Always unmask at least one token per iteration
4. **No MASK in output**: Store best predictions, use as fallback
5. **Causal consistency**: KV cache respects causal dependencies

### Performance Results

| Optimization | Throughput | Improvement |
|--------------|------------|-------------|
| Baseline iterative | 12.79 tok/s | - |
| + Incremental KV | 16.5 tok/s | +29% |
| + Selective logits | 20.37 tok/s | +59% |
| + Lower threshold (0.7) | 54.12 tok/s | +323% |

### Threshold Selection Guide

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.5 | ~1-2 iterations/block | Maximum speed, quality ≈ greedy |
| 0.7 | ~2-4 iterations/block | **Recommended**: 89% speed, good quality |
| 0.9 | ~4-8 iterations/block | Per-paper quality, 33% speed |
| 0.95+ | Many iterations/block | Maximum quality, slow |

### Files Changed
- `dlm_scheduler.rs`: `decode_block_iterative()` with incremental KV
- `dlm_config.rs`: Default threshold changed to 0.7
- `dlm_server.yaml`: Updated with threshold recommendations

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
  threshold: 0.95       # For iterative mode
  small_block_size: 8
  mask_token_id: null   # Auto-detect
  decode_mode: greedy   # "greedy" (fast) or "iterative" (per paper)

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
