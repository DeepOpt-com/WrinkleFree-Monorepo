# GGUF Conversion Guide

This guide covers converting DLM/BitNet checkpoints to GGUF format for inference with llama.cpp.

## Quick Start

```bash
# Download checkpoint from GCS (skip optimizer state)
mkdir -p models/dlm-2b
gcloud storage cp \
    'gs://wrinklefree-checkpoints/dlm/checkpoint/*.json' \
    'gs://wrinklefree-checkpoints/dlm/checkpoint/*.safetensors' \
    models/dlm-2b/

# Convert with validation
python packages/inference/scripts/convert_checkpoint_to_gguf.py \
    models/dlm-2b \
    --outfile models/dlm-2b.gguf \
    --validate

# Serve with vanilla llama.cpp
llama-server -m models/dlm-2b.gguf --port 30000
```

## Quantization Format Comparison

| Format | Size (2B) | Vanilla llama.cpp | Notes |
|--------|-----------|-------------------|-------|
| **i2_s** | ~1.1GB | Yes | **RECOMMENDED** - 2-bit integer, multiply-add |
| tq2_0 | ~1.2GB | Yes | Alternative ternary format |
| tl1 | ~1.1GB | No* | LUT-based, requires kernel config |
| tl2 | ~1.1GB | No* | LUT-based, AVX512 optimized |
| tq1_0 | ~1.1GB | **No** | Type 36 conflict - use BitNet.cpp fork |
| f16 | ~4.5GB | Yes | **DO NOT USE** - 4x larger, slower |

*TL1/TL2 require pre-generated kernel config files matching your model dimensions.

## The Problem: Why Conversion Matters

DLM checkpoints store weights in one of two formats:

### 1. Online Quantization (bf16 weights)
- Weights are continuous float values
- Quantization happens at runtime: `round(w / mean_abs(w)).clip(-1, 1)`
- Result: ~30% zeros, ~35% each +1/-1

### 2. Offline Quantization (packed 2-bit)
- Weights are pre-quantized and packed (4 values per byte)
- Separate `weight_scale` tensors store the scaling factors
- Shape appears as `[N/4, K]` instead of `[N, K]`

**The standard llama.cpp converter does NOT handle either case correctly!**

## Common Errors and Solutions

### Error: Shape Mismatch
```
expected 2560, 2560, got 660, 2560, 1, 1
```
**Cause**: Packed 2-bit weights not unpacked.
**Solution**: Use `convert_checkpoint_to_gguf.py` which handles unpacking.

### Error: Gibberish Output
```
Output: "GGGGG..." or random characters
```
**Cause**:
1. Used `np.sign()` instead of proper quantization (produces 0% zeros instead of ~30%)
2. Used post-hoc `llama-quantize` on already-ternary weights

**Solution**: Use the fixed converter in `extern/BitNet/utils/convert-hf-to-gguf-bitnet.py`

### Error: Tokenizer Not Found
```
FileNotFoundError: tokenizer.model
```
**Cause**: Model uses GPT2/BPE tokenizer, not sentencepiece.
**Solution**: The converter now supports both tokenizer types automatically.

### Error: Architecture Not Found
```
KeyError: 'BitNetForCausalLM'
```
**Cause**: Training uses `BitNetForCausalLM`, llama.cpp expects `BitnetForCausalLM`.
**Solution**: The converter fixes this automatically, or run:
```bash
sed -i 's/BitNetForCausalLM/BitnetForCausalLM/g' checkpoint/config.json
```

### Error: Type 36 Unknown
```
Unknown GGUF type 36
```
**Cause**: TQ1_0 format (type 36) conflicts with IQ4_NL_4_4 in vanilla llama.cpp.
**Solution**: Use I2_S format instead, or use Microsoft's BitNet.cpp fork.

## Conversion Workflow Details

### Step 1: Download Checkpoint
```bash
# Skip optimizer state to save bandwidth
gcloud storage cp \
    'gs://wrinklefree-checkpoints/dlm/checkpoint/*.json' \
    'gs://wrinklefree-checkpoints/dlm/checkpoint/*.safetensors' \
    'gs://wrinklefree-checkpoints/dlm/checkpoint/*.jinja' \
    models/my-checkpoint/
```

### Step 2: Convert to GGUF
```bash
# I2_S format (recommended, works with vanilla llama.cpp)
python packages/inference/scripts/convert_checkpoint_to_gguf.py \
    models/my-checkpoint \
    --outfile models/my-model.gguf \
    --outtype i2_s \
    --validate
```

### Step 3: Verify Output
```bash
# Check file size (~1.1GB for 2B model in I2_S)
ls -lh models/my-model.gguf

# Quick inference test
echo '{"prompt": "2+2=", "n_predict": 10}' | \
    curl -s http://localhost:30000/completion -d @-
```

## Which Converter to Use

| Checkpoint Type | Script | Notes |
|-----------------|--------|-------|
| DLM (bf16 online-quant) | `convert_checkpoint_to_gguf.py` | Auto-detects and handles |
| DLM (packed 2-bit) | `convert_checkpoint_to_gguf.py` | Auto-unpacks weights |
| Microsoft BitNet | `setup_env.py` | Use official workflow |

## Runtime Requirements

### Vanilla llama.cpp (recommended)
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build -j4
./build/bin/llama-server -m model.gguf --port 30000
```

### Microsoft BitNet.cpp (for TQ1_0/TL2)
```bash
cd extern/BitNet
python setup_env.py --hf-repo my-model -q tl2
./run_inference.py --model my-model -p "Hello" -n 100
```

## Testing Conversion

Run the test suite:
```bash
uv run pytest packages/inference/tests/test_gguf_conversion.py -v
```

Key tests:
- `test_online_quantization_distribution`: Verifies ~30% zeros, ~35% each +1/-1
- `test_unpack_shape_transformation`: Verifies [N/4, K] → [N, K] unpacking
- `test_validate_correct_size_i2s`: Verifies ~1.1GB output for 2B model
