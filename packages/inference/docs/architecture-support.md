# Multi-Architecture Support for DLM Inference

This document describes the architecture requirements and conversion process for different BitNet/DLM model types.

## Supported Architectures

| Architecture | Model Examples | GGUF Support | SubLN | Notes |
|--------------|----------------|--------------|-------|-------|
| **LLaMA** | Microsoft BitNet-2B, Qwen-based | Full | No | Primary, well-tested |
| **SmolLM2** | SmolLM2-135M | Limited | **Yes** | Requires SubLN support in llama.cpp |
| **Qwen** | Qwen2, Qwen3 | Full | No | GQA support |
| **Mistral** | Mistral 7B | Full | No | Sliding window attention |

## Architecture Detection

The GGUF converter reads `config.json` to detect the architecture:

```json
{
  "architectures": ["BitnetForCausalLM"],
  "model_type": "llama",
  "num_attention_heads": 20,
  "num_key_value_heads": 5,
  "head_dim": 128,
  "rope_theta": 500000.0
}
```

## SubLN Architecture (IMPORTANT)

Some WrinkleFree-trained models use **SubLN (Sub-Layer Normalization)**, which adds scale tensors after certain layers:

```
Standard LLaMA:     o_proj.weight
SubLN Architecture: o_proj.weight + o_proj_scale.weight (or o_proj.0/1)
```

### SubLN Detection

Check your checkpoint for SubLN format:

```python
from safetensors import safe_open
with safe_open("model.safetensors", framework='pt') as f:
    keys = list(f.keys())
    subln = [k for k in keys if '.0.weight' in k or '_scale' in k]
    if subln:
        print(f"SubLN architecture detected: {len(subln)} scale tensors")
```

### SubLN Limitations

**Standard llama.cpp does NOT support SubLN** - attempting to load will fail with:
```
error loading model: check_tensor_dims: tensor 'blk.0.attn_sub_norm.weight' not found
```

Options for SubLN models:
1. Use a llama.cpp fork with SubLN support (not yet available)
2. Merge scales into weights (requires non-trivial weight transformation)
3. Train without SubLN for inference compatibility

## Weight Formats

| Checkpoint Type | Weight Format | Conversion |
|-----------------|---------------|------------|
| Training (bf16) | Full precision | Quantize to TQ1_0/TQ2_0 |
| Training (ternary) | Already {-1, 0, +1} | Direct pack to I2_S |
| Pre-packed (2-bit) | 4 values/byte | Unpack then repack for GGUF |

### Check Weight Format

```python
import torch
ckpt = torch.load("checkpoint.pt", map_location='cpu')
state = ckpt.get('model_state_dict', ckpt)
w = list(state.values())[0]
print(f"dtype: {w.dtype}")
print(f"unique values: {len(w.unique())}")  # Ternary = 3 values
```

## GGUF Quantization Types

| Type | Size | Speed | Use Case |
|------|------|-------|----------|
| **TQ1_0** | Smallest | Fast | Ternary models (RECOMMENDED) |
| TQ2_0 | Small | Fast | NOT for bf16 checkpoints |
| I2_S | Medium | Medium | Legacy ternary |
| F16 | Large | Slow | Full precision fallback |

**WARNING**: TQ2_0 produces garbage output for bf16 DLM checkpoints. Always use TQ1_0.

## Conversion Workflow by Architecture

### Standard LLaMA/BitNet (No SubLN)

```bash
# 1. Verify no SubLN
python -c "from safetensors import safe_open; f=safe_open('model.safetensors','pt'); print([k for k in f.keys() if '_scale' in k or '.0.weight' in k][:5])"

# 2. Convert to GGUF
python convert_hf_to_gguf.py /path/to/checkpoint --outtype tq1_0 --outfile model.gguf

# 3. Verify
ls -lh model.gguf  # Should be ~0.6GB per 1B params
```

### SubLN Architecture (SmolLM2, etc.)

SubLN models require special handling. Current workaround:

```python
# Option 1: Skip scales (loses SubLN benefit, may affect quality)
from safetensors.torch import save_file
import torch

ckpt = torch.load("checkpoint.pt", map_location='cpu')
state = ckpt['model_state_dict']

# Remove SubLN scale tensors
filtered = {k: v for k, v in state.items()
            if '.0.weight' not in k and '_scale' not in k}

# Rename .1.weight -> .weight
renamed = {}
for k, v in filtered.items():
    renamed[k.replace('.1.weight', '.weight')] = v

save_file(renamed, "model_no_subln.safetensors")
```

**Note**: This removes SubLN normalization and may affect model quality.

## DLM-Specific Requirements

DLM (Diffusion LLM) models require:

1. **Mask Token**: Token ID for `|<MASK>|` in vocabulary
2. **Block Size**: Must match training (default: 32)
3. **Threshold**: Confidence threshold for iterative mode

### Verify DLM Compatibility

```bash
# Check for mask token
./llama-cli -m model.gguf --prompt "|<MASK>|" -n 1 2>&1 | grep -i mask
```

Expected: Should show mask token ID (e.g., `128256`)

### DLM Server Modes

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| `greedy` | ~61 tok/s | Baseline | Maximum speed |
| `iterative` | ~54 tok/s | Good | Balanced |
| `adaptive` | ~61 tok/s | Best | **RECOMMENDED** |

## Troubleshooting

### "tensor 'blk.0.attn_sub_norm.weight' not found"

Model has SubLN architecture. See SubLN section above.

### Output contains `|<MASK>|` tokens

DLM mode isn't unmasking properly. Check:
1. Mask token ID detection (should auto-detect)
2. Use `--decode-mode greedy` for testing

### "Can't quantize tensor with shape..."

Tensor dimensions don't match TQ1_0 requirements. Converter falls back to F16.

### Garbage output (repeating tokens)

Used wrong quantization type. Always use TQ1_0 for bf16 checkpoints.

## Model Sources

| Checkpoint | Location | Architecture |
|------------|----------|--------------|
| BitNet 2B (DLM) | `gs://wrinklefree-checkpoints/dlm/*.gguf` | LLaMA |
| SmolLM2 135M | `gs://wrinklefree-checkpoints/checkpoints/bitdistill_smollm2_135m/` | SmolLM2 (SubLN) |

## Related Documentation

- [DLM Pipeline](dlm-pipeline.md) - End-to-end inference setup
- [DLM Troubleshooting](dlm-troubleshooting.md) - Common issues
- [GGUF Conversion](gguf-conversion.md) - Conversion details
