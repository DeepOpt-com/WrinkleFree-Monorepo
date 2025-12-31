# WrinkleFree Monorepo

1.58-bit quantized LLM research using uv workspaces.

## MUST-DO Rules

1. **SYNC BEFORE REMOTE:** Before ANY ssh/remote command:
   `uv run gcd sync-ssh <host> --smart`

2. **NO GCP GPU:** Use Nebius or RunPod only

3. **NEVER CANCEL OTHERS' JOBS:** Only cancel SkyPilot jobs you started

4. **Read Package CLAUDE.md:** Before modifying a package, read its `packages/<pkg>/CLAUDE.md`

## Quick Commands

| Task | Command |
|------|---------|
| Sync to Desktop | `uv run gcd sync-ssh desktop --smart` |
| Start watch mode | `uv run gcd sync-ssh desktop --watch` |
| Run training | `uv run --package wrinklefree python scripts/train.py` |
| Run distillation | `uv run --package wrinklefree-distillation python scripts/distill.py` |
| Deploy to cloud | `wf train -m smollm2_135m -s 2 --cloud nebius` |

## Package Navigation

| Package | CLAUDE.md | Purpose |
|---------|-----------|---------|
| training | `packages/training/CLAUDE.md` | Stages 1, 1.9, 2 |
| distillation | `packages/distillation/CLAUDE.md` | Stage 3+ |
| architecture | `packages/architecture/CLAUDE.md` | BitLinear/SubLN |
| data_handler | `packages/data_handler/CLAUDE.md` | Data loading |
| inference | `packages/inference/CLAUDE.md` | Model serving |
| deployer | `packages/deployer/CLAUDE.md` | Cloud deploy |

## DLM GGUF Conversion (CRITICAL)

**NEVER use native F16 format for BitNet inference - always use I2_S or TQ2_0!**

### The Problem: Packed 2-bit Weights
DLM checkpoints store weights in a **packed 2-bit format** (4 values per byte) with separate `weight_scale` tensors. The standard `convert_hf_to_gguf.py` does NOT unpack these correctly, causing:
- Shape mismatch errors (e.g., `[640,2560]` instead of `[2560,2560]`)
- Gibberish output if converted without unpacking

### The Fix: Use Microsoft BitNet's Converter
The fix is in `extern/reference/BitNet.cpp/utils/convert-hf-to-gguf-bitnet.py`:
1. Build a `scale_map` from `weight_scale` tensors
2. Unpack 2-bit packed weights: `(data >> shift) & 3 - 1` to get ternary values
3. Reshape from `[N/4, ...]` to `[N, ...]`

### Correct Workflow
```bash
# 1. Download DLM checkpoint from GCS (skip optimizer state)
mkdir -p models/dlm-bitnet-2b
gcloud storage cp \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-3600/*.json' \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-3600/*.safetensors' \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-3600/*.jinja' \
    models/dlm-bitnet-2b/

# 2. Fix architecture name (capital N -> lowercase n)
sed -i 's/BitNetForCausalLM/BitnetForCausalLM/g' models/dlm-bitnet-2b/config.json

# 3. Convert using Microsoft BitNet's converter with TL2 output
cd extern/reference/BitNet.cpp
python utils/convert-hf-to-gguf-bitnet.py \
    ../../../models/dlm-bitnet-2b \
    --outtype tl2 \
    --outfile ../../../models/dlm-bitnet-2b.gguf

# 4. Verify model size (~1.1-1.2GB for 2B model, NOT 4.5GB)
ls -lh models/dlm-bitnet-2b.gguf
```

### Quantization Types
| Type | Size | Speed | Notes |
|------|------|-------|-------|
| I2_S | ~1.1GB | Fast | 2-bit ternary, multiply-add |
| TL2 | ~1.1GB | Faster | LUT-based, 5-bit index per 3 weights |
| TQ2_0 | ~1.2GB | Fast | llama.cpp ternary format |
| F16 | ~4.5GB | SLOW | DO NOT USE for inference! |

## Reference
For detailed docs (pipeline diagrams, GCP config, troubleshooting): `docs/ai-code/reference.md`
