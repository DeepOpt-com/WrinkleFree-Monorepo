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

**NEVER use F16 or TQ2_0 for bf16 DLM checkpoints - use TQ1_0!**

### The Problem: bf16 "Online-Quant" Checkpoints
DLM bf16 checkpoints store continuous float weights that are quantized at runtime. Using `llama-quantize` to convert F16 → TQ2_0 **destroys the model output** (produces garbage).

### Quantization Format Benchmarks (GCP C3D-32, Dec 2025)
| Type | Size | Speed | Output Quality | Notes |
|------|------|-------|----------------|-------|
| **TQ1_0** | ~678MB | 63 tok/s | ✅ Coherent | **RECOMMENDED** |
| TQ2_0 | ~779MB | 76 tok/s | ❌ GARBAGE | Do NOT use for bf16! |
| I2_S | ~1.1GB | ~55 tok/s | ✅ Coherent | Larger but works |
| TL2 | ~1.1GB | ~80 tok/s | ✅ Coherent | Requires kernel config |
| F16 | ~4.5GB | ~30 tok/s | ✅ Coherent | Too slow/large |

### Correct Workflow
```bash
# 1. Download DLM checkpoint from GCS (skip optimizer state)
mkdir -p models/dlm-bitnet-2b
gcloud storage cp \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-3600/*.json' \
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-3600/*.safetensors' \
    models/dlm-bitnet-2b/

# 2. Convert using our wrapper script (handles architecture name, TQ1_0 default)
uv run python packages/inference/scripts/convert_checkpoint_to_gguf.py \
    models/dlm-bitnet-2b \
    --outfile models/dlm-bitnet-2b.gguf \
    --validate

# 3. Verify model has correct architecture
xxd models/dlm-bitnet-2b.gguf | head -5  # Should show "bitnet-b1.58"

# 4. Serve with llama.cpp
llama-server -m models/dlm-bitnet-2b.gguf --port 30000
```

### Manual Conversion (Alternative)
```bash
# Fix architecture name first
sed -i 's/BitNetForCausalLM/BitnetForCausalLM/g' models/dlm-bitnet-2b/config.json

# Convert directly with BitNet's converter
uv run python extern/BitNet/utils/convert-hf-to-gguf-bitnet.py \
    models/dlm-bitnet-2b \
    --outtype tq1_0 \
    --outfile models/dlm-bitnet-2b.gguf
```

### Why TQ2_0 Fails
TQ2_0 conversion uses `llama-quantize` on F16 intermediate, which re-quantizes already-ternary-intended weights and destroys the distribution. TQ1_0 correctly preserves the ternary structure.

## Reference
For detailed docs (pipeline diagrams, GCP config, troubleshooting): `docs/ai-code/reference.md`
