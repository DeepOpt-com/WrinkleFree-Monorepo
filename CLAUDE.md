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
| Run BitDistill | `uv run --package wrinklefree python scripts/train.py training=bitdistill_full` |
| Run LRC calibration | `uv run --package wrinklefree python scripts/train.py training=lrc_calibration` |
| Deploy to cloud | `wf train -m smollm2_135m -s 2 --cloud nebius` |

## Package Navigation

| Package | CLAUDE.md | Purpose |
|---------|-----------|---------|
| training | `packages/training/CLAUDE.md` | Stages 1-3 + Distillation + LRC |
| architecture | `packages/architecture/CLAUDE.md` | BitLinear/BitLinearLRC/SubLN |
| data_handler | `packages/data_handler/CLAUDE.md` | Data loading |
| inference | `packages/inference/CLAUDE.md` | Model serving |
| deployer | `packages/deployer/CLAUDE.md` | Cloud deploy |
| mobile | `packages/mobile/CLAUDE.md` | Android inference |
| eval | `packages/eval/CLAUDE.md` | Model evaluation |

> **Note**: Legacy packages (`distillation`, `converter`, `cheapertraining`) are archived in `packages/_legacy/`.

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
    'gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/checkpoint-step-3600/*.jinja' \
    models/dlm-bitnet-2b/

# 2. Fix architecture name (capital N -> lowercase n)
sed -i 's/BitNetForCausalLM/BitnetForCausalLM/g' models/dlm-bitnet-2b/config.json

# 3. Convert using Microsoft BitNet's converter with TQ1_0 output
cd extern/reference/BitNet.cpp
python utils/convert-hf-to-gguf-bitnet.py \
    ../../../models/dlm-bitnet-2b \
    --outtype tq1_0 \
    --outfile ../../../models/dlm-bitnet-2b.gguf

# 4. Verify model size (~678MB for TQ1_0, NOT 4.5GB)
ls -lh models/dlm-bitnet-2b.gguf
```

### Why TQ2_0 Fails
TQ2_0 conversion uses `llama-quantize` on F16 intermediate, which re-quantizes already-ternary-intended weights and destroys the distribution. TQ1_0 correctly preserves the ternary structure.

## Reference
For detailed docs (pipeline diagrams, GCP config, troubleshooting): `docs/ai-code/reference.md`
