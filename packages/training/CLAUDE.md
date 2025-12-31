# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WrinkleFree is a repository for training and serving 1.58-bit (ternary) LLM models using:
- **Training**: BitDistill approach from arxiv.org/abs/2510.13998
- **Serving**: microsoft/BitNet as git submodule (at monorepo root: `extern/BitNet`)
- **Config**: Hydra
- **Package management**: uv
- **Distributed**: FSDP for single/multi-GPU
- **Precision**: bfloat16 for training stability

## Monorepo Integration

This package is part of the WrinkleFree monorepo and depends on:
- **data_handler**: Shared data loading and influence functions
- **bitnet_arch**: BitNet layers (BitLinear, SubLN) and model conversion

**Related packages**:
| Package | Relationship |
|---------|--------------|
| `data_handler` | Data loading, influence optimization |
| `architecture` | BitNet layers and model conversion |
| `distillation` | Knowledge distillation (Stage 3+) |
| `deployer` | Cloud deployment (launches training jobs) |
| `converter` | Converts trained models to DLM format |
| `inference` | Serves trained models |
| `eval` | Evaluates trained models |

**Running from monorepo root**:
```bash
uv run --package wrinklefree python packages/training/scripts/train.py model=smollm2_135m training=stage2_pretrain
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run SmolLM2-135M training (smallest model, good for testing)
uv run python scripts/train.py model=smollm2_135m training=stage1_subln
uv run python scripts/train.py model=smollm2_135m training=stage1_9_layerwise data=fineweb
uv run python scripts/train.py model=smollm2_135m training=stage2_pretrain data=fineweb

# Run larger models
uv run python scripts/train.py model=qwen3_4b training=stage2_pretrain data=fineweb
```

## Training Pipeline

### Unified Training (Recommended)

The `unified` config combines STE quantization training with DLM (Diffusion Language Model) objectives in a single pass:

```bash
# Combined STE + DLM training (GitHub Issue #2)
uv run python scripts/train.py model=smollm2_135m training=unified data=fineweb

# Key features:
# - Auto-converts model to BitNet if needed
# - Multi-task: LM loss + DLM masking loss on same data
# - Curriculum: Phases ramp up DLM weight over training
# - MuonClip optimizer with QK clipping
# - WandB logging with per-objective losses
```

**Configurable Resume**:
```bash
# Resume with fresh optimizer (new LR schedule)
uv run python scripts/train.py training=unified \
  training.resume.checkpoint_path=gs://bucket/checkpoint.pt \
  training.resume.load_optimizer_state=false \
  training.resume.load_scheduler_state=false
```

### Legacy Stages (Still Supported)

| Stage | Config | Purpose | Tokens |
|-------|--------|---------|--------|
| 1 | `stage1_subln` | Convert model: insert SubLN + BitLinear | N/A (conversion only) |
| 1.9 | `stage1_9_layerwise` | Layer-wise distillation to align with teacher | ~100M |
| 2 | `stage2_pretrain` | Continue pre-training with ternary weights | ~10B |
| 3 | **Moved to `distillation` package** | Knowledge distillation fine-tuning | ~1B |

### Training Commands by Stage

```bash
# Stage 1: SubLN Insertion (no actual training, just conversion)
uv run python scripts/train.py \
  model=smollm2_135m \
  training=stage1_subln \
  distributed=single_gpu

# Stage 1.9: Layer-wise Distillation (quick alignment, ~100M tokens)
uv run python scripts/train.py \
  model=smollm2_135m \
  training=stage1_9_layerwise \
  data=fineweb \
  distributed=single_gpu

# Stage 2: Continue Pre-training (~10B tokens)
uv run python scripts/train.py \
  model=smollm2_135m \
  training=stage2_pretrain \
  data=fineweb \
  distributed=fsdp_multi

# Stage 3: Distillation (use the separate distillation package)
# See packages/distillation for the distillation package
uv run --package wrinklefree-distillation python scripts/distill.py \
  student.checkpoint_path=outputs/stage2/checkpoint.pt
```

### Hydra Override Examples

```bash
# Limit training steps (for smoke tests)
uv run python scripts/train.py model=smollm2_135m training=stage1_9_layerwise \
  training.max_steps=100

# Change output directory
uv run python scripts/train.py model=smollm2_135m training=stage1_9_layerwise \
  training.output_dir=/tmp/checkpoints

# Disable wandb logging
uv run python scripts/train.py model=smollm2_135m training=stage1_9_layerwise \
  training.logging.wandb.enabled=false

# Multi-GPU with FSDP
uv run python scripts/train.py model=qwen3_4b training=stage2_pretrain \
  distributed=fsdp_multi
```

## Checkpoint Path Structure

Checkpoints are auto-discovered across local, GCS, and Modal using a unified path structure:

```
{output_dir}/checkpoints/{experiment_name}/{stage}_checkpoint/checkpoints/final/checkpoint.pt
gs://{bucket}/checkpoints/{experiment_name}/{stage}_checkpoint/checkpoints/final/checkpoint.pt
```

**Stage names**: `stage1_checkpoint`, `stage1_9_checkpoint`, `stage2_checkpoint`

**Auto-discovery priority**:
1. Local path (fastest)
2. GCS bucket (if `gcs.enabled=true`)
3. HuggingFace Hub (fallback)

**Multi-stage pipeline**: Each stage automatically finds the previous stage's output:
- Stage 1.9 → looks for `stage1_checkpoint`
- Stage 2 → looks for `stage1_9_checkpoint` (falls back to `stage1_checkpoint`)
- Stage 3 → looks for `stage2_checkpoint`

**Enable GCS checkpointing**:
```bash
uv run python scripts/train.py ... gcs.enabled=true gcs.bucket=wrinklefree-checkpoints
```

## Resume from Checkpoint

To resume training from a specific checkpoint (e.g., step 5000):

```bash
# Via deployer CLI (recommended)
wf train -m qwen3_4b -s 2 --cloud nebius --resume gs://wrinklefree-checkpoints/checkpoints/bitdistill_qwen3_4b/stage2_checkpoint/checkpoints/step_5000/checkpoint.pt

# Or via environment variable
RESUME_CHECKPOINT=gs://...checkpoint.pt python scripts/train.py ...
```

**Important**: When resuming, the script:
1. Downloads the resume checkpoint directly from GCS
2. Creates model architecture (skipping stage1_9 download - saves ~15GB!)
3. Loads weights from the resume checkpoint
4. Continues training from the saved step

The checkpoint must be a **file path** (ending in `checkpoint.pt`), not a directory.

## Cloud Deployment (SkyPilot)

Training can be run on cloud GPUs via SkyPilot (configured in `WrinkleFree-Deployer`).

**Important**: Use **on-demand** instances (not spot) for smoke tests and time-sensitive runs. Spot H100:4 instances are often unavailable on RunPod.

```bash
cd ../deployer

# Load credentials
source credentials/.env

# Activate venv (required for sky command)
source .venv/bin/activate

# Launch smoke test on RunPod (on-demand for reliability)
sky launch skypilot/smoke_test.yaml -y --cluster smoke-test

# Monitor logs
sky logs smoke-test

# Terminate when done
sky down smoke-test -y
```

## Cloud Deployment (Modal)

Training can also run on Modal with automatic W&B logging.

### Quick Start

```bash
# Set W&B key locally (gets passed to Modal automatically)
export WANDB_API_KEY=your_key_here

# Single stage training
modal run modal_train.py --model smollm2_135m --stage 2

# Full pipeline (all stages)
modal run modal_train.py::run_full_pipeline --model smollm2_135m
```

### W&B Logging on Modal

W&B logging is enabled automatically when `WANDB_API_KEY` is available. Two methods:

**Method 1: Local Environment Variable (Recommended)**
```bash
export WANDB_API_KEY=your_key_here
modal run modal_train.py --model smollm2_135m --stage 2
```
The CLI will show: `W&B: enabled (key from local env)`

**Method 2: Modal Secret (Fallback)**
```bash
# One-time setup
modal secret create wandb-api-key WANDB_API_KEY=your_key_here

# Run training (will use Modal secret if local env not set)
modal run modal_train.py --model smollm2_135m --stage 2
```

### How It Works

1. Local entrypoint reads `WANDB_API_KEY` from your local environment
2. Passes it as a function argument to the remote Modal function
3. Remote function sets `os.environ["WANDB_API_KEY"]` before training
4. Falls back to Modal secret if local env not set

### Verify W&B is Working

Check the logs for:
```
WandB API key: configured
```

If you see `WandB API key: MISSING!`, the key isn't being passed correctly.

## Available Models

| Model | Config | Params | VRAM (Stage 2) |
|-------|--------|--------|----------------|
| SmolLM2-135M | `smollm2_135m` | 135M | ~4GB |
| Qwen3-4B | `qwen3_4b` | 4B | ~24GB |

## GPU Profiles

**IMPORTANT**: Always use GPU-appropriate batch sizes to maximize VRAM utilization.
Profiles are in `configs/gpu/`. Reference them when setting batch sizes.

| GPU | Profile | VRAM | Qwen3-4B Stage 1.9 | Qwen3-4B Stage 2 |
|-----|---------|------|--------------------|--------------------|
| A100-80GB | `a100_80gb` | 80GB | batch=8, accum=8 | batch=16, accum=4 |
| A100-40GB | `a100_40gb` | 40GB | batch=2, accum=32 | batch=4, accum=16 |
| H100-80GB | `h100_80gb` | 80GB | batch=8, accum=8 | batch=16, accum=4 |
| A10G | `a10g_24gb` | 24GB | batch=1, accum=64 | batch=2, accum=32 |
| RTX 4090 | `rtx4090` | 24GB | batch=1, accum=64 | batch=2, accum=32 |

### Cloud Deployment Defaults

When deploying to cloud (SkyPilot), always specify:
- **disk_size: 100** (storage is cheap, prevents failures)
- **Batch sizes from GPU profile** (see table above)

Example for A100-80GB:
```bash
# Stage 1.9
training.batch_size=8 training.gradient_accumulation_steps=8

# Stage 2
training.batch_size=16 training.gradient_accumulation_steps=4
```

## Architecture

### Core Components
- `src/wrinklefree/models/bitlinear.py` - BitLinear layer with STE quantization (ternary weights)
- `src/wrinklefree/models/subln.py` - SubLN normalization (key BitDistill component)
- `src/wrinklefree/distillation/` - Logits KL + attention + layer-wise distillation losses
- `src/wrinklefree/training/stage1.py` - Stage 1 SubLN insertion
- `src/wrinklefree/training/stage1_9.py` - Stage 1.9 layer-wise distillation
- `src/wrinklefree/training/fsdp_wrapper.py` - FSDP wrapping with activation checkpointing

### Quantization
- **BitNet 1.58-bit**: Ternary weights {-1, 0, 1} (3 levels, 1.58 bits/weight)

### Configuration
All configs in `configs/` using Hydra:
- `model/` - Model architecture configs (smollm2_135m, qwen3_4b)
- `training/` - Stage-specific training configs
- `data/` - Dataset configs (fineweb, falcon, downstream)
- `distributed/` - FSDP/DDP settings (single_gpu, fsdp_multi)

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/

# Type check
uv run mypy src/
```

## MoE (Mixture of Experts) Support

The repo includes MoE infrastructure for testing and training MoE variants of BitNet models.

### Core MoE Components
- `src/wrinklefree/moe/router.py` - TopKRouter, IdentityRouter for expert selection
- `src/wrinklefree/moe/expert.py` - BitNetExpertFFN, BitNetMoEFFN with 1.58-bit weights
- `src/wrinklefree/moe/fake_moe.py` - Convert dense models to MoE for testing

### Fake MoE Testing
```python
from wrinklefree.moe import create_fake_moe_from_dense, verify_moe_matches_dense

# Convert dense model to MoE (all experts share weights, IdentityRouter)
moe_model = create_fake_moe_from_dense(model, num_experts=8, top_k=2)

# Verify outputs match (should be identical)
matches, error = verify_moe_matches_dense(model, moe_model, input_ids)
```

## GGUF Conversion

Convert trained models to GGUF format for BitNet.cpp inference.

### Conversion Functions
```python
from wrinklefree.serving.converter import (
    convert_to_gguf,           # Dense model to GGUF
    convert_moe_to_gguf,       # MoE model to GGUF
    convert_dense_to_fake_moe_gguf,  # Dense -> Fake MoE -> GGUF
)

# Convert trained checkpoint
convert_to_gguf(
    model_path=Path("outputs/stage2/model.safetensors"),
    output_path=Path("outputs/model.gguf"),
    quant_type="i2_s",  # CPU-optimized
)
```

### GGUF Tensor Naming (BitNet format)
- `token_embd.weight` - Token embeddings
- `blk.{n}.attn_q.weight` - Attention Q projection
- `blk.{n}.ffn_gate.weight` - FFN gate (SwiGLU)
- `blk.{n}.ffn_gate_exps.weight` - MoE expert gates (3D)
- `blk.{n}.ffn_gate_inp.weight` - MoE router

## W&B Tracking

Intelligent run naming for training and benchmarks.

### Training Runs
```python
from wrinklefree.training.run_naming import generate_run_name

# Auto-generates: qwen3_4b-s2-muon-lr2.4e3-bs64-a3f
name = generate_run_name(hydra_config)
```

### Benchmark Runs
```python
from wrinklefree.training.run_naming import generate_benchmark_name, generate_moe_benchmark_name

# Dense: bn2b-i2s-ctx4096-t16-a3f
name = generate_benchmark_name("bitnet-2b", "i2_s", 4096, 16)

# MoE: bn2b-moe8k2-i2s-ctx4096-t16-a3f
name = generate_moe_benchmark_name("bitnet-2b", 8, 2, "i2_s", 4096, 16)
```

### W&B Tracker
```python
from benchmark.core.wandb_tracker import WandBTracker, create_inference_tracker

# Training
tracker = WandBTracker(project="wrinklefree")
tracker.init_training_run(config)
tracker.log_training_step({"loss": 0.5})
tracker.finish()

# Inference benchmarks
tracker = create_inference_tracker("bitnet-2b", "i2_s", 4096, 16)
tracker.log_inference_metrics(metrics)
tracker.finish()
```

## Q-Sparse Activation Sparsity (Optional)

Q-Sparse adds activation sparsity for inference efficiency. Based on [arxiv:2407.10969](https://arxiv.org/abs/2407.10969).

**Status**: Disabled by default due to ~35% training slowdown.

```bash
# Enable Q-Sparse (61% activation sparsity)
uv run python scripts/train.py model=smollm2_135m training=stage2_pretrain \
  training.activation_sparsity.enabled=true

# Or via Modal
modal run src/wf_deployer/modal_deployer.py --model smollm2_135m --stage 2 \
  --hydra-overrides "training.activation_sparsity.enabled=true"
```

**Ablation results (200 steps, SmolLM2-135M):**
- Without Q-Sparse: loss ~6.93, 480s
- With Q-Sparse: loss ~6.90, 648s

**When to enable**: Production models where inference efficiency matters. The 35% training overhead pays off at inference where 61% of activations can be skipped.

**Config options** (in `configs/training/stage2_pretrain.yaml`):
- `activation_sparsity.enabled`: Toggle (default: false)
- `activation_sparsity.sparsity_ratio`: Target sparsity (default: 0.61)
- `activation_sparsity.mode`: "topk" or "block" (N:M structured)
- `activation_sparsity.warmup.warmup_steps`: Gradual warmup (default: 1000)

## FSDP Multi-GPU Training

When using `distributed=fsdp_multi` for multi-GPU training, be aware of these critical requirements:

### Collective Operations
FSDP uses collective operations that **ALL ranks must participate in**:
- `save_checkpoint()` - Gathering sharded state dict is collective
- `eval_loss` must be synchronized across ranks for consistent checkpoint save decisions

**Key fixes in trainer.py:**
1. **Best checkpoint save**: All ranks call `save_checkpoint("best")`, not just rank 0
2. **Eval loss sync**: Added `dist.all_reduce` to synchronize eval loss across ranks
3. **Dataloader verification**: Added check that batch counts match across ranks

### Muon Optimizer with FSDP

**CRITICAL**: The original `muon-clip` package is **incompatible with FSDP** because it broadcasts raw parameters, but FSDP shards them across ranks.

**Solution**: Use `muon-fsdp2` (from PyPI) which uses gather-scatter instead of broadcast:

```python
from muon_fsdp2 import Muon

optimizer = Muon([
    {"params": muon_params, "lr": lr_muon, "use_muon": True},
    {"params": adam_params, "lr": lr_adam, "use_muon": False}
])
```

When `training.optimizer.type=muonclip` is specified, the trainer automatically uses `muon_fsdp2.Muon`.

### Common FSDP Hangs

| Symptom | Cause | Fix |
|---------|-------|-----|
| Hang at checkpoint | Only rank 0 calls save_checkpoint | All ranks must call |
| Hang at eval | Different ranks make different save decisions | Sync eval loss with all_reduce |
| Muon collective mismatch | Broadcast with sharded params | Use muon_fsdp2 |
| Dataloader mismatch | Different batch counts per rank | Use drop_last=True |

## Notes

- Training uses bfloat16 for numerical stability
- Teacher models are loaded in bfloat16 to match student dtype
- Use 8-bit AdamW (bitsandbytes) or Muon optimizer for memory efficiency
- **FSDP with Muon**: Use `muon-fsdp2` package (automatically selected when `optimizer.type=muonclip`)
- BitNet submodule (at meta-repo root ../extern/BitNet) is for inference only
- MoE support uses llama.cpp's Mixtral-style tensor packing

## Training Data (from data_handler)

**Data configs are managed by data_handler, NOT this package.**

This package's `configs/data/default.yaml` just specifies `config_name: mixed_pretrain`, which loads
the actual data config from `data_handler/configs/data/mixed_pretrain.yaml`.

The `mixed_pretrain` config includes:
- 6 data sources (DCLM, FineWeb-Edu, GitHub Code 2025, FineMath, SlimPajama, SYNTH)
- Multi-domain probe loaders for influence-based remixing
- All sources are commercially friendly (CC-BY, ODC-By, MIT, Apache 2.0, CDLA)

**To use a different data config**, override `data.config_name`:
```bash
uv run python scripts/train.py model=smollm2_135m training=stage2_pretrain \
  data.config_name=fineweb  # Use data_handler's fineweb.yaml
```

**Available configs** (in `packages/data_handler/configs/data/`):
- `mixed_pretrain` - Multi-source with influence (default, recommended)
- `fineweb` - Single-source FineWeb-Edu (no influence)
- `downstream` - SFT/finetuning tasks (Stage 3)

## A10G GPU Settings

For A10G (24GB VRAM) deployments on Modal with `--scale dev`:

| Model | Stage | Batch Size | Grad Accum |
|-------|-------|------------|------------|
| SmolLM2-135M | Stage 2 | 32 | 4 |
| SmolLM2-135M | Stage 1.9 | 64 | 2 |

**IMPORTANT**: Use `training.batch_size=32` for SmolLM2-135M Stage 2 on A10G to avoid OOM.
