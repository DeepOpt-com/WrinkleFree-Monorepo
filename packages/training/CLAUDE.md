# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL Rules

1. **USE LIGHTNING TRAINER**: Use `train_lightning.py` (legacy `train.py` has been removed)
2. **AUTO BATCH SIZE**: Use `training.auto_batch_size=true` for single GPU runs only (NOT supported with DDP/FSDP!)
3. **CLEAN CHECKPOINTS**: Before re-running failed jobs, clean `/tmp/checkpoints/` on remote
4. **EXPERIMENTAL CODE**: MoE and TensorParallel are in `_experimental/` (not production-ready)
5. **LEGACY CODE**: Legacy trainers are in `training/_legacy/` (use Lightning instead)

## Known Bugs

| Bug | Workaround | Status |
|-----|------------|--------|
| MuonClip + BatchSizeFinder | `MuonClipInitCallback` patches upstream bug | **Fixed** (2026-01-01) |
| BatchSizeFinder + WandB resume | Clean checkpoints before retry | Fixed in code |
| PyTorch 2.6 weights_only | Added safe_globals for omegaconf types | Fixed |
| BatchSizeFinder + DDP/FSDP | Auto-skipped with warning; manually tune batch_size | **Known limitation** |

**MuonClip Fix Details**: The upstream muon-clip package has a bug where `HookRecorder.remove_hooks()`
doesn't reset `is_registered=False`. This causes hooks to never re-register after BatchSizeFinder's
`eval/train` cycles. Fixed via `MuonClipInitCallback` + patched `remove_hooks()` in module.py.

## Project Overview

WrinkleFree is a repository for training and serving 1.58-bit (ternary) LLM models using:
- **Training**: BitDistill approach from arxiv.org/abs/2510.13998
- **Serving**: microsoft/BitNet as git submodule (at monorepo root: `extern/BitNet`)
- **Config**: Hydra
- **Package management**: uv
- **Distributed**: FSDP for single/multi-GPU
- **Precision**: bfloat16 for training stability
- **TF32**: Enabled by default for Ampere+ GPUs (10-20% matmul speedup)

## Performance Optimizations

| Optimization | Status | Speedup | Notes |
|--------------|--------|---------|-------|
| **TF32** | Enabled | 10-20% | Auto-enabled for Ampere+ GPUs (A100, H100, RTX 30xx/40xx) |
| **Sequence Packing** | Enabled | 1.4-2x | Packs sequences to reduce padding waste |
| **torch.compile** | Not used | - | Incompatible with FSDP multi-GPU |
| **FlashAttention (SDPA)** | Via HuggingFace | ~2x attention | HF models use SDPA by default |

## Monorepo Integration

This package is part of the WrinkleFree monorepo and depends on:
- **data_handler**: Shared data loading and influence functions
- **bitnet_arch**: BitNet layers (BitLinear, SubLN) and model conversion

**Related packages**:
| Package | Relationship |
|---------|--------------|
| `data_handler` | Data loading, influence optimization |
| `architecture` | BitNet layers and model conversion |
| `deployer` | Cloud deployment (launches training jobs) |
| `inference` | Serves trained models |
| `eval` | Evaluates trained models |

**Note**: Knowledge distillation (BitDistill, TCS) is now integrated into the objectives system.
The old `distillation` package has been moved to `_legacy/distillation/`.

**Running from monorepo root**:
```bash
uv run --package wrinklefree python packages/training/scripts/train_lightning.py model=smollm2_135m training=base
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run SmolLM2-135M training (smallest model, good for testing)
uv run python scripts/train_lightning.py model=smollm2_135m training=base

# With auto batch size scaling
uv run python scripts/train_lightning.py model=smollm2_135m training=base \
    training.auto_batch_size=true

# Run larger models
uv run python scripts/train_lightning.py model=qwen3_4b training=base
```

## Training Pipeline

### Unified Training (Recommended)

The `base` config combines STE quantization training with DLM (Diffusion Language Model) objectives in a single pass:

```bash
# Combined STE + DLM training (GitHub Issue #2)
uv run python scripts/train_lightning.py model=smollm2_135m training=base

# Key features:
# - Auto-converts model to BitNet if needed
# - Multi-task: LM loss + DLM masking loss on same data
# - Curriculum: Phases ramp up DLM weight over training
# - MuonClip optimizer with QK clipping
# - Influence-based data remixing (dynamic dataset weights)
# - WandB logging with per-objective losses
```

**Multi-Task Objectives**:
The `ObjectiveManager` runs multiple objectives on the same batch:
- `continue_pretrain`: Standard next-token prediction loss
- `dlm`: Diffusion Language Model masking loss

When DLM is enabled, the batch is preprocessed:
1. Original labels stored in `batch["_original_labels"]`
2. Input tokens masked with `mask_token_id`
3. CE objective uses original labels; DLM uses masked positions

**Curriculum Phases** (configurable in `unified.yaml`):
```yaml
curriculum:
  phases:
    - name: warmup         # Steps 0-10%
      objectives: {continue_pretrain: 1.0, dlm: 0.0}
    - name: dlm_ramp       # Steps 10-30%
      objectives: {continue_pretrain: 1.0, dlm: 0.3}
    - name: main           # Steps 30-80%
      objectives: {continue_pretrain: 1.0, dlm: 0.5}
    - name: dlm_focus      # Steps 80-100%
      objectives: {continue_pretrain: 0.5, dlm: 1.0}
```

**Configurable Resume**:
```bash
# Resume with fresh optimizer (new LR schedule)
uv run python scripts/train_lightning.py training=base \
  training.resume.checkpoint_path=gs://bucket/checkpoint.pt \
  training.resume.load_optimizer_state=false \
  training.resume.load_scheduler_state=false

# Resume options:
#   load_optimizer_state: true/false (reset LR schedule)
#   load_scheduler_state: true/false
#   load_training_state: true/false (reset step counter)
#   strict_model_load: true/false (allow missing keys)
```

### Inference Compatibility

Models trained with `training=base` support **BOTH** inference modes:
- **Autoregressive** (llama-cli): Works because we train with CE loss throughout
- **Block diffusion** (dlm_server): Works because we train with DLM loss; may be faster

| Training Config | Inference Mode | Notes |
|-----------------|----------------|-------|
| `objectives.continue_pretrain.enabled=true` | llama-cli (autoregressive) | Standard next-token prediction |
| `objectives.dlm.enabled=true` | dlm_server (block diffusion) | Parallel token prediction, potentially faster |

**Critical Notes**:
1. Use I2_S format for production (fastest, AVX-512 optimized)
2. **NEVER use TQ2_0** for bf16 checkpoints - it corrupts ternary weights
3. If llama-cli produces garbage, check quantization format first

**DLM Settings** (if using dlm_server):
- `mask_token_id` must match between training and inference (typically 0)
- Block diffusion speedup depends on model size and hardware

See `packages/inference/CLAUDE.md` for conversion and serving details.

### PyTorch Lightning Training (New)

The Lightning-based trainer provides a cleaner, more maintainable training loop with auto batch size scaling:

```bash
# Basic Lightning training
uv run python scripts/train_lightning.py model=smollm2_135m training=base

# With auto batch size scaling (finds max batch that fits GPU)
uv run python scripts/train_lightning.py model=smollm2_135m training=base \
  training.auto_batch_size=true

# All objectives work unchanged (DLM, LRC, distillation)
uv run python scripts/train_lightning.py model=smollm2_135m training=base \
  training.objectives.dlm.enabled=true \
  training.objectives.dlm.weight=0.5
```

**Key Features**:
- **Auto batch size**: `BatchSizeFinder` probes GPU memory at startup
- **Built-in DDP/FSDP**: Seamless distributed training
- **All objectives work**: ObjectiveManager reused unchanged
- **Custom callbacks**: GCS upload, ZClip, TokenCount, QKClip, LambdaWarmup

**Lightning Components** (`src/wrinklefree/lightning/`):
| File | Purpose |
|------|---------|
| `module.py` | `WrinkleFreeLightningModule` - wraps model + ObjectiveManager |
| `datamodule.py` | `WrinkleFreeDataModule` - wraps existing dataloaders |
| `callbacks.py` | Custom callbacks (GCS, ZClip, TokenCount, InfluenceTracker, etc.) |

**Smoke Tests** (L40 GPU):
```bash
cd packages/deployer
source credentials/.env

# Run specific objective combo
sky launch skypilot/smoke_test_lightning.yaml -y --cluster lightning-smoke \
  --env OBJECTIVE_COMBO=dlm

# Available combos: ce_only, dlm, distill, bitdistill, lrc
```

### Influence-Based Data Remixing

Dynamic dataset weight optimization during training (MobileLLM-R1 methodology).

**Lightning Integration**: Uses `InfluenceTrackerCallback` which wraps `data_handler.influence.InfluenceTracker`.

```bash
# Enable influence remixing with mixed_pretrain data (Lightning trainer)
uv run python scripts/train_lightning.py model=smollm2_135m training=base \
  data.config_name=mixed_pretrain \
  training.influence.enabled=true \
  training.influence.warmup_steps=1000 \
  training.influence.update_interval=5000 \
  training.influence.learning_rate=0.1

# What happens:
# 1. InfluenceTrackerCallback caches probe gradients at train start
# 2. Warmup (first 1000 steps): Use initial dataset weights
# 3. After warmup: Every 5000 steps, compute influence scores
# 4. Adjust dataset weights to maximize influence on probe domains
# 5. Log weights to WandB: influence/weight_{dataset_name}
```

**Smoke Test for Influence**:
```bash
cd packages/deployer
source credentials/.env
sky launch skypilot/smoke_test_influence.yaml -y --cluster influence-smoke
```

**How Influence Works**:
1. **Probe domains** (in `mixed_pretrain.yaml`): web_edu, code, math, dclm, diverse, reasoning
2. **DataInf algorithm**: Efficient gradient-based influence estimation
3. **Weight update**: `new_weight = (1 - lr) * current + lr * optimal`
4. **Constraints**: min_weight=0.05, max_weight=0.60 (no domain dominates)

**WandB Metrics**:
```
train/loss                    # Combined weighted loss
train/continue_pretrain_loss  # LM objective loss
train/dlm_loss               # DLM objective loss
train/dlm_num_masked         # Tokens masked per batch
schedule/continue_pretrain_weight  # Curriculum weight
schedule/dlm_weight               # Curriculum weight
influence/weight_{dataset}        # Per-dataset mixture weight
```

### LRC Calibration (Post-Quantization Recovery)

Low-Rank Correction (LRC) adds trainable low-rank matrices (U, V) to correct quantization errors.
Based on [Low-Rank Correction for Quantized LLMs](https://arxiv.org/abs/2412.07902).

```bash
# Train LRC adapters on calibration data
uv run python scripts/train_lightning.py model=smollm2_135m training=lrc_calibration data=fineweb

# Key features:
# - Converts BitLinear -> BitLinearLRC (adds U, V matrices)
# - Freezes ALL params except U, V (only LRC matrices trained)
# - Uses hidden state matching loss (teacher = original fp16 model)
# - Short calibration run (~50M tokens)
```

**How LRC Works**:
- Forward: `output = W_quant @ Q_a(X) + U @ V^T @ X`
  - `W_quant @ Q_a(X)`: frozen quantized path
  - `U @ V^T @ X`: trainable correction on unquantized activations
- Loss: `||h_teacher - h_student||²` per layer
- Rank: 10% of min(in, out) → ~50% error reduction

**Config** (`lrc_calibration.yaml`):
```yaml
lrc:
  rank_percentage: 0.1  # 10% rank
  init_method: zeros    # or "svd_residual"

objectives:
  lrc_reconstruction:
    enabled: true
    loss_type: mse
    layer_weights: progressive
```

### Legacy Stages (Still Supported)

| Stage | Config | Purpose | Tokens |
|-------|--------|---------|--------|
| 1 | `stage1_subln` | Convert model: insert SubLN + BitLinear | N/A (conversion only) |
| 1.9 | `stage1_9_layerwise` | Layer-wise distillation to align with teacher | ~100M |
| 2 | `stage2_pretrain` | Continue pre-training with ternary weights | ~10B |
| 3 | `bitdistill_full` | Knowledge distillation (BitDistill objectives) | ~1B |
| LRC | `lrc_calibration` | Post-quantization low-rank correction | ~50M |

### Training Commands by Stage

```bash
# Stage 1: SubLN Insertion (no actual training, just conversion)
uv run python scripts/train_lightning.py \
  model=smollm2_135m \
  training=stage1_subln \
  distributed=single_gpu

# Stage 1.9: Layer-wise Distillation (quick alignment, ~100M tokens)
uv run python scripts/train_lightning.py \
  model=smollm2_135m \
  training=stage1_9_layerwise \
  data=fineweb \
  distributed=single_gpu

# Stage 2: Continue Pre-training (~10B tokens)
uv run python scripts/train_lightning.py \
  model=smollm2_135m \
  training=stage2_pretrain \
  data=fineweb \
  distributed=fsdp_multi

# Stage 3: BitDistill (knowledge distillation via objectives)
uv run python scripts/train_lightning.py \
  model=smollm2_135m \
  training=bitdistill_full \
  data=mixed_pretrain

# LRC Calibration (post-quantization recovery)
uv run python scripts/train_lightning.py \
  model=smollm2_135m \
  training=lrc_calibration \
  data=fineweb
```

### Hydra Override Examples

```bash
# Limit training steps (for smoke tests)
uv run python scripts/train_lightning.py model=smollm2_135m training=stage1_9_layerwise \
  training.max_steps=100

# Change output directory
uv run python scripts/train_lightning.py model=smollm2_135m training=stage1_9_layerwise \
  training.output_dir=/tmp/checkpoints

# Disable wandb logging
uv run python scripts/train_lightning.py model=smollm2_135m training=stage1_9_layerwise \
  training.logging.wandb.enabled=false

# Multi-GPU with FSDP
uv run python scripts/train_lightning.py model=qwen3_4b training=stage2_pretrain \
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
uv run python scripts/train_lightning.py ... gcs.enabled=true gcs.bucket=wrinklefree-checkpoints
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

## Smoke Tests

Smoke tests validate the full training pipeline in ~5 minutes:

```bash
cd packages/deployer

# 1x L40 smoke test (20 steps with influence remixing)
sky launch skypilot/smoke_test_unified_1gpu.yaml -y --cluster unified-1gpu

# 2x L40 smoke test (FSDP data parallelism)
sky launch skypilot/smoke_test_unified_2gpu.yaml -y --cluster unified-2gpu

# Monitor
sky logs unified-1gpu
sky logs unified-2gpu

# Teardown
sky down unified-1gpu unified-2gpu -y
```

**Smoke Test Configuration**:
- **Steps**: 20 total (4 warmup + 16 with influence)
- **First 20%** (steps 1-4): fineweb-edu warmup, no influence
- **Remaining 80%** (steps 5-20): mixed_pretrain with influence updates
- **Checkpoints**: GCS upload every 10 steps
- **Verifies**: Loss decreases, MuonClip works, GCS/WandB logging works

**Expected Output**:
```
First loss: ~10-12
Last loss: ~6-8 (should decrease!)
Checkpoints: step_10/, step_20/, final/
WandB: https://wandb.ai/wrinklefree/runs/{run_id}
```

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

**Objectives** (`src/wrinklefree/objectives/`):
- `manager.py` - ObjectiveManager: runs multiple objectives on same batch
- `continue_pretrain.py` - ContinuePretrainObjective: next-token prediction
- `dlm.py` - DLMObjective: diffusion language model masking
- `layerwise_distill.py` - LayerwiseDistillationObjective: hidden state alignment
- `logits_distill.py` - LogitsDistillationObjective: KL divergence on teacher logits
- `attention_distill.py` - AttentionRelationDistillationObjective: attention pattern matching
- `tcs_distill.py` - TCSDistillationObjective: Target Concrete Score for DLM students
- `bitdistill.py` - BitDistillObjective: combined logits + attention distillation
- `lrc_reconstruction.py` - LRCReconstructionObjective: low-rank correction training
- `factory.py` - Creates ObjectiveManager from config
- `curriculum.py` - CurriculumScheduler: phase-based weight transitions

**Training** (`src/wrinklefree/training/`):
- `fsdp_wrapper.py` - FSDP wrapping with activation checkpointing (ACTIVE)
- `auto_setup.py` - Auto-magic checkpoint resolution + BitNet conversion (ACTIVE)
- `_legacy/trainer.py` - Legacy base Trainer (DEPRECATED - use Lightning)
- `_legacy/continued_pretraining.py` - Legacy ContinuedPretrainingTrainer (DEPRECATED)
- `_legacy/stage1.py` - Stage 1 SubLN insertion (DEPRECATED - use bitnet_arch.auto_convert_if_needed)

**Experimental** (`src/wrinklefree/_experimental/`):
- `moe/` - Mixture of Experts (benchmark-only, not production-ready)
- `tensor_parallel/` - Tensor parallelism utilities (experimental)

**Models** (`src/wrinklefree/models/`):
- `bitlinear.py` - BitLinear layer with STE quantization (ternary weights)
- `subln.py` - SubLN normalization (key BitDistill component)

**Data** (`src/wrinklefree/data/`):
- Imports from `data_handler` package
- `InfluenceTracker` - Training callback for weight updates
- `MixedDataset` - Runtime dataset with dynamic weights

### Key Patterns

**ObjectiveManager Pattern**:
```python
# ObjectiveManager runs all enabled objectives
manager = ObjectiveManager(
    objectives={"continue_pretrain": CPObj(), "dlm": DLMObj()},
    weights={"continue_pretrain": 1.0, "dlm": 0.5},
)

# Preprocess applies DLM masking, stores originals
batch = manager.preprocess_batch(batch)

# Forward computes all losses, returns weighted sum
output = manager(model_outputs, batch)
# output.loss = 1.0 * cp_loss + 0.5 * dlm_loss
```

**Auto-Setup Pattern**:
```python
from wrinklefree.training.auto_setup import auto_setup_model

# Resolves checkpoint (local/GCS/HuggingFace)
# Auto-converts to BitNet if needed
model, tokenizer = auto_setup_model(config, device)
```

### Quantization
- **BitNet 1.58-bit**: Ternary weights {-1, 0, 1} (3 levels, 1.58 bits/weight)

### Configuration
All configs in `configs/` using Hydra:
- `model/` - Model architecture configs (smollm2_135m, qwen3_4b)
- `training/` - Stage-specific training configs (unified, stage2_pretrain)
- `data/` - Dataset configs (default points to data_handler)
- `distributed/` - FSDP/DDP settings (single_gpu, fsdp_multi)

**Key Config Files**:
| Config | Purpose |
|--------|---------|
| `training/base.yaml` | Combined STE+DLM with curriculum |
| `training/stage2_pretrain.yaml` | Legacy Stage 2 |
| `data/default.yaml` | Points to data_handler |

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
from wrinklefree._experimental.moe import create_fake_moe_from_dense, verify_moe_matches_dense

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
uv run python scripts/train_lightning.py model=smollm2_135m training=stage2_pretrain \
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
uv run python scripts/train_lightning.py model=smollm2_135m training=stage2_pretrain \
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
