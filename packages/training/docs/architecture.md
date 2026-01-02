# System Architecture

## Overview

WrinkleFree Training uses PyTorch Lightning with a multi-objective training system. The architecture centers around the **ObjectiveManager** pattern, which runs multiple training objectives on the same batch.

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    PyTorch Lightning                        │
│  train_lightning.py → WrinkleFreeLightningModule           │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │           ObjectiveManager (multi-task)                 ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              ││
│  │  │ CE Loss  │  │   DLM    │  │ Distill  │  ...         ││
│  │  └──────────┘  └──────────┘  └──────────┘              ││
│  └─────────────────────────────────────────────────────────┘│
│         ↓                                                   │
│  Callbacks: BatchSizeFinder, GCS, ZClip, InfluenceTracker  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Lightning Module (`lightning/`)

| File | Purpose |
|------|---------|
| `module.py` | `WrinkleFreeLightningModule` - wraps model + ObjectiveManager |
| `datamodule.py` | `WrinkleFreeDataModule` - wraps dataloaders from data_handler |
| `callbacks.py` | GCS upload, ZClip, TokenCount, InfluenceTracker, MuonClipInit |

### 2. ObjectiveManager (`objectives/`)

The ObjectiveManager runs multiple objectives on the same batch and combines their losses:

```python
# Example usage
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

**Available Objectives:**

| Objective | File | Purpose |
|-----------|------|---------|
| `continue_pretrain` | `continue_pretrain.py` | Next-token prediction (CE loss) |
| `dlm` | `dlm.py` | Diffusion Language Model masking |
| `layerwise` | `layerwise.py` | Hidden state alignment with teacher |
| `logits_distill` | `logits_distill.py` | KL divergence on teacher logits |
| `attention_distill` | `attention_distill.py` | Attention pattern matching |
| `bitdistill` | `bitdistill.py` | Combined logits + attention distillation |
| `tcs_distill` | `tcs_distill.py` | Target Concrete Score for DLM |
| `lrc_reconstruction` | `lrc_reconstruction.py` | Low-rank correction training |

### 3. Curriculum Scheduler

Phase-based weight transitions for objectives:

```yaml
# configs/training/unified.yaml
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

### 4. Model Architecture

**BitLinear Layer** (`models/bitlinear.py`):
- Ternary weight quantization {-1, 0, 1}
- 8-bit per-token activation quantization
- Straight-Through Estimator (STE) for gradient computation

**SubLN** (`models/subln.py`):
- RMSNorm inserted before output projections
- Stabilizes training for quantization

**Integration with bitnet_arch package**:
- `BitLinear`, `BitLinearLRC`, `SubLN` layers
- Auto-conversion of HuggingFace models to BitNet

### 5. Quantization (`quantization/`)

| File | Purpose |
|------|---------|
| `weight_quant.py` | Ternary weight quantization |
| `activation_quant.py` | 8-bit per-token activation quantization |
| `ste.py` | Straight-through estimator |
| `lambda_warmup.py` | Gradual quantization warmup |

**Quantization Warmup**:
```
Steps 0-1000:    lambda = step/1000 (linear warmup)
Steps 1000+:     lambda = 1.0 (full quantization)
```

### 6. Data Integration

Data loading is handled by the **data_handler** package:

```yaml
# configs/data/default.yaml
config_name: mixed_pretrain  # Loads from data_handler
```

**Influence-Based Data Remixing**:
- `InfluenceTrackerCallback` wraps `data_handler.influence.InfluenceTracker`
- Dynamically adjusts dataset weights based on influence scores
- Updates weights at configurable intervals

### 7. Training Configuration

| Directory | Purpose |
|-----------|---------|
| `training/auto_setup.py` | Auto-magic checkpoint resolution + BitNet conversion |
| `training/fsdp_wrapper.py` | FSDP wrapping with activation checkpointing |
| `training/trainer.py` | Legacy trainer (prefer Lightning) |

## Directory Structure

```
src/wrinklefree/
├── lightning/           # PyTorch Lightning integration
│   ├── module.py        # WrinkleFreeLightningModule
│   ├── datamodule.py    # WrinkleFreeDataModule
│   └── callbacks.py     # Training callbacks
├── objectives/          # Multi-task objectives
│   ├── manager.py       # ObjectiveManager
│   ├── factory.py       # Creates ObjectiveManager from config
│   ├── curriculum.py    # Phase-based weight transitions
│   ├── continue_pretrain.py
│   ├── dlm.py
│   ├── layerwise.py
│   ├── logits_distill.py
│   ├── attention_distill.py
│   ├── bitdistill.py
│   ├── tcs_distill.py
│   └── lrc_reconstruction.py
├── training/            # Training utilities
│   ├── auto_setup.py    # Checkpoint resolution
│   ├── fsdp_wrapper.py  # FSDP integration
│   └── trainer.py       # Legacy trainer
├── models/              # Model components
│   ├── bitlinear.py     # BitLinear layer
│   ├── subln.py         # SubLN normalization
│   ├── attention.py     # Multi-head attention
│   └── ffn.py           # Feed-forward network
├── quantization/        # Quantization logic
│   ├── weight_quant.py
│   ├── activation_quant.py
│   ├── ste.py
│   └── lambda_warmup.py
├── teachers/            # Teacher models for distillation
│   ├── local_teacher.py
│   └── vllm_teacher.py
├── serving/             # Inference export
│   ├── converter.py     # GGUF conversion
│   └── bitnet_wrapper.py
├── _experimental/       # Not production-ready
│   ├── moe/             # Mixture of Experts
│   ├── tensor_parallel/
│   └── fp8/
└── utils/               # Utilities
    ├── run_fingerprint.py
    ├── run_manager.py
    └── audit_logger.py
```

## Monorepo Dependencies

| Package | Relationship |
|---------|--------------|
| `data_handler` | Data loading, influence optimization |
| `bitnet_arch` | BitNet layers and model conversion |
| `deployer` | Cloud deployment (launches training jobs) |
| `inference` | Serves trained models |
| `eval` | Evaluates trained models |

## LRC Calibration

Low-Rank Correction (LRC) adds trainable low-rank matrices to correct quantization errors:

```
output = W_quant @ Q_a(X) + U @ V^T @ X
         ─────────────────   ───────────
         Frozen quantized     Trainable LRC
         path                 correction
```

**Training**:
1. Convert BitLinear → BitLinearLRC (adds U, V matrices)
2. Freeze ALL parameters except U, V
3. Train using hidden state matching loss vs. fp16 teacher

## FSDP Multi-GPU Training

When using `distributed=fsdp_multi`:

- Uses `muon_fsdp2.Muon` optimizer (not `muon-clip` which is incompatible with FSDP)
- All ranks must participate in collective operations (save_checkpoint, eval sync)
- Use `drop_last=True` in dataloaders to prevent batch count mismatches
