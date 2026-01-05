# Contributing to Training (wrinklefree)

> Contributor guide for navigating and understanding the training package codebase.

## Quick Orientation

### What This Package Does
PyTorch Lightning-based training pipeline for 1.58-bit (ternary) LLMs with multi-objective support, influence-based data remixing, and cloud deployment integration.

### Dependencies

| Depends On | What For |
|------------|----------|
| `wf-arch` | BitLinear, SubLN, model conversion |
| `wf-data` | MixedDataset, InfluenceTracker |
| pytorch-lightning | Training loop, callbacks, distributed |
| hydra | Configuration management |

| Used By | What For |
|---------|----------|
| `deployer` | Launches training jobs on cloud |
| `eval` | Evaluates trained checkpoints |
| `inference` | Serves trained models |

---

## Codebase Architecture

### Directory Structure

```
src/wrinklefree/
├── lightning/               # PyTorch Lightning integration
│   ├── module.py            # WrinkleFreeLightningModule (main training loop)
│   ├── datamodule.py        # WrinkleFreeDataModule (data loading)
│   └── callbacks.py         # GCS, ZClip, TokenCount, InfluenceTracker, etc.
│
├── objectives/              # Multi-task objectives (core abstractions)
│   ├── base.py              # Objective base class
│   ├── manager.py           # ObjectiveManager (runs all objectives)
│   ├── factory.py           # Creates objectives from config
│   ├── continue_pretrain.py # Standard LM loss
│   ├── dlm.py               # Diffusion LM masking
│   ├── layerwise.py         # Hidden state distillation
│   ├── logits_distill.py    # KL divergence on logits
│   ├── lrc_reconstruction.py # Low-rank correction training
│   └── bitdistill.py        # Combined logits + attention
│
├── training/                # Training utilities
│   ├── auto_setup.py        # Checkpoint resolution + BitNet conversion
│   ├── fsdp_wrapper.py      # FSDP wrapping with activation checkpointing
│   ├── run_naming.py        # W&B run name generation
│   └── muon_patch.py        # MuonClip bug workaround
│
├── teachers/                # Teacher model wrappers for distillation
│   ├── base.py              # Teacher interface
│   ├── local_teacher.py     # HuggingFace model teacher
│   └── vllm_teacher.py      # vLLM-based teacher
│
├── quantization/            # Quantization utilities
│   ├── ste.py               # Straight-Through Estimator
│   ├── weight_quant.py      # Weight quantization functions
│   └── activation_quant.py  # Activation quantization
│
├── _legacy/                 # Deprecated (use Lightning instead)
└── _experimental/           # MoE, TensorParallel (not production)

scripts/
├── train_lightning.py       # Main entry point
└── ...

configs/
├── training/                # Training configs (unified, stage2_pretrain, etc.)
├── model/                   # Model configs (smollm2_135m, qwen3_4b)
├── data/                    # Data config pointer (uses wf_data)
└── distributed/             # FSDP/single_gpu configs
```

### Key Abstractions

| Class | File | Purpose |
|-------|------|---------|
| `WrinkleFreeLightningModule` | `lightning/module.py` | Main training loop, wraps model + ObjectiveManager |
| `ObjectiveManager` | `objectives/manager.py` | Runs multiple objectives, curriculum scheduling |
| `Objective` | `objectives/base.py` | Base class for all objectives |
| `CurriculumScheduler` | `objectives/manager.py` | Phase-based weight transitions |
| `auto_setup_model()` | `training/auto_setup.py` | Checkpoint resolution + BitNet conversion |

---

## Code Flow

### Training Step Flow

```
train_lightning.py
│
├─► Hydra loads config
│
├─► auto_setup_model(config)
│   ├─► Resolve checkpoint (local/GCS/HuggingFace)
│   └─► auto_convert_if_needed() (BitNet conversion)
│
├─► create_objective_manager(config)
│   └─► Instantiate all enabled objectives
│
├─► WrinkleFreeLightningModule(model, objective_manager)
│
└─► pl.Trainer.fit()
    │
    └─► Each training_step():
        │
        ├─► objective_manager.preprocess_batch(batch)
        │   └─► DLM masking if enabled
        │
        ├─► model.forward(batch)
        │
        └─► objective_manager(model_outputs, batch)
            └─► Returns weighted sum of all objective losses
```

### Adding a New Objective

```
1. Create objectives/my_objective.py
│
├─► class MyObjective(Objective):
│       def __call__(self, outputs, batch, teacher_outputs=None):
│           # Compute loss
│           return ObjectiveOutput(loss=loss, metrics={...})
│
2. Register in objectives/factory.py
│
├─► In create_objectives():
│       if cfg.objectives.my_objective.enabled:
│           objectives["my_objective"] = MyObjective(...)
│
3. Add config in configs/training/my_config.yaml
│
└─► objectives:
      my_objective:
        enabled: true
        weight: 0.5
```

### Curriculum Scheduling Flow

```
unified.yaml config:
  curriculum:
    phases:
      - name: warmup (0-10%)
        objectives: {continue_pretrain: 1.0, dlm: 0.0}
      - name: main (10-80%)
        objectives: {continue_pretrain: 1.0, dlm: 0.5}

Runtime:
│
├─► CurriculumScheduler.get_weights(current_step)
│   └─► Returns interpolated weights based on phase
│
└─► ObjectiveManager uses weights to combine losses
```

---

## Entry Points

| Task | Start Here |
|------|------------|
| Modify training loop | `lightning/module.py:training_step()` |
| Add new objective | `objectives/base.py`, then `objectives/my_obj.py` |
| Change checkpoint loading | `training/auto_setup.py:auto_setup_model()` |
| Add training callback | `lightning/callbacks.py` |
| Modify data loading | `lightning/datamodule.py` (wraps wf_data) |
| Change optimizer | `lightning/module.py:configure_optimizers()` |
| Add curriculum phase | `configs/training/base.yaml:curriculum.phases` |

---

## Patterns & Conventions

### Objective Interface Pattern

All objectives implement the same interface:
```python
class MyObjective(Objective):
    def __call__(
        self,
        model_outputs: ModelOutput,
        batch: dict,
        teacher_outputs: Optional[ModelOutput] = None,
    ) -> ObjectiveOutput:
        loss = compute_loss(...)
        return ObjectiveOutput(
            loss=loss,
            metrics={"my_metric": value}
        )

    @property
    def modifies_input(self) -> bool:
        return False  # True if objective modifies batch in preprocess

    def preprocess_batch(self, batch: dict) -> dict:
        return batch  # Override to modify batch before forward
```

### ObjectiveManager Pattern

```python
# ObjectiveManager orchestrates all objectives
manager = ObjectiveManager(objectives, curriculum)

# Preprocess (DLM masking, etc.)
batch = manager.preprocess_batch(batch)

# Forward all objectives
outputs = model(**batch)
result = manager(outputs, batch, teacher_outputs)
# result.loss = weighted sum
# result.metrics = {"objective_name/metric": value, ...}
```

### Auto-Setup Pattern

```python
from wf_train.training.auto_setup import auto_setup_model

# Handles: GCS download, BitNet conversion, checkpoint loading
model, tokenizer = auto_setup_model(config, device)
# Model is ready for training
```

---

## Testing

### Running Tests

```bash
# All tests
uv run --package wf-train pytest packages/training/tests/ -v

# Unit tests only
uv run --package wf-train pytest packages/training/tests/unit/ -v

# Integration tests (require GPU)
uv run --package wf-train pytest packages/training/tests/integration/ -v
```

### Smoke Tests (Cloud)

```bash
cd packages/deployer
source credentials/.env

# Lightning smoke test
sky launch skypilot/smoke_test_lightning.yaml -y --cluster lightning-smoke

# Influence smoke test
sky launch skypilot/smoke_test_influence.yaml -y --cluster influence-smoke

# Check logs
sky logs lightning-smoke
```

---

## Common Tasks

### Adding a New Training Objective

1. Create `objectives/my_objective.py`:
   ```python
   class MyObjective(Objective):
       def __call__(self, outputs, batch, teacher_outputs=None):
           # Your loss computation
           return ObjectiveOutput(loss=loss, metrics={})
   ```

2. Register in `objectives/factory.py:create_objectives()`:
   ```python
   if cfg.objectives.my_objective.enabled:
       objectives["my_objective"] = MyObjective(cfg.objectives.my_objective)
   ```

3. Add config section in your training YAML:
   ```yaml
   objectives:
     my_objective:
       enabled: true
       weight: 0.5
   ```

4. Add to curriculum phases if needed

### Adding a New Callback

1. Create callback class in `lightning/callbacks.py`:
   ```python
   class MyCallback(pl.Callback):
       def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
           # Your logic
   ```

2. Add to trainer in `train_lightning.py`:
   ```python
   callbacks.append(MyCallback())
   ```

### Modifying Checkpoint Loading

1. Edit `training/auto_setup.py:auto_setup_model()`
2. Checkpoint resolution order: local → GCS → HuggingFace
3. BitNet conversion happens automatically if model isn't already BitNet

---

## Gotchas & Tips

- **MuonClip Bug**: The upstream muon-clip package has a hook registration bug. Fixed via `MuonClipInitCallback` in callbacks.py. Don't remove this callback.

- **DLM + shift_labels**: DLM and objectives with `shift_labels=True` are incompatible. DLM stores original labels in `batch["_original_labels"]`.

- **Batch Preprocessing Order**: `ObjectiveManager.preprocess_batch()` runs before model forward. Objectives with `modifies_input=True` get to modify the batch.

- **FSDP Checkpointing**: All ranks must call `save_checkpoint()`. Don't gate checkpoint saves with `if rank == 0`.

- **Auto Batch Size**: `training.auto_batch_size=true` probes GPU memory at startup. Clean `/tmp/checkpoints/` before retrying failed jobs.

- **Config Inheritance**: Training configs can include other configs. Check `defaults:` section in YAML files.

- **Test Integration**: Changes affect deployer, eval, inference. Run smoke tests after significant changes:
  ```bash
  uv run --package wf-train python scripts/train_lightning.py \
    model=smollm2_135m training=base training.max_steps=10
  ```

- **Legacy Code**: `_legacy/` contains deprecated trainers. Use Lightning trainer for all new work.

- **Experimental Code**: `_experimental/` contains MoE and TensorParallel. Not production-ready.
