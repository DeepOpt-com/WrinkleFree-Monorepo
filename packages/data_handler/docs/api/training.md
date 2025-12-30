# Training API

Training infrastructure and stages for CheaperTraining.

## StageConfig

Configuration for training stages.

```python
from cheapertraining.training.stages import StageConfig

config = StageConfig(
    name="pretrain_phase1",
    num_steps=100000,
    batch_size_per_gpu=16,
    learning_rate=4e-3,
    weight_decay=0.1,
    max_grad_norm=1.0,
    warmup_steps=2000,
    lr_decay_ratio=0.1,
    gradient_accumulation_steps=1,
    dtype="bfloat16",
)
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | - | Stage identifier |
| `num_steps` | int | - | Total training steps |
| `batch_size_per_gpu` | int | - | Batch size per GPU |
| `learning_rate` | float | - | Peak learning rate |
| `weight_decay` | float | 0.1 | Weight decay coefficient |
| `max_grad_norm` | float | 1.0 | Gradient clipping threshold |
| `warmup_steps` | int | 0 | LR warmup steps |
| `warmup_ratio` | float | 0.0 | Warmup as ratio of total |
| `lr_decay_ratio` | float | 0.1 | Final LR ratio |
| `gradient_accumulation_steps` | int | 1 | Accumulation steps |
| `dtype` | str | "bfloat16" | Training data type |

## TrainingStage (Base Class)

Abstract base class for all training stages.

```python
from cheapertraining.training.stages import TrainingStage

class CustomStage(TrainingStage):
    def train_step(self, batch):
        # Implement custom training logic
        loss = self.model(batch["input_ids"], labels=batch["labels"]).loss
        return {"loss": loss}
```

### Methods

#### `train_step(batch) -> dict`

Abstract method. Implement training logic for a single batch.

**Parameters:**
- `batch` (dict): Batch of data from dataloader

**Returns:**
- `dict`: Dictionary with at least `"loss"` key

#### `train_loop(dataloader, optimizer, scheduler)`

Main training loop with logging and checkpointing.

#### `checkpoint(path)`

Save model and optimizer state.

#### `load_checkpoint(path)`

Load model and optimizer state.

## PretrainStage

Standard causal language modeling pretraining.

```python
from cheapertraining.training.stages import PretrainStage, StageConfig

config = StageConfig(
    name="pretrain",
    num_steps=100000,
    batch_size_per_gpu=16,
    learning_rate=4e-3,
)

stage = PretrainStage(model, config)
stage.train_loop(dataloader, optimizer, scheduler)
```

### Loss Function

Standard cross-entropy loss for next-token prediction:

```
L = -sum(log P(x_t | x_{<t}))
```

### Metrics

- `loss`: Cross-entropy loss
- `accuracy`: Token prediction accuracy
- `perplexity`: exp(loss)

## MidtrainStage

Knowledge distillation from teacher model.

```python
from cheapertraining.training.stages import MidtrainStage, StageConfig

config = StageConfig(
    name="midtrain",
    num_steps=50000,
    batch_size_per_gpu=4,
    learning_rate=3.6e-4,
)

stage = MidtrainStage(
    model=student_model,
    config=config,
    teacher=teacher_model,
    temperature=1.0,
    alpha=1.0,  # Weight for KD loss (1.0 = pure KD)
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `teacher` | nn.Module | - | Frozen teacher model |
| `temperature` | float | 1.0 | Softmax temperature |
| `alpha` | float | 1.0 | KD loss weight (vs CE loss) |

### Loss Function

KL divergence with temperature scaling:

```
L_KD = T² × KL(P_teacher(T) || P_student(T))
L_total = α × L_KD + (1-α) × L_CE
```

### Metrics

- `kd_loss`: Knowledge distillation loss
- `ce_loss`: Cross-entropy loss
- `student_accuracy`: Student prediction accuracy
- `teacher_agreement`: Student-teacher prediction agreement

## PosttrainSFTStage

Supervised fine-tuning with prompt masking.

```python
from cheapertraining.training.stages import PosttrainSFTStage, StageConfig

config = StageConfig(
    name="sft",
    num_steps=10000,
    batch_size_per_gpu=8,
    learning_rate=1e-5,
)

stage = PosttrainSFTStage(model, config)
```

### Loss Function

Cross-entropy only on completion tokens (prompt tokens masked with -100):

```
L = -sum(log P(y_t | x, y_{<t}))  # Only for completion tokens
```

### Metrics

- `loss`: Masked loss
- `accuracy`: Completion token accuracy
- `perplexity`: exp(loss)
- `completion_ratio`: Fraction of non-masked tokens

## Trainer

Orchestrates multi-stage training pipeline.

```python
from cheapertraining.training import Trainer

trainer = Trainer(
    stages=[pretrain_stage, midtrain_stage, sft_stage],
    checkpoint_dir="/path/to/checkpoints",
    checkpoint_interval=1000,
)

trainer.run()
```

### Methods

#### `run()`

Execute all training stages sequentially.

#### `save_checkpoint(path)`

Save full training state.

#### `load_checkpoint(path)`

Resume from checkpoint.

## Optimizers

### create_optimizer

Factory function for optimizers with proper weight decay handling.

```python
from cheapertraining.training.optimizer import create_optimizer

optimizer = create_optimizer(
    model=model,
    optimizer_type="adamw",
    learning_rate=4e-3,
    weight_decay=0.1,
    betas=(0.9, 0.95),
)
```

### Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `optimizer_type` | str | "adam", "adamw", "sgd" | Optimizer type |
| `learning_rate` | float | - | Learning rate |
| `weight_decay` | float | - | Weight decay |
| `betas` | tuple | - | Adam beta parameters |

Weight decay is automatically excluded from biases and layer norms.

## Schedulers

### LinearWarmupLinearDecay

Linear warmup followed by linear decay.

```python
from cheapertraining.training.scheduler import LinearWarmupLinearDecay

scheduler = LinearWarmupLinearDecay(
    optimizer=optimizer,
    warmup_steps=2000,
    total_steps=100000,
    min_lr_ratio=0.1,
)
```

### LinearWarmupLinearDecayToZero

Linear warmup followed by decay to zero.

```python
from cheapertraining.training.scheduler import LinearWarmupLinearDecayToZero

scheduler = LinearWarmupLinearDecayToZero(
    optimizer=optimizer,
    warmup_steps=0,  # No warmup
    total_steps=50000,
)
```

### CosineWarmup

Linear warmup followed by cosine decay.

```python
from cheapertraining.training.scheduler import CosineWarmup

scheduler = CosineWarmup(
    optimizer=optimizer,
    warmup_steps=2000,
    total_steps=100000,
    min_lr_ratio=0.1,
)
```

### create_scheduler

Factory function for schedulers.

```python
from cheapertraining.training.scheduler import create_scheduler

scheduler = create_scheduler(
    optimizer=optimizer,
    scheduler_type="linear_warmup_linear_decay",
    warmup_steps=2000,
    total_steps=100000,
    min_lr_ratio=0.1,
)
```

## TrainingMetrics

Tracks and logs training metrics.

```python
from cheapertraining.training.trainer import TrainingMetrics

metrics = TrainingMetrics()
metrics.update(loss=2.5, accuracy=0.45, grad_norm=1.2)
metrics.log(step=1000)
```

### Methods

- `update(**kwargs)`: Update metrics
- `log(step)`: Log to console/wandb
- `reset()`: Reset running averages
