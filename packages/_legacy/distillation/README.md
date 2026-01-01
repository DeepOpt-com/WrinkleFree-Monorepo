# WrinkleFree Distillation

Knowledge distillation package for quantized LLMs (BitNet, Fairy2, etc.).

## Features

- **BitDistill-style distillation** (logits + attention)
- **Toggle-able attention distillation** for different use cases
- **Multiple teacher backends**: Local HuggingFace models or vLLM servers
- **Influence-based dataset rebalancing** via data_handler

## Installation

```bash
# From monorepo root
uv sync --package wrinklefree-distillation
```

## Usage

```bash
# Distill against original model (default)
uv run --package wrinklefree-distillation python scripts/distill.py \
  student.checkpoint_path=outputs/stage2/checkpoint.pt

# Distill with different teacher
uv run --package wrinklefree-distillation python scripts/distill.py \
  student.checkpoint_path=outputs/stage2/checkpoint.pt \
  teacher.model_name=meta-llama/Llama-3.2-3B

# Logits-only (no attention distillation)
uv run --package wrinklefree-distillation python scripts/distill.py \
  student.checkpoint_path=outputs/stage2/checkpoint.pt \
  distillation=logits_only

# With vLLM teacher
uv run --package wrinklefree-distillation python scripts/distill.py \
  student.checkpoint_path=outputs/stage2/checkpoint.pt \
  teacher.use_vllm=true \
  teacher.vllm_url=http://localhost:8000
```

## Configuration

See `configs/` for available configuration options:

- `distillation/bitdistill.yaml` - Default BitDistill settings
- `distillation/logits_only.yaml` - Logits-only distillation
- `distillation/classification.yaml` - Classification task settings
- `distillation/summarization.yaml` - Generation task settings

## API

```python
from distillation import BitDistillLoss, LocalTeacher, DistillationTrainer

# Create teacher
teacher = LocalTeacher(
    model_name_or_path="HuggingFaceTB/SmolLM2-135M",
    device=torch.device("cuda"),
)

# Create loss
loss_fn = BitDistillLoss(
    lambda_logits=10.0,
    gamma_attention=1e-5,
    temperature=5.0,
)

# Create trainer
trainer = DistillationTrainer(
    student=student_model,
    teacher=teacher,
    train_dataloader=train_loader,
    config=config,
)

# Train
trainer.train()
```

## References

- [BitDistill Paper](https://arxiv.org/abs/2510.13998)
