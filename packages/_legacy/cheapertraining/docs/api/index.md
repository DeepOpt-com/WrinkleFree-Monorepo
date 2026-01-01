# API Reference

API documentation for CheaperTraining.

## Modules

- [Models](models.md) - Model architecture classes
- [Training](training.md) - Training stages and infrastructure
- [Data](data.md) - Data loading and processing
- [Influence](influence.md) - Influence-based data mixture optimization

## Quick Reference

### Creating a Model

```python
from cheapertraining.models import MobileLLM, MobileLLMConfig, get_mobilellm_config

# Using predefined config
config = get_mobilellm_config("950m")
model = MobileLLM(config)

# Custom config
config = MobileLLMConfig(
    vocab_size=128256,
    num_layers=22,
    num_heads=24,
    num_kv_heads=6,
    embed_dim=1536,
    hidden_dim=6144,
    max_seq_len=32768,
)
model = MobileLLM(config)
```

### Running Training

```python
from cheapertraining.training import Trainer, PretrainStage, StageConfig
from cheapertraining.training.optimizer import create_optimizer
from cheapertraining.training.scheduler import create_scheduler

# Create training stage
stage_config = StageConfig(
    name="pretrain_phase1",
    num_steps=100000,
    batch_size_per_gpu=16,
    learning_rate=4e-3,
    weight_decay=0.1,
)
stage = PretrainStage(model, stage_config)

# Run training
trainer = Trainer(stages=[stage])
trainer.run()
```

### Loading Data

```python
from cheapertraining.data import TokenizerWrapper, MixedDataset, PackedDataset

# Initialize tokenizer
tokenizer = TokenizerWrapper("meta-llama/Llama-3.2-1B")

# Create mixed dataset
mixture = [
    {"name": "fineweb", "weight": 0.6, "path": "HuggingFaceFW/fineweb-edu"},
    {"name": "code", "weight": 0.4, "path": "bigcode/starcoderdata"},
]
dataset = MixedDataset(mixture, tokenizer)

# Pack into fixed-length sequences
packed = PackedDataset(dataset, max_length=2048)
```
