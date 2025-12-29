# Usage Guide

This guide covers common usage patterns for the DLM Converter.

## Basic Conversion

### Using the CLI

```bash
# Convert with default settings
wf-dlm convert -m qwen3_4b -c hf://org/bitnet-checkpoint

# View available options
wf-dlm convert --help

# Show supported models and settings
wf-dlm info
```

### Using the Python API

```python
from wf_dlm_converter import convert, validate

# Basic conversion
result = convert(
    model="qwen3_4b",
    checkpoint_path="hf://org/bitnet-checkpoint",
)

# With custom settings
result = convert(
    model="smollm2_135m",
    checkpoint_path="./checkpoints/stage2",
    total_tokens=500_000_000,
    learning_rate=1e-4,
    block_size=64,
)
```

## Checkpoint Sources

The converter supports multiple checkpoint sources:

### HuggingFace Hub

```bash
# Using hf:// prefix
wf-dlm convert -m qwen3_4b -c hf://organization/model-name

# Direct repo ID also works
wf-dlm convert -m qwen3_4b -c organization/model-name
```

### Local Path

```bash
# Absolute path
wf-dlm convert -m qwen3_4b -c /path/to/checkpoint

# Relative path
wf-dlm convert -m qwen3_4b -c ./checkpoints/stage2
```

### Modal Volume

When running on Modal, checkpoints can be loaded from volumes:

```bash
# From checkpoints volume
wf-dlm convert -m qwen3_4b -c /checkpoints/qwen3_4b/stage2
```

## Custom Training Settings

### Adjusting Token Budget

```bash
# Smaller budget for testing
wf-dlm convert -m smollm2_135m -c ./ckpt --tokens 100000000

# Larger budget for better quality
wf-dlm convert -m qwen3_4b -c ./ckpt --tokens 2000000000
```

### Learning Rate

```bash
# Higher LR for faster convergence (may affect quantization)
wf-dlm convert -m qwen3_4b -c ./ckpt --lr 1e-4

# Lower LR for better preservation of BitLinear
wf-dlm convert -m qwen3_4b -c ./ckpt --lr 1e-5
```

### Block Diffusion Parameters

```bash
# Larger blocks (faster inference, may affect quality)
wf-dlm convert -m qwen3_4b -c ./ckpt --block-size 64

# More diffusion steps (better quality, slower inference)
wf-dlm convert -m qwen3_4b -c ./ckpt --steps 16
```

## Validation

### Basic Validation

```bash
# Validate with default prompt
wf-dlm validate -m ./outputs/dlm/qwen3_4b

# Custom test prompt
wf-dlm validate -m ./outputs/dlm/qwen3_4b \
    --prompt "Explain quantum computing in simple terms"
```

### Validation Metrics

The validation command reports:
- **Tokens/second**: Generation speed
- **New tokens**: Number of tokens generated
- **Elapsed time**: Total generation time

```python
result = validate(model_path="./outputs/dlm/qwen3_4b")
print(f"Speed: {result['tokens_per_second']:.2f} tok/s")
```

## Job Management

### View Logs

```bash
# View logs for a completed job
wf-dlm logs dlm-convert-qwen3_4b-abc123

# Stream logs from running job
wf-dlm logs dlm-convert-qwen3_4b-abc123 --follow
```

### Cancel Jobs

```bash
wf-dlm cancel dlm-convert-qwen3_4b-abc123
```

## Using with Hydra

For more complex configurations, use Hydra directly:

```bash
# Override multiple settings
uv run python scripts/convert.py \
    model=qwen3_4b \
    source.path=hf://org/checkpoint \
    conversion.total_tokens=500000000 \
    conversion.learning_rate=1e-4 \
    block_diffusion.block_size=64

# Use different conversion config
uv run python scripts/convert.py \
    model=smollm2_135m \
    conversion=quick_test
```

## Local vs Modal

### Local Execution

Requires a GPU (CUDA):

```bash
wf-dlm convert -m smollm2_135m -c ./ckpt --backend local
```

### Modal Execution (Recommended)

Runs on Modal's H100 GPUs:

```bash
wf-dlm convert -m qwen3_4b -c hf://org/model --backend modal
```

## Integration with WrinkleFree

### After 1.58Quant Training

```bash
# After Stage 2 training completes
wf-dlm convert -m qwen3_4b \
    -c /checkpoints/qwen3_4b/stage2/final
```

### With WrinkleFree-Deployer

```python
from wf_deployer import train
from wf_dlm_converter import convert

# Train BitNet model
train_result = train(model="qwen3_4b", stage=2)

# Convert to DLM
dlm_result = convert(
    model="qwen3_4b",
    checkpoint_path=train_result["checkpoint_path"],
)
```

## Troubleshooting

### Out of Memory

If you encounter OOM errors:

```bash
# Use smaller batch size
uv run python scripts/convert.py \
    conversion.batch_size=4 \
    conversion.gradient_accumulation_steps=16

# Enable gradient checkpointing
uv run python scripts/convert.py \
    conversion.compute.gradient_checkpointing=true
```

### Slow Convergence

If training loss doesn't decrease:

```bash
# Increase learning rate
wf-dlm convert -m qwen3_4b -c ./ckpt --lr 1e-4

# Check if checkpoint is valid
wf-dlm validate -m ./source_checkpoint
```

### BitLinear Not Preserved

If converted model loses ternary properties:

```bash
# Use lower learning rate
wf-dlm convert -m qwen3_4b -c ./ckpt --lr 1e-5

# Reduce training tokens
wf-dlm convert -m qwen3_4b -c ./ckpt --tokens 500000000
```
