# WrinkleFree DLM Converter

Convert BitNet 1.58-bit quantized models to Diffusion Language Models (DLMs) for faster parallel inference.

## Overview

This library implements a commercially usable version of the [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) approach for converting autoregressive language models to diffusion-based generation. Key benefits:

- **~2.5x faster inference** via parallel block decoding
- **~500x less training** compared to training DLMs from scratch
- **Preserves BitNet quantization** for memory-efficient deployment
- **Apache 2.0 licensed** for commercial use

## Installation

```bash
# Clone the repository
git clone git@github.com:DeepOpt-com/WrinkleFree-DLM-Converter.git
cd WrinkleFree-DLM-Converter

# Initialize submodules
git submodule update --init

# Install with uv
uv sync
```

## Quick Start

### Convert a BitNet Model

```bash
# Convert using Modal (recommended)
wf-dlm convert -m qwen3_4b -c hf://org/bitnet-qwen3-4b

# Convert locally (requires GPU)
wf-dlm convert -m smollm2_135m -c ./checkpoints/stage2 --backend local

# With custom settings
wf-dlm convert -m qwen3_4b -c hf://org/model \
    --tokens 500000000 \
    --lr 1e-4 \
    --block-size 64
```

### Validate Conversion

```bash
wf-dlm validate -m ./outputs/dlm/qwen3_4b

wf-dlm validate -m ./outputs/dlm/smollm2_135m \
    --prompt "Write a haiku about coding"
```

### Python API

```python
from wf_dlm_converter import convert, validate

# Convert
result = convert(
    model="qwen3_4b",
    checkpoint_path="hf://org/bitnet-checkpoint",
    total_tokens=1_000_000_000,
)

print(f"Output: {result['output_path']}")

# Validate
validation = validate(model_path=result['output_path'])
print(f"Tokens/sec: {validation['tokens_per_second']:.2f}")
```

## How It Works

### Block Diffusion

The conversion modifies the model's attention mechanism to use block-wise causal attention:

```
Standard causal:     Block causal (block_size=2):
[1,0,0,0]           [1,1,0,0]  ← Block 1 sees all of block 1
[1,1,0,0]           [1,1,0,0]
[1,1,1,0]           [1,1,1,1]  ← Block 2 sees blocks 1+2
[1,1,1,1]           [1,1,1,1]
```

This enables parallel generation within each block (32 tokens by default), achieving significant speedups over traditional autoregressive decoding.

### Conversion Pipeline

1. **Load** BitNet checkpoint from WrinkleFree-1.58Quant
2. **Adapt** model with block causal attention masks and noise embeddings
3. **Fine-tune** with ~1B tokens using masked token prediction
4. **Export** in Fast-dLLM compatible format

### Preserving BitLinear

The fine-tuning uses a moderate learning rate (5e-5) to preserve the ternary weight distribution from BitNet quantization while adapting the model for diffusion-based generation.

## Configuration

### Model Configs

Located in `configs/model/`:

- `smollm2_135m.yaml` - Small model for fast iteration
- `qwen3_4b.yaml` - Production-quality model

### Conversion Config

Located in `configs/conversion/finetune.yaml`:

```yaml
total_tokens: 1_000_000_000
learning_rate: 5.0e-5
block_size: 32
num_diffusion_steps: 8
```

### Using Hydra

```bash
# Override config values
uv run python scripts/convert.py \
    model=qwen3_4b \
    conversion.total_tokens=500000000 \
    conversion.learning_rate=1e-4

# Override block diffusion settings
uv run python scripts/convert.py \
    block_diffusion.block_size=64 \
    block_diffusion.num_diffusion_steps=16
```

## Integration

### With WrinkleFree-1.58Quant

The converter expects BitNet checkpoints from the WrinkleFree-1.58Quant training pipeline:

```bash
# After Stage 2 training in 1.58Quant
wf-dlm convert -m qwen3_4b -c /checkpoints/qwen3_4b/stage2
```

### With Fast-dLLM Inference

Converted models are compatible with Fast-dLLM's inference optimizations:

```python
# Use Fast-dLLM for inference
from fast_dllm import FastDLLM

model = FastDLLM.from_pretrained("./outputs/dlm/qwen3_4b")
output = model.generate(
    prompt="Hello, world!",
    max_length=128,
    parallel_decode=True,
)
```

## Modal Deployment

The converter uses Modal for GPU-accelerated training:

```bash
# Deploy with Modal
modal deploy src/wf_dlm_converter/modal/deployer.py

# Run conversion job
modal run src/wf_dlm_converter/modal/deployer.py \
    --model qwen3_4b \
    --checkpoint hf://org/model
```

## License

MIT License (this library)

The Fast-dLLM reference implementation is Apache 2.0 licensed.

## References

- [Fast-dLLM Paper](https://arxiv.org/abs/2512.14067)
- [Fast-dLLM GitHub](https://github.com/NVlabs/Fast-dLLM)
- [WrinkleFree-1.58Quant](https://github.com/DeepOpt-com/WrinkleFree-1.58Quant)
