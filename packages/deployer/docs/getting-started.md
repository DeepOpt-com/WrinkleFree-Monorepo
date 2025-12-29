# Getting Started

Train 1.58-bit LLMs in 5 minutes.

## 1. Install

```bash
git clone https://github.com/your-org/WrinkleFree-Deployer
cd WrinkleFree-Deployer
uv sync
```

## 2. Set Credentials

```bash
cp .env.example .env
# Edit .env - add your WANDB_API_KEY (required)
```

The CLI auto-loads `.env` - no need to export.

## 3. Setup Nebius (Recommended - $1.99/hr H100)

```bash
pip install "skypilot[nebius]"
sky check nebius
```

## 4. Train

```bash
uv run wf train -m qwen3_4b -s 1.9
```

## Scale Profiles

| Scale | GPUs | Use Case |
|-------|------|----------|
| dev | 1x H100 | Testing |
| small | 1x H100 | Single GPU |
| xlarge | 8x H100 | Production |

Note: Nebius only has 1 or 8 GPU configs. For 2/4 GPU use RunPod.

## Training Stages

| Stage | Purpose | Cost (Nebius) |
|-------|---------|---------------|
| 1 | Model conversion | ~$0 |
| 1.9 | Distillation | $5-25 |
| 2 | Pretraining | $30-120 |
| 3 | Fine-tuning | $5-25 |

## Complete Workflow

```bash
uv run wf train -m qwen3_4b -s 1      # Stage 1: Model conversion
uv run wf train -m qwen3_4b -s 1.9    # Stage 1.9: Distillation
uv run wf train -m qwen3_4b -s 2 --scale xlarge  # Stage 2: Pretraining
```

## Monitor

```bash
uv run wf runs                        # List runs
uv run wf logs wrinklefree-train      # View logs
uv run wf logs wrinklefree-train -f   # Follow logs
uv run wf cancel wrinklefree-train    # Cancel job
uv run wf wandb-status                # Check W&B metrics
```

## Troubleshooting

**"WANDB_API_KEY not set"** - Check `.env` file has `WANDB_API_KEY=your_key`

**"sky: command not found"** - Run: `pip install "skypilot[nebius]"`

## 5. Convert to DLM (Optional - 2.5x Faster Inference)

After training, convert your BitNet model to a Diffusion LLM for ~2.5x faster inference via parallel block decoding.

```bash
# Convert Stage 2 checkpoint to DLM
wf dlm -m qwen3_4b -s hf://org/bitnet-checkpoint

# With custom settings
wf dlm -m qwen3_4b -s gs://bucket/checkpoint conversion.total_tokens=500000000

# Monitor
wf logs wf-dlm-train
```

### Local Development

```bash
cd ../WrinkleFree-DLM-Converter
uv sync

# Run training locally (uses Hydra configs)
uv run python scripts/train_dlm.py model=qwen3_4b source.path=hf://org/checkpoint
```

### Validate & Use

```bash
# Validate conversion
wf-dlm validate -m ./outputs/dlm/qwen3_4b --prompt "Write a haiku"
```

```python
from fast_dllm.generation import batch_sample

model = AutoModelForCausalLM.from_pretrained("outputs/dlm/qwen3_4b")
tokenizer = AutoTokenizer.from_pretrained("outputs/dlm/qwen3_4b")

# Generate with block diffusion (2.5x faster)
outputs = batch_sample(model, tokenizer, prompts=["Hello!"], block_size=32)
```

See [WrinkleFree-DLM-Converter](../../WrinkleFree-DLM-Converter/README.md) for full documentation.

## Next Steps

- [Training Guide](training.md) - Deep dive on training stages
- [Serving Guide](serving.md) - Deploy for inference
- [Library API](library-api.md) - Python API reference
