# WrinkleFree-DLM-Converter

Convert BitNet 1.58-bit models to Diffusion LLMs (DLMs) for faster parallel inference using Fast-dLLM v2.

**For detailed documentation, see `docs/`.**

## Quick Reference

```bash
# Via Deployer CLI (recommended)
cd ../WrinkleFree-Deployer
wf dlm -m qwen3_4b -s hf://org/checkpoint

# Or run locally with Hydra
cd ../WrinkleFree-DLM-Converter
uv run python scripts/train_dlm.py model=qwen3_4b source.path=hf://org/checkpoint

# With overrides
uv run python scripts/train_dlm.py model=smollm2_135m conversion.total_tokens=100000000
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/train_dlm.py` | **Training recipe** - Fast-dLLM v2 SFT training |
| `src/wf_dlm_converter/core.py` | High-level API: convert(), validate() |
| `src/wf_dlm_converter/cli.py` | CLI commands |
| `extern/Fast-dLLM/` | Reference implementation (submodule) |

## Fast-dLLM v2 Training (arXiv:2509.26328)

**Key Insight**: Training uses **SFT with response-only loss** - loss is computed ONLY on assistant responses, not prompts. Block diffusion (complementary masks, token shift) happens at **inference time**.

### Training Recipe

| Parameter | Default | Notes |
|-----------|---------|-------|
| Training objective | **SFT** | Loss only on assistant responses (prompts masked with -100) |
| Dataset | nvidia/Llama-Nemotron-Post-Training-Dataset | 3.9M conversations, CC-BY-4.0 |
| Total tokens | 1B | ~500x less than training DLM from scratch |
| Learning rate | **2e-5** | Fast-dLLM v2 default |
| Warmup | **3%** | 3% of total steps |
| Sequence length | **512** | Fast-dLLM v2 default |
| Block size (bd_size) | 32 | Tokens per block for parallel generation |
| Scheduler | constant_with_warmup | Hold LR after warmup |

### What Gets Added to Model

1. **Mask token** `|<MASK>|` added to tokenizer vocabulary
2. **bd_size** stored in model config (e.g., `model.config.bd_size = 32`)
3. **Response tokens padded** to multiples of bd_size with mask token

### Output Format

```
output/model_name/
├── config.json           # HF config with bd_size
├── model.safetensors     # Model weights
├── tokenizer.json        # Tokenizer with |<MASK>| token
├── tokenizer_config.json
└── dlm_config.json       # Fast-dLLM v2 metadata
```

`dlm_config.json` contains:
```json
{
  "bd_size": 32,
  "mask_token": "|<MASK>|",
  "mask_token_id": 151665,
  "training_method": "fast-dllm-v2-sft"
}
```

## Inference with Fast-dLLM v2

After training, use the model with Fast-dLLM v2's `batch_sample()` for 2.5x faster inference:

```python
from fast_dllm.generation import batch_sample

# Load converted model
model = AutoModelForCausalLM.from_pretrained("output/bitnet_2b")
tokenizer = AutoTokenizer.from_pretrained("output/bitnet_2b")

# Generate with block diffusion (2.5x faster)
outputs = batch_sample(
    model, tokenizer,
    prompts=["Hello, world!"],
    block_size=32,
    threshold=0.95,
)
```

## Deployment via WrinkleFree-Deployer

Training runs on Nebius via SkyPilot using the `wf` CLI:

```bash
# From WrinkleFree-Deployer directory
cd ../WrinkleFree-Deployer

# Launch DLM training
wf dlm -m qwen3_4b -s hf://org/checkpoint

# With custom scale (GPU count)
wf dlm -m qwen3_4b -s hf://org/checkpoint --scale large  # 4x H100

# Monitor
wf logs wf-dlm-train

# Checkpoints uploaded to GCS
gsutil ls gs://wrinklefree-checkpoints/dlm/
```

## Supported Models

- `bitnet_2b` - BitNet 2B (uses bf16 variant for training)
- `smollm2_135m` - SmolLM2 135M (fast iteration)
- `qwen3_4b` - Qwen3 4B (production quality)

## Local Development

```bash
# Install dependencies
uv sync

# Run training locally (uses Hydra configs)
uv run python scripts/train_dlm.py model=smollm2_135m source.path=Qwen/Qwen2.5-0.5B-Instruct

# With custom settings
uv run python scripts/train_dlm.py model=qwen3_4b conversion.total_tokens=10000000

# With W&B logging
WANDB_API_KEY=xxx uv run python scripts/train_dlm.py model=qwen3_4b source.path=...
```
