# WrinkleFree

Meta repository for WrinkleFree projects - efficient LLM training and quantization.

## Quick Start

This repository uses [meta](https://github.com/mateodelnorte/meta) to manage multiple related repositories.

### Prerequisites

```bash
npm install -g meta
```

### Clone All Repositories

```bash
meta git clone git@github.com:DeepOpt-com/WrinkleFree.git
cd WrinkleFree
```

Or if you already have this repo cloned:

```bash
meta git update
```

## Projects

| Repository | Description |
|------------|-------------|
| [WrinkleFree-1.58Quant](https://github.com/DeepOpt-com/WrinkleFree-1.58Quant) | 1.58-bit (ternary) LLM training using BitDistill approach |
| [WrinkleFree-CheaperTraining](https://github.com/DeepOpt-com/WrinkleFree-CheaperTraining) | Cost-efficient training strategies |

## Common Commands

```bash
# Check status across all repos
meta git status

# Pull latest changes in all repos
meta git pull

# Run a command in all repos
meta exec "git branch"
```

## Development

Each child repository has its own development setup. See individual READMEs for details.

### WrinkleFree-1.58Quant

Training framework for 1.58-bit (ternary) LLM models using:
- BitDistill approach from [arxiv.org/abs/2510.13998](https://arxiv.org/abs/2510.13998)
- Microsoft BitNet for inference
- FSDP for distributed training

```bash
cd WrinkleFree-1.58Quant
uv sync
uv run python scripts/train.py model=llama_7b training=stage3_distill
```
