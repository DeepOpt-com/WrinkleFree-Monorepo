# CheaperTraining

Efficient LLM training library implementing the [MobileLLM-R1](https://arxiv.org/abs/2509.24945) training methodology. This is a **clean room implementation** based solely on the paper description, designed for commercial use.

## Overview

CheaperTraining provides a complete training pipeline for language models from pretraining through post-training, optimized for efficiency at scale (up to 671B parameters). It serves as the continued pre-training foundation for the sister [WrinkleFree-1.58Quant](../training) BitNet fine-tuning library.

## Features

- **Multi-stage Training Pipeline**: Pretraining, mid-training with knowledge distillation, post-training SFT
- **Scalable**: FSDP2, tensor parallelism, pipeline parallelism for models up to 671B parameters
- **Efficient**: FP8/BF16 mixed precision, activation checkpointing, sequence packing
- **Modular**: Each training stage can run independently
- **Commercial-Friendly**: Apache 2.0 license, uses only commercially-licensed datasets

## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

```bash
# Run pretraining phase 1
cheapertraining-pretrain training=pretrain_phase1 model=mobilellm_950m

# Run with custom config
cheapertraining-train \
    model=mobilellm_950m \
    training=pretrain_phase1 \
    data=pretrain_phase1_mix \
    distributed=fsdp2
```

## Training Pipeline

| Phase | Description | Key Details |
|-------|-------------|-------------|
| Pretrain Phase 1 | Initial pretraining | 2T tokens, diverse data mix |
| Pretrain Phase 2 | Math-focused pretraining | 2T tokens, increased math/code |
| Mid-training | Knowledge distillation | KL divergence from teacher model |
| Post-train General | Instruction tuning | General SFT with chat data |
| Post-train Reasoning | Reasoning enhancement | Math/code/science reasoning |

## Project Structure

```
packages/cheapertraining/
├── src/cheapertraining/          # Main source code
│   ├── models/                   # Model architecture
│   │   ├── config.py                # MobileLLM configs (140M-671B)
│   │   ├── mobilellm.py             # Main MobileLLM model class
│   │   ├── transformer.py           # TransformerBlock & TransformerDecoder
│   │   └── attention.py             # MultiHeadAttention, RMSNorm, RoPE, FeedForward
│   │
│   ├── training/                 # Training infrastructure
│   │   ├── trainer.py               # Main Trainer orchestrator
│   │   ├── optimizer.py             # Adam/AdamW/SGD with param groups
│   │   ├── scheduler.py             # LR schedulers (linear, cosine warmup)
│   │   └── stages/
│   │       ├── base.py              # TrainingStage abstract base class
│   │       ├── pretrain.py          # Pretraining (next-token prediction)
│   │       ├── midtrain.py          # Mid-training (KL divergence distillation)
│   │       └── posttrain.py         # Post-training SFT (masked prompt loss)
│   │
│   ├── data/                     # Data loading & processing
│   │   ├── tokenization.py          # TokenizerWrapper (LLaMA3.2-1B)
│   │   ├── mixing.py                # MixedDataset, PackedDataset
│   │   └── datasets/
│   │       ├── pretrain.py          # PretrainDataset with packing
│   │       └── sft.py               # SFTDataset with chat templates
│   │
│   ├── distillation/             # Knowledge distillation
│   │   ├── teacher.py               # TeacherWrapper, CachedTeacher
│   │   └── losses.py                # KL divergence losses
│   │
│   ├── distributed/              # Distributed training
│   │   ├── fsdp2.py                 # FSDP2 wrapping & checkpointing
│   │   └── parallelism.py           # Tensor/pipeline parallelism
│   │
│   ├── precision/                # Mixed precision (BF16/FP8)
│   └── utils/                    # Utilities
│
├── configs/                      # Hydra configuration files
│   ├── config.yaml                  # Main config with defaults
│   ├── model/                       # Model configs (140m, 360m, 950m, 7b, 70b, 671b)
│   ├── training/                    # Training stage configs
│   ├── data/                        # Data mixture configs
│   ├── distributed/                 # Distributed setup configs
│   └── optimizer/                   # Optimizer configs
│
├── scripts/
│   └── train.py                  # Main entry point
│
├── tests/
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│
├── docs/                         # Documentation
│   ├── architecture.md              # High-level system design
│   ├── experiments.md               # Reproduction guide
│   └── api/                         # API reference
│
├── pyproject.toml                # Project dependencies
├── README.md                     # This file
└── LICENSE                       # Apache 2.0
```

## Configuration

Uses [Hydra](https://hydra.cc/) for configuration management:

```yaml
# configs/config.yaml
defaults:
  - model: mobilellm_950m
  - training: pretrain_phase1
  - data: pretrain_phase1_mix
  - distributed: fsdp2
```

## Integration with WrinkleFree-1.58Quant

This library is designed to work with the BitNet 1.58-bit fine-tuning pipeline:

1. **CheaperTraining**: Efficient continued pre-training
2. **WrinkleFree-1.58Quant**: SubLN insertion and 1.58-bit quantization

Checkpoints from CheaperTraining can be directly loaded into WrinkleFree-1.58Quant's Stage 1 (SubLN insertion).

## References

This implementation is based on publicly available research papers and documentation:

### Primary Reference
- **MobileLLM-R1**: Zhao et al., "MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes" ([arXiv:2509.24945](https://arxiv.org/abs/2509.24945))

### Training Methodology
- **TorchTitan**: PyTorch native distributed training patterns ([GitHub](https://github.com/pytorch/torchtitan))
- **DeepSeek-V3**: Large-scale training efficiency techniques ([arXiv:2412.19437](https://arxiv.org/html/2412.19437v1))

### Datasets (Commercial-Friendly)
- **FineWeb-Edu**: Educational web content, ODC-By license ([HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu))
- **OpenWebMath**: Mathematical web content, ODC-By license ([HuggingFace](https://huggingface.co/datasets/open-web-math/open-web-math))
- **OpenMathReasoning**: NVIDIA, permissive license ([HuggingFace](https://huggingface.co/datasets/nvidia/OpenMathReasoning))
- **OpenCodeReasoning**: NVIDIA, CC BY 4.0 ([HuggingFace](https://huggingface.co/datasets/nvidia/OpenCodeReasoning))

### Architecture References
- **LLaMA**: Touvron et al., foundation architecture patterns
- **Grouped Query Attention**: Ainslie et al., efficient attention mechanism
- **RoPE**: Su et al., rotary position embeddings

### Clean Room Implementation
This is a clean room implementation following the methodology described in [Wikipedia: Clean room design](https://en.wikipedia.org/wiki/Clean_room_design). The implementation is based solely on the paper descriptions and publicly documented techniques, without reference to any proprietary source code.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Citation

If you use this library, please cite the original MobileLLM-R1 paper:

```bibtex
@article{zhao2025mobilellmr1,
  title={MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes},
  author={Zhao, Changsheng and Chang, Ernie and Liu, Zechun and others},
  journal={arXiv preprint arXiv:2509.24945},
  year={2025}
}
```
