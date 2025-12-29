# Experiments

Guide to reproducing the training experiments and results from the MobileLLM-R1 methodology.

## Overview

The training pipeline consists of five sequential stages:

| Stage | Tokens | Purpose |
|-------|--------|---------|
| Pretrain Phase 1 | 2T | Initial language modeling |
| Pretrain Phase 2 | 2T | Math/code focused |
| Mid-training | 100B | Knowledge distillation |
| Post-train General | - | Instruction following |
| Post-train Reasoning | - | Long-form reasoning |

## Prerequisites

### Hardware Requirements

| Model | Minimum GPUs | Recommended |
|-------|--------------|-------------|
| 140M  | 1× A100 40GB | 1× A100 80GB |
| 360M  | 1× A100 80GB | 2× A100 80GB |
| 950M  | 2× A100 80GB | 4× A100 80GB |
| 7B    | 8× A100 80GB | 8× H100 80GB |
| 70B   | 32× A100 80GB | 64× H100 80GB |
| 671B  | 256× H100 80GB | 512× H100 80GB |

### Software Setup

```bash
# Clone and install
git clone https://github.com/WrinkleFree/WrinkleFree-CheaperTraining.git
cd WrinkleFree-CheaperTraining
uv sync

# Or with pip
pip install -e .
```

## Experiment 1: Pretrain Phase 1

Initial pretraining on diverse data mixture.

### Data Mixture

| Dataset | Weight | Description |
|---------|--------|-------------|
| FineWeb-Edu | 63.75% | Educational web content |
| StarCoder | 10.66% | Code |
| OpenWebMath | 6.93% | Mathematical content |
| ArXiv | 6.36% | Scientific papers |
| Wikipedia | 5.03% | Encyclopedia |
| StackExchange | 5.03% | Q&A |
| Algebraic Stack | 2.25% | Mathematical proofs |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 4.0e-3 |
| Warmup Steps | 2,000 |
| LR Decay | Linear to 10% |
| Batch Size (per GPU) | 16 |
| Sequence Length | 2,048 |
| Weight Decay | 0.1 |
| Gradient Clip | 1.0 |
| Total Tokens | 2T |

### Commands

**Single GPU (140M model):**
```bash
python scripts/train.py \
    model=mobilellm_140m \
    training=pretrain_phase1 \
    data=pretrain_phase1_mix \
    distributed=single_gpu
```

**Multi-GPU with FSDP2 (950M model):**
```bash
torchrun --nproc_per_node=4 scripts/train.py \
    model=mobilellm_950m \
    training=pretrain_phase1 \
    data=pretrain_phase1_mix \
    distributed=fsdp2
```

**Large scale (7B+ models):**
```bash
torchrun --nnodes=8 --nproc_per_node=8 scripts/train.py \
    model=mobilellm_7b \
    training=pretrain_phase1 \
    data=pretrain_phase1_mix \
    distributed=fsdp2_tp \
    distributed.tensor_parallel_degree=4
```

### Expected Results

| Model | Final Loss | Perplexity | Tokens/sec/GPU |
|-------|------------|------------|----------------|
| 140M  | ~2.8       | ~16        | ~50k           |
| 360M  | ~2.6       | ~13        | ~35k           |
| 950M  | ~2.4       | ~11        | ~20k           |

## Experiment 2: Pretrain Phase 2

Math and code focused pretraining.

### Data Mixture Changes

Increased emphasis on mathematical and code content compared to Phase 1.

### Commands

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    model=mobilellm_950m \
    training=pretrain_phase2 \
    data=pretrain_phase2_mix \
    distributed=fsdp2 \
    training.checkpoint_path=/path/to/phase1/checkpoint
```

## Experiment 3: Mid-training (Knowledge Distillation)

Distillation from a larger teacher model (Llama-3.1-8B-Instruct).

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3.6e-4 |
| Warmup | None (immediate decay) |
| LR Decay | Linear to 0 |
| Batch Size (per GPU) | 4 |
| Sequence Length | 4,096 |
| Temperature | 1.0 |
| KD Alpha | 1.0 (pure KD) |
| Total Tokens | 100B |

### Commands

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    model=mobilellm_950m \
    training=midtrain \
    data=pretrain_phase1_mix \
    distributed=fsdp2 \
    training.checkpoint_path=/path/to/phase2/checkpoint \
    training.teacher_model=meta-llama/Llama-3.1-8B-Instruct
```

### Expected Results

| Metric | Target |
|--------|--------|
| KD Loss | < 2.0 |
| Teacher Agreement | > 60% |
| Student Accuracy | > 45% |

## Experiment 4: Post-training General SFT

Instruction following fine-tuning.

### Commands

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    model=mobilellm_950m \
    training=posttrain_general_sft \
    data=general_sft \
    distributed=fsdp2 \
    training.checkpoint_path=/path/to/midtrain/checkpoint
```

## Experiment 5: Post-training Reasoning SFT

Long chain-of-thought reasoning.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence Length | 32,768 |
| Learning Rate | 1.0e-5 |
| Epochs | 3 |

### Commands

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    model=mobilellm_950m \
    training=posttrain_reasoning_sft \
    data=reasoning_sft \
    distributed=fsdp2 \
    training.checkpoint_path=/path/to/general_sft/checkpoint
```

## Configuration Overrides

Override any configuration parameter from command line:

```bash
# Change learning rate
python scripts/train.py ... training.stage.learning_rate=1e-4

# Change batch size
python scripts/train.py ... training.stage.batch_size_per_gpu=8

# Change model architecture
python scripts/train.py ... model.num_layers=24 model.num_heads=16

# Enable gradient accumulation
python scripts/train.py ... training.stage.gradient_accumulation_steps=4
```

## Checkpointing

### Save Checkpoints

Checkpoints are saved automatically based on `checkpoint_interval`:

```yaml
# In training config
checkpoint_interval: 1000  # Save every 1000 steps
checkpoint_dir: /path/to/checkpoints
```

### Resume Training

```bash
python scripts/train.py ... \
    training.checkpoint_path=/path/to/checkpoint \
    training.resume=true
```

## Logging & Monitoring

### WandB Integration

```bash
python scripts/train.py ... \
    logging.wandb.enabled=true \
    logging.wandb.project=cheapertraining \
    logging.wandb.run_name=mobilellm_950m_phase1
```

### Metrics Tracked

- `train/loss`: Training loss
- `train/accuracy`: Token prediction accuracy
- `train/perplexity`: Perplexity
- `train/learning_rate`: Current learning rate
- `train/grad_norm`: Gradient norm
- `train/tokens_per_second`: Training throughput
- `system/gpu_memory`: GPU memory usage

## Reproducing Paper Results

To reproduce the full MobileLLM-R1 training pipeline:

```bash
# Phase 1: Pretrain (2T tokens)
./scripts/run_phase1.sh

# Phase 2: Pretrain math-focused (2T tokens)
./scripts/run_phase2.sh

# Mid-training: KD (100B tokens)
./scripts/run_midtrain.sh

# Post-training: General SFT
./scripts/run_general_sft.sh

# Post-training: Reasoning SFT
./scripts/run_reasoning_sft.sh
```

## Troubleshooting

### Out of Memory

1. Reduce batch size: `training.stage.batch_size_per_gpu=4`
2. Enable gradient checkpointing: `distributed.activation_checkpointing=full`
3. Use more GPUs with FSDP2

### Slow Training

1. Enable sequence packing: `data.packing=true`
2. Use Flash Attention: `model.use_flash_attention=true`
3. Increase batch size if memory allows

### NaN Loss

1. Reduce learning rate: `training.stage.learning_rate=1e-4`
2. Enable gradient clipping: `training.stage.max_grad_norm=1.0`
3. Check data for issues

## Evaluation

After training, evaluate models using standard benchmarks:

```bash
# Run evaluation
python scripts/evaluate.py \
    --model_path /path/to/checkpoint \
    --benchmarks arc_easy,hellaswag,mmlu,gsm8k
```

### Expected Benchmark Results (950M model)

| Benchmark | Score |
|-----------|-------|
| ARC-Easy | ~65% |
| HellaSwag | ~55% |
| MMLU | ~35% |
| GSM8K | ~25% |
