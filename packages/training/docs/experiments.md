# Experiments and Reproduction

This document details how to reproduce the key results and experiments for WrinkleFree-1.58Quant.

## Environment Setup

Ensure you have installed the dependencies as described in the [README](../README.md).

```bash
uv sync --all-extras
```

## Experiment 1: SmolLM2-135M 1.58-bit Conversion

This experiment demonstrates the full pipeline on a small model (SmolLM2-135M).

### 1. Stage 1: SubLN Insertion
Stabilize the model by inserting SubLN layers.

```bash
uv run python scripts/train.py \
    model=smollm2_135m \
    training=stage1_subln \
    distributed=single_gpu
```

**Expected Result**: Training should converge quickly (few hundred steps). Loss should decrease initially and stabilize.

### 2. Stage 2: Continue Pre-training
Adapt the model to 1.58-bit weights.

#### Basic (Single Dataset)
```bash
uv run python scripts/train.py \
    model=smollm2_135m \
    training=stage2_pretrain \
    data=fineweb \
    distributed=single_gpu
```

#### Influence-Based Mixed Pretraining (Recommended)
For better performance, use influence-based data selection with multiple data sources:

```bash
uv run python scripts/train.py \
    model=smollm2_135m \
    training=stage2_pretrain \
    data=mixed_pretrain \
    distributed=single_gpu
```

This uses the `mixed_pretrain` config which includes:
- **FineWeb** (40%): General web content
- **SlimPajama** (30%): Curated web content
- **OpenWebMath** (15%): Mathematical reasoning
- **CodeParrot** (15%): Code (OSS, ungated)

The influence system dynamically adjusts these weights based on a probe dataset (FineWeb-Edu) to optimize for the target distribution.

**Note**: This stage typically requires ~10B tokens for good performance.
**Influence Integration**: Enabled by default in `stage2_pretrain.yaml`. The optimizer is wrapped with `InfluenceAwareOptimizer` which updates mixture weights every 1000 steps.

### 3. Stage 3: Distillation
Fine-tune with teacher guidance.

```bash
uv run python scripts/train.py \
    model=smollm2_135m \
    training=stage3_distill_smollm2 \
    data=downstream \
    distillation=classification \
    distributed=single_gpu
```

**Metrics**: Monitor `eval/loss`, `eval/accuracy` (if applicable), and distillation losses (`loss_logits`, `loss_attention`).

## Experiment 2: Influence-based Data Selection

Verify the impact of influence-based data weighting.

1.  **Baseline**: Run Stage 2 with static mixture weights (e.g., 50/50).
2.  **Influence**: Run Stage 2 with `influence.enabled=true` and a relevant probe set.
3.  **Comparison**: Compare validation loss on the target domain. The influence-guided run should achieve lower loss with fewer tokens.

## Troubleshooting

- **OOM Errors**: Reduce `batch_size` or `max_seq_length`. Enable `gradient_checkpointing`.
- **Divergence**: Check learning rate (lower it for quantized training). Ensure `warmup_steps` is sufficient in Stage 2.
