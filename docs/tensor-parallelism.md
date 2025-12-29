# Tensor Parallelism + FSDP2 (2D Parallelism)

This document describes the 2D parallelism implementation for training BitNet models at scale.

## Overview

2D parallelism combines:
- **Tensor Parallelism (TP)**: Shards model layers across GPUs within a node (uses NVLink)
- **FSDP2 (Data Parallelism)**: Shards parameters across nodes (uses IB/ethernet)

This follows the [TorchTitan](https://arxiv.org/html/2410.06511v3) approach for efficient large-scale training.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    8 GPU Training                        │
├─────────────────────────────────────────────────────────┤
│  Node 1                      Node 2                      │
│  ┌────────┐ ┌────────┐      ┌────────┐ ┌────────┐       │
│  │ GPU 0  │ │ GPU 1  │      │ GPU 4  │ │ GPU 5  │       │
│  │  TP=0  │ │  TP=1  │      │  TP=0  │ │  TP=1  │       │
│  │  DP=0  │ │  DP=0  │      │  DP=1  │ │  DP=1  │       │
│  └────────┘ └────────┘      └────────┘ └────────┘       │
│  ┌────────┐ ┌────────┐      ┌────────┐ ┌────────┐       │
│  │ GPU 2  │ │ GPU 3  │      │ GPU 6  │ │ GPU 7  │       │
│  │  TP=0  │ │  TP=1  │      │  TP=0  │ │  TP=1  │       │
│  │  DP=2  │ │  DP=2  │      │  DP=3  │ │  DP=3  │       │
│  └────────┘ └────────┘      └────────┘ └────────┘       │
└─────────────────────────────────────────────────────────┘

With TP=2, DP=4:
- Each TP group (2 GPUs) shares the model layers
- Each DP group (4 groups) has a full model copy, parameters sharded via FSDP
```

## Quick Start

### Training with TP+FSDP2

```bash
# 8 GPUs with TP=2, DP=4
torchrun --standalone --nproc_per_node=8 \
  scripts/train.py \
  model=smollm2_135m \
  training=stage2_pretrain \
  distributed=tp_fsdp \
  distributed.tensor_parallel.tp_size=2

# 8 GPUs with TP=8, DP=1 (pure tensor parallelism)
torchrun --standalone --nproc_per_node=8 \
  scripts/train.py \
  distributed=tp_fsdp \
  distributed.tensor_parallel.tp_size=8
```

### Smoke Test

```bash
# Test on 2xH100
torchrun --standalone --nproc_per_node=2 \
  scripts/test_tp_smoke.py --tp-size 2 --steps 30

# Test with different configurations
./scripts/test_tp_runpod.sh 4      # 4 GPUs, TP=4
./scripts/test_tp_runpod.sh 8 4    # 8 GPUs, TP=4, DP=2
```

## Configuration

### Hydra Config: `configs/distributed/tp_fsdp.yaml`

```yaml
backend: nccl
strategy: tp_fsdp

tensor_parallel:
  enabled: true
  tp_size: 0  # 0 = auto-infer from world_size

fsdp:
  enabled: true
  sharding_strategy: FULL_SHARD
  mixed_precision:
    enabled: true
    param_dtype: bfloat16
    reduce_dtype: float32
  activation_checkpointing:
    enabled: true
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tensor_parallel.tp_size` | 0 | TP degree (0 = all GPUs on node) |
| `fsdp.sharding_strategy` | FULL_SHARD | ZERO-3 sharding |
| `fsdp.mixed_precision.enabled` | true | Use bfloat16 for compute |

## Implementation Details

### Layer Parallelization

BitNetDecoderLayer is parallelized as follows:

| Module | Style | Shards |
|--------|-------|--------|
| `self_attn.q_proj` | ColwiseParallel | Output features |
| `self_attn.k_proj` | ColwiseParallel | Output features |
| `self_attn.v_proj` | ColwiseParallel | Output features |
| `self_attn.o_proj` | RowwiseParallel | Input features |
| `mlp.gate_proj` | ColwiseParallel | Output features |
| `mlp.up_proj` | ColwiseParallel | Output features |
| `mlp.down_proj` | RowwiseParallel | Input features |

### SubLN Handling

SubLN (RMSNorm) before output projections receives sharded inputs. Standard RMSNorm computes local variance, which is incorrect. We use `DistributedSubLN` which performs an all-reduce:

```python
# DistributedSubLN computes global variance correctly
local_sum_sq = x.pow(2).sum(dim=-1, keepdim=True)
dist.all_reduce(local_sum_sq, group=tp_group)
variance = local_sum_sq / global_hidden_size
```

### BitLinear Compatibility

BitLinear's quantization (`weight.abs().mean()`) works correctly with DTensor because the reduction is automatically distributed.

## When to Use TP vs FSDP-only

| Scenario | Recommendation |
|----------|----------------|
| Single node, 2-8 GPUs | TP+FSDP or pure TP |
| Multi-node training | TP within node, FSDP across nodes |
| Memory-constrained | FSDP-only (more memory efficient) |
| Latency-sensitive | TP (fewer communication rounds per step) |

### Performance Considerations

1. **TP requires NVLink**: TP should only shard within NVLink-connected GPUs (typically within a node)
2. **FSDP works across nodes**: FSDP can use slower interconnects (IB, ethernet)
3. **TP overhead**: Each layer has allreduce overhead; more layers = more overhead
4. **Optimal TP degree**: Usually 2-8, matching GPUs per node

## Troubleshooting

### "Distributed must be initialized"

Ensure you're using `torchrun` or `torch.distributed.launch`:
```bash
torchrun --standalone --nproc_per_node=N script.py
```

### "World size not divisible by TP size"

TP size must divide world size evenly:
- 8 GPUs: TP can be 1, 2, 4, or 8
- 6 GPUs: TP can be 1, 2, 3, or 6

### NCCL Timeout

Increase timeout for slow networks:
```bash
export NCCL_TIMEOUT=1800  # 30 minutes
```

### OOM with TP

TP doesn't reduce memory per GPU (just computation). Use FSDP for memory reduction:
```yaml
distributed.tensor_parallel.tp_size: 2  # Small TP degree
distributed.fsdp.sharding_strategy: FULL_SHARD  # Maximum sharding
```

## API Reference

### `setup_2d_parallel(model, tp_size, mixed_precision, activation_checkpointing)`

Main entry point for 2D parallelism.

```python
from wrinklefree.training.tensor_parallel import setup_2d_parallel

model, device_mesh = setup_2d_parallel(
    model,
    tp_size=2,  # 0 = auto-infer
    mixed_precision=True,
    activation_checkpointing=True,
)
```

### `create_device_mesh(tp_size)`

Creates 2D DeviceMesh with ("dp", "tp") dimensions.

### `DistributedSubLN`

TP-aware RMSNorm that computes correct global variance.

## References

- [PyTorch TP Tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [PyTorch FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [TorchTitan Paper](https://arxiv.org/html/2410.06511v3)
- [PyTorch Lightning 2D Parallelism](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/tp_fsdp.html)
