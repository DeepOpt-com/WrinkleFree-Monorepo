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

## Model Requirements for TP

### Attention Head Divisibility

For tensor parallelism to work, **both `num_attention_heads` AND `num_kv_heads` must be divisible by `tp_size`**:

| TP Size | Valid num_heads | Invalid |
|---------|-----------------|---------|
| 2 | 2, 4, 6, 8, 16... | 1, 3, 5, 7... |
| 4 | 4, 8, 12, 16... | 1, 2, 3, 5... |
| 8 | 8, 16, 24, 32... | 1, 2, 4, 6... |

Example TP-compatible config:
```python
config = BitNetConfig(
    hidden_size=512,        # 512 = 8 heads * 64 head_dim
    num_attention_heads=8,  # Divisible by 1, 2, 4, 8
    num_kv_heads=8,         # Must ALSO be divisible by TP size
    ...
)
```

### Attention TP Implementation (TorchTitan Pattern)

The attention module uses the **TorchTitan pattern** of using `-1` in `view()` to infer local heads dynamically from tensor size:

```python
# After q_proj with TP, output dimension is reduced
xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

# Use -1 to infer actual local heads from tensor size
# TP may have sharded the projection outputs, reducing the head count
xq = xq.view(batch_size, seq_len, -1, self.head_dim)
xk = xk.view(batch_size, seq_len, -1, self.head_dim)
xv = xv.view(batch_size, seq_len, -1, self.head_dim)
```

This approach from [TorchTitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model/model.py) ensures the reshape works regardless of TP sharding. The comment in TorchTitan explains: *"Use -1 instead of `n_heads` to infer the actual local heads from sizes of xq, xk, and xv as TP may have sharded them after the above linear ops."*

**Key benefits:**
- No need to track TP world size in attention module
- Works with default `use_local_output=True` (ColwiseParallel default)
- No mixed DTensor/Tensor issues with RoPE freqs_cis buffer

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

### "Shape '[B, S, H, D]' is invalid for input"

This error means the attention heads aren't divisible by TP size. Ensure:
1. `num_attention_heads % tp_size == 0`
2. `num_kv_heads % tp_size == 0`

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

### SkyPilot: CUDA_VISIBLE_DEVICES Empty

When launching jobs on SkyPilot without the `--gpus` flag, `CUDA_VISIBLE_DEVICES` may not be set. Add it explicitly to your YAML:

```yaml
envs:
  # Fix: SkyPilot doesn't set CUDA_VISIBLE_DEVICES for jobs without --gpus
  # See: https://github.com/skypilot-org/skypilot/issues/2510
  CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
```

### SkyPilot: "No module named 'wrinklefree'"

Ensure you use `uv run torchrun` (not bare `torchrun`) in your run script:

```yaml
run: |
  source ~/.local/bin/env
  uv run torchrun --standalone --nproc_per_node=8 scripts/test_tp_smoke.py
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
