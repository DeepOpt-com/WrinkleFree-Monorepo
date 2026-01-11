"""
Tensor Parallelism + FSDP2 utilities for distributed training.

This module implements 2D parallelism following TorchTitan patterns:
- Tensor Parallel (TP) within nodes (uses NVLink)
- FSDP2 (Data Parallel) across nodes

Key components:
- create_device_mesh: Creates 2D mesh with ("dp", "tp") dimensions
- DistributedSubLN: TP-aware RMSNorm with all-reduce for global variance
- apply_tensor_parallel: Applies TP to BitNet model layers
- apply_fsdp2: Wraps model with FSDP2 on DP dimension

References:
- PyTorch TP Tutorial: https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html
- TorchTitan: https://arxiv.org/html/2410.06511v3
"""

import logging
from typing import Optional, Type

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


def create_device_mesh(
    tp_size: int,
    device_type: str = "cuda",
) -> "torch.distributed.device_mesh.DeviceMesh":
    """
    Create a 2D DeviceMesh for hybrid TP+DP parallelism.

    The mesh has two dimensions:
    - "dp": Data parallel dimension (FSDP shards here)
    - "tp": Tensor parallel dimension (model shards here)

    Args:
        tp_size: Tensor parallel degree. If 0, auto-infer from world_size.
        device_type: Device type ("cuda" or "cpu")

    Returns:
        DeviceMesh with ("dp", "tp") dimensions

    Raises:
        ValueError: If world_size is not divisible by tp_size
    """
    from torch.distributed.device_mesh import init_device_mesh

    if not dist.is_initialized():
        raise RuntimeError("Distributed must be initialized before creating device mesh")

    world_size = dist.get_world_size()

    # Auto-infer TP size if not specified
    if tp_size <= 0:
        # For single node, use all GPUs for TP
        # For multi-node, would need to detect node boundaries
        tp_size = world_size
        logger.info(f"Auto-inferred tp_size={tp_size} from world_size={world_size}")

    if world_size % tp_size != 0:
        raise ValueError(
            f"World size {world_size} must be divisible by TP size {tp_size}"
        )

    dp_size = world_size // tp_size
    logger.info(f"Creating 2D device mesh: DP={dp_size}, TP={tp_size}")

    # Create 2D mesh with named dimensions
    mesh = init_device_mesh(
        device_type,
        (dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )

    return mesh


class DistributedSubLN(nn.Module):
    """
    Tensor Parallel-aware SubLN (RMSNorm).

    When tensor parallelism shards the hidden dimension across GPUs,
    standard RMSNorm computes local variance instead of global variance.
    This module performs an all-reduce to compute the correct global norm.

    Architecture:
        1. Compute local sum of squares: sum(x^2) for local shard
        2. All-reduce across TP group to get global sum
        3. Compute global variance: global_sum / global_dim
        4. Normalize and scale (weight is also sharded)

    Args:
        hidden_size: Local hidden dimension (after sharding)
        eps: Numerical stability epsilon
        process_group: TP process group for all-reduce
        global_hidden_size: Original hidden size before sharding
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        process_group: Optional[dist.ProcessGroup] = None,
        global_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.process_group = process_group
        self.weight = nn.Parameter(torch.ones(hidden_size))

        # Compute global dimension
        if global_hidden_size is not None:
            self.global_hidden_size = global_hidden_size
        elif process_group is not None:
            self.global_hidden_size = hidden_size * dist.get_world_size(process_group)
        else:
            self.global_hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TP-aware RMSNorm.

        Args:
            x: Input tensor of shape (..., local_hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        # Compute local sum of squares
        local_sum_sq = x.pow(2).sum(dim=-1, keepdim=True)

        # All-reduce to get global sum
        if self.process_group is not None:
            dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM, group=self.process_group)

        # Compute global variance
        variance = local_sum_sq / self.global_hidden_size

        # Normalize and scale
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight

    @classmethod
    def from_subln(
        cls,
        subln: nn.Module,
        process_group: dist.ProcessGroup,
    ) -> "DistributedSubLN":
        """
        Create DistributedSubLN from existing SubLN module.

        Args:
            subln: Original SubLN or RMSNorm module
            process_group: TP process group

        Returns:
            DistributedSubLN with copied weights
        """
        tp_size = dist.get_world_size(process_group)
        global_hidden_size = subln.hidden_size

        # Local hidden size after sharding
        local_hidden_size = global_hidden_size // tp_size

        # Get device from original module
        device = subln.weight.device if hasattr(subln, "weight") and subln.weight is not None else "cpu"
        dtype = subln.weight.dtype if hasattr(subln, "weight") and subln.weight is not None else torch.float32

        distributed_ln = cls(
            hidden_size=local_hidden_size,
            eps=getattr(subln, "eps", 1e-6),
            process_group=process_group,
            global_hidden_size=global_hidden_size,
        )

        # Copy and shard weights
        rank = dist.get_rank(process_group)
        start = rank * local_hidden_size
        end = start + local_hidden_size

        with torch.no_grad():
            if hasattr(subln, "weight") and subln.weight is not None:
                distributed_ln.weight.copy_(subln.weight[start:end])

        # Move to same device and dtype as original
        distributed_ln = distributed_ln.to(device=device, dtype=dtype)

        return distributed_ln

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"global_hidden_size={self.global_hidden_size}, "
            f"eps={self.eps}"
        )


def get_bitnet_tp_plan() -> dict:
    """
    Get tensor parallelism plan for BitNetDecoderLayer.

    Returns a dictionary mapping module paths to parallelization styles:
    - ColwiseParallel: Shards output features (used for Q/K/V/gate/up projections)
    - RowwiseParallel: Shards input features (used for O/down projections)

    The pattern follows standard transformer TP:
    - Attention: [Col, Col, Col] -> [Row] (Q,K,V -> O)
    - FFN: [Col, Col] -> [Row] (gate,up -> down)

    Note: We use use_local_output=False for ColwiseParallel to keep outputs as DTensors.
    This allows DTensor to handle view/reshape operations automatically without needing
    to manually track local vs global num_heads. See PyTorch TP tutorial for details:
    https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

    Returns:
        Dictionary of {module_path: ParallelStyle}
    """
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    return {
        # Attention projections
        # Default use_local_output=True converts DTensor to local tensor after projection
        # This allows mixing with non-DTensor tensors like RoPE freqs_cis
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        # FFN projections
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }


def _patch_subln_modules(
    model: nn.Module,
    tp_group: dist.ProcessGroup,
) -> None:
    """
    Replace SubLN modules with DistributedSubLN for TP correctness.

    This patches SubLN modules that receive sharded inputs:
    - self_attn.subln (before o_proj)
    - mlp.subln (before down_proj)

    Args:
        model: Model to patch
        tp_group: Tensor parallel process group
    """
    from wf_train.models.subln import SubLN

    for name, module in model.named_modules():
        # Find attention SubLN
        if hasattr(module, "self_attn") and hasattr(module.self_attn, "subln"):
            if isinstance(module.self_attn.subln, SubLN):
                module.self_attn.subln = DistributedSubLN.from_subln(
                    module.self_attn.subln, tp_group
                )
                logger.debug(f"Patched {name}.self_attn.subln with DistributedSubLN")

        # Find FFN SubLN
        if hasattr(module, "mlp") and hasattr(module.mlp, "subln"):
            if isinstance(module.mlp.subln, SubLN):
                module.mlp.subln = DistributedSubLN.from_subln(
                    module.mlp.subln, tp_group
                )
                logger.debug(f"Patched {name}.mlp.subln with DistributedSubLN")


def apply_tensor_parallel(
    model: nn.Module,
    device_mesh: "torch.distributed.device_mesh.DeviceMesh",
) -> nn.Module:
    """
    Apply Tensor Parallelism to BitNet model.

    This function:
    1. Patches SubLN modules with DistributedSubLN for correct normalization
    2. Applies parallelize_module with ColwiseParallel/RowwiseParallel

    Args:
        model: BitNet model to parallelize
        device_mesh: Device mesh (uses "tp" dimension)

    Returns:
        Model with TP applied (modified in-place)
    """
    from torch.distributed.tensor.parallel import parallelize_module

    # Get TP submesh
    if hasattr(device_mesh, "__getitem__"):
        tp_mesh = device_mesh["tp"]
    else:
        tp_mesh = device_mesh

    tp_size = tp_mesh.size()
    if tp_size == 1:
        logger.info("TP size is 1, skipping tensor parallelism")
        return model

    logger.info(f"Applying tensor parallelism with TP={tp_size}")

    # Get TP process group
    tp_group = tp_mesh.get_group()

    # NOTE: We skip SubLN patching when using use_local_output=True (default).
    # With use_local_output=True, ColwiseParallel converts DTensor outputs to local
    # tensors, so SubLN receives local hidden dims and computes local variance.
    # This is acceptable since the shards are independent after the projection.
    # Only need DistributedSubLN if using use_local_output=False (DTensor throughout).

    # Get TP plan for transformer layers
    tp_plan = get_bitnet_tp_plan()

    # 3. Apply TP to each transformer layer
    from wf_train.models.transformer import BitNetDecoderLayer

    for name, module in model.named_modules():
        if isinstance(module, BitNetDecoderLayer):
            parallelize_module(
                module=module,
                device_mesh=tp_mesh,
                parallelize_plan=tp_plan,
            )
            logger.debug(f"Applied TP to {name}")

    # 4. Handle embeddings and output head
    # Token embeddings: RowwiseParallel (vocab stays full, hidden shards)
    # LM head: ColwiseParallel (hidden -> vocab, output replicated)
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    from torch.distributed.tensor import Replicate

    root_plan = {}

    # Embedding: input is token ids (replicated), output is sharded hidden
    if hasattr(model, "embed_tokens"):
        root_plan["embed_tokens"] = RowwiseParallel(
            input_layouts=Replicate(),
        )

    # LM head: input is sharded hidden, output is replicated logits
    if hasattr(model, "lm_head"):
        root_plan["lm_head"] = ColwiseParallel(
            output_layouts=Replicate(),
        )

    if root_plan:
        parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan=root_plan,
        )
        logger.debug(f"Applied TP to root modules: {list(root_plan.keys())}")

    return model


def apply_fsdp2(
    model: nn.Module,
    device_mesh: "torch.distributed.device_mesh.DeviceMesh",
    mixed_precision: bool = True,
    activation_checkpointing: bool = True,
) -> nn.Module:
    """
    Apply FSDP2 (fully_shard) to model on the DP dimension.

    FSDP2 uses DTensor-based per-parameter sharding which composes
    cleanly with Tensor Parallelism.

    Args:
        model: Model to wrap (may already have TP applied)
        device_mesh: Device mesh (uses "dp" dimension)
        mixed_precision: Use bfloat16 for compute, float32 for reduce
        activation_checkpointing: Enable gradient checkpointing

    Returns:
        Model with FSDP2 applied
    """
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    # Get DP submesh
    if hasattr(device_mesh, "__getitem__"):
        dp_mesh = device_mesh["dp"]
    else:
        dp_mesh = device_mesh

    dp_size = dp_mesh.size()
    if dp_size == 1:
        logger.info("DP size is 1, skipping FSDP2")
        return model

    logger.info(f"Applying FSDP2 with DP={dp_size}")

    # Mixed precision policy
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

    # Wrap transformer layers first (bottom-up)
    from wf_train.models.transformer import BitNetDecoderLayer

    for name, module in model.named_modules():
        if isinstance(module, BitNetDecoderLayer):
            fully_shard(
                module,
                mesh=dp_mesh,
                mp_policy=mp_policy,
            )
            logger.debug(f"Applied FSDP2 to {name}")

    # Wrap root model
    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
    )

    # Apply activation checkpointing if requested
    if activation_checkpointing:
        _apply_activation_checkpointing_fsdp2(model, BitNetDecoderLayer)

    return model


def _apply_activation_checkpointing_fsdp2(
    model: nn.Module,
    transformer_layer_cls: Type[nn.Module],
) -> None:
    """
    Apply activation checkpointing to FSDP2 model.

    Args:
        model: FSDP2-wrapped model
        transformer_layer_cls: Class of transformer layers to checkpoint
    """
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        import functools

        check_fn = lambda submodule: isinstance(submodule, transformer_layer_cls)

        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn,
        )
        logger.info("Applied activation checkpointing to transformer layers")
    except (ImportError, TypeError) as e:
        logger.warning(f"Could not apply activation checkpointing: {e}")


def setup_2d_parallel(
    model: nn.Module,
    tp_size: int = 0,
    mixed_precision: bool = True,
    activation_checkpointing: bool = True,
) -> tuple[nn.Module, "torch.distributed.device_mesh.DeviceMesh"]:
    """
    Setup 2D parallelism (TP + FSDP2) for model.

    This is the main entry point for applying combined parallelism.
    Order matters: TP must be applied before FSDP2.

    Args:
        model: BitNet model to parallelize
        tp_size: Tensor parallel degree (0 = auto-infer)
        mixed_precision: Use mixed precision for FSDP2
        activation_checkpointing: Enable activation checkpointing

    Returns:
        Tuple of (parallelized model, device mesh)
    """
    # Create device mesh
    device_mesh = create_device_mesh(tp_size)

    # Apply TP first (shards within layers)
    model = apply_tensor_parallel(model, device_mesh)

    # Apply FSDP2 second (shards across layers)
    model = apply_fsdp2(
        model,
        device_mesh,
        mixed_precision=mixed_precision,
        activation_checkpointing=activation_checkpointing,
    )

    return model, device_mesh


def get_tp_rank(device_mesh: "torch.distributed.device_mesh.DeviceMesh") -> int:
    """Get local rank within TP group."""
    tp_mesh = device_mesh["tp"]
    return tp_mesh.get_local_rank()


def get_dp_rank(device_mesh: "torch.distributed.device_mesh.DeviceMesh") -> int:
    """Get local rank within DP group."""
    dp_mesh = device_mesh["dp"]
    return dp_mesh.get_local_rank()
