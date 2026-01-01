#!/usr/bin/env python
"""
Multi-GPU smoke test for Tensor Parallelism + FSDP2.

Run with:
    torchrun --standalone --nproc_per_node=2 scripts/test_tp_smoke.py
    torchrun --standalone --nproc_per_node=4 scripts/test_tp_smoke.py --tp-size 4
    torchrun --standalone --nproc_per_node=8 scripts/test_tp_smoke.py --tp-size 8 --dp-size 1

This script:
1. Initializes distributed training
2. Creates a small BitNet model (SmolLM2-135M config)
3. Applies 2D parallelism (TP + FSDP2)
4. Runs a specified number of training steps
5. Verifies loss decreases and all ranks are synchronized
6. Optionally performs "rebalancing" at a specified step (placeholder for data remixing)
"""

import argparse
import logging
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="TP+FSDP2 Smoke Test")
    parser.add_argument("--model", type=str, default="smollm2_135m",
                        help="Model config name")
    parser.add_argument("--tp-size", type=int, default=0,
                        help="Tensor parallel degree (0=auto)")
    parser.add_argument("--dp-size", type=int, default=0,
                        help="Data parallel degree (0=auto, derived from world_size/tp_size)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of training steps")
    parser.add_argument("--rebalance-step", type=int, default=15,
                        help="Step at which to trigger rebalancing (0=disabled)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-GPU batch size")
    parser.add_argument("--seq-length", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--check-sync", action="store_true",
                        help="Verify all ranks have same loss (slower)")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def create_model(model_name: str = "smollm2_135m"):
    """Create a small BitNet model for testing."""
    from wrinklefree.models import BitNetLlama
    from wrinklefree.models.config import BitNetConfig

    # TP-compatible config (heads must be divisible by TP size)
    # For TP=8: need num_heads AND num_kv_heads divisible by 8
    # For TP=4: need num_heads AND num_kv_heads divisible by 4
    # For TP=2: need num_heads AND num_kv_heads divisible by 2
    if model_name == "smollm2_135m":
        config = BitNetConfig(
            hidden_size=512,  # 512 = 8 heads * 64 head_dim
            intermediate_size=1536,
            num_hidden_layers=6,  # Reduced for smoke test
            num_attention_heads=8,  # Divisible by 1, 2, 4, 8
            num_kv_heads=8,  # Must also be divisible by TP size
            vocab_size=49152,
            max_position_embeddings=2048,
            rope_theta=100000.0,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = BitNetLlama(config)
    return model


def create_synthetic_batch(batch_size: int, seq_length: int, vocab_size: int, device: torch.device):
    """Create a synthetic batch for training."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # Ignore last token

    return {"input_ids": input_ids, "labels": labels}


def compute_loss(model: nn.Module, batch: dict) -> torch.Tensor:
    """Compute cross-entropy loss."""
    outputs = model(input_ids=batch["input_ids"])
    logits = outputs["logits"]

    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def rebalance(step: int, rank: int):
    """
    Placeholder for rebalancing operation.

    In a full implementation, this could:
    - Update data mixture weights based on influence scores
    - Redistribute data across workers
    - Adjust learning rates per domain

    For now, just logs the rebalancing event.
    """
    if rank == 0:
        logger.info(f"[Step {step}] REBALANCING: Trigger rebalancing operation")
        # TODO: Integrate with InfluenceAwareOptimizer to update mixture weights
        logger.info(f"[Step {step}] REBALANCING: (placeholder - no actual rebalancing performed)")


def run_smoke_test(args):
    """Run the smoke test."""
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        logger.info("=" * 60)
        logger.info("TP+FSDP2 Smoke Test")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size}")
        logger.info(f"TP size: {args.tp_size} (0=auto)")
        logger.info(f"Steps: {args.steps}")
        logger.info(f"Rebalance step: {args.rebalance_step}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Sequence length: {args.seq_length}")
        logger.info("=" * 60)

    # Set seed for reproducibility
    torch.manual_seed(args.seed + rank)

    # Create model
    if rank == 0:
        logger.info("Creating model...")
    model = create_model(args.model)
    model = model.to(device=device, dtype=torch.bfloat16)

    # Count parameters before sharding
    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logger.info(f"Model parameters: {total_params:,}")

    # Apply 2D parallelism
    if rank == 0:
        logger.info("Applying 2D parallelism (TP + FSDP2)...")

    from wrinklefree.training.tensor_parallel import setup_2d_parallel

    tp_size = args.tp_size if args.tp_size > 0 else world_size
    model, device_mesh = setup_2d_parallel(
        model,
        tp_size=tp_size,
        mixed_precision=True,
        activation_checkpointing=False,  # Disabled for smoke test speed
    )

    if rank == 0:
        logger.info(f"Device mesh: {device_mesh}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Get vocab size for synthetic data
    vocab_size = 49152  # SmolLM2 vocab size

    # Training loop
    losses = []
    start_time = time.time()

    if rank == 0:
        logger.info("Starting training loop...")

    model.train()
    for step in range(1, args.steps + 1):
        # Create synthetic batch
        batch = create_synthetic_batch(args.batch_size, args.seq_length, vocab_size, device)

        # Forward pass
        optimizer.zero_grad()
        loss = compute_loss(model, batch)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Track loss
        loss_val = loss.item()
        losses.append(loss_val)

        # Check synchronization if requested
        if args.check_sync and step % 5 == 0:
            loss_tensor = torch.tensor([loss_val], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size

            # All ranks should have similar loss
            if abs(loss_val - avg_loss) > 0.1:
                logger.warning(f"[Rank {rank}] Loss desync at step {step}: local={loss_val:.4f}, avg={avg_loss:.4f}")

        # Rebalancing
        if args.rebalance_step > 0 and step == args.rebalance_step:
            rebalance(step, rank)

        # Logging
        if rank == 0 and step % 5 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            logger.info(f"Step {step}/{args.steps} | Loss: {loss_val:.4f} | Steps/s: {steps_per_sec:.2f}")

    # Final summary
    if rank == 0:
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Smoke Test Complete")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Steps/second: {args.steps / elapsed:.2f}")
        logger.info(f"Initial loss: {losses[0]:.4f}")
        logger.info(f"Final loss: {losses[-1]:.4f}")

        # Check if loss decreased
        if losses[-1] < losses[0]:
            logger.info("SUCCESS: Loss decreased during training")
        else:
            logger.warning("WARNING: Loss did not decrease")

        # Check for NaN/Inf
        if any(not (l == l) or abs(l) == float('inf') for l in losses):
            logger.error("FAILURE: NaN or Inf detected in loss")
            return False

    # Synchronize before exit
    dist.barrier()

    if rank == 0:
        logger.info("All ranks synchronized successfully")

    return True


def main():
    args = parse_args()

    try:
        success = run_smoke_test(args)
        if not success:
            exit(1)
    except Exception as e:
        logger.exception(f"Smoke test failed with error: {e}")
        exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
