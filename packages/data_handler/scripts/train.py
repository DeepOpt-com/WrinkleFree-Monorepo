#!/usr/bin/env python3
"""Main training script for CheaperTraining.

Usage:
    python scripts/train.py model=mobilellm_950m training=pretrain_phase1

Reference: MobileLLM-R1 paper (arXiv:2509.24945)
"""

import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_handler.models import MobileLLM, MobileLLMConfig
from data_handler.training import Trainer
from data_handler.training.stages.base import StageConfig
from data_handler.training.optimizer import create_optimizer
from data_handler.training.scheduler import create_scheduler
from data_handler.data.tokenization import TokenizerWrapper
from data_handler.data.mixing import create_mixed_dataset
from data_handler.data.datasets.pretrain import create_pretrain_dataloader


def setup_distributed():
    """Initialize distributed training if applicable."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")

        return rank, local_rank, world_size
    return 0, 0, 1


def setup_logging(config: DictConfig, rank: int):
    """Setup logging (WandB, etc.)."""
    if rank != 0:
        return None

    if config.logging.wandb.enabled:
        import wandb

        wandb.init(
            project=config.logging.wandb.project,
            entity=config.logging.wandb.entity,
            name=config.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
            tags=config.logging.wandb.get("tags", []),
        )
        return wandb
    return None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """Main training entry point."""
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Print config on rank 0
    if rank == 0:
        print("=" * 60)
        print("CheaperTraining - MobileLLM-R1 Training")
        print("Reference: https://arxiv.org/abs/2509.24945")
        print("=" * 60)
        print(OmegaConf.to_yaml(config))

    # Setup logging
    wandb = setup_logging(config, rank)

    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create model config
    model_config = MobileLLMConfig(
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        num_kv_heads=config.model.num_kv_heads,
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        vocab_size=config.model.vocab_size,
        max_seq_len=config.model.max_seq_len,
        use_qk_norm=config.model.use_qk_norm,
        use_weight_sharing=config.model.use_weight_sharing,
        rope_base=config.model.rope_base,
        norm_eps=config.model.norm_eps,
        dropout=config.model.dropout,
        attention_dropout=config.model.attention_dropout,
    )

    # Create model
    model = MobileLLM(model_config)
    model = model.to(device)

    # Enable gradient checkpointing if configured
    if config.training.stage.get("use_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            mode=config.training.stage.get("gradient_checkpointing_mode", "quantized")
        )
        if rank == 0:
            print(f"Gradient checkpointing enabled: {config.training.stage.gradient_checkpointing_mode}")

    if rank == 0:
        num_params = model.num_parameters()
        print(f"\nModel: {config.model.name}")
        print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"Device: {device}")
        print(f"World size: {world_size}")

    # Setup tokenizer
    if rank == 0:
        print("\nLoading tokenizer...")
    tokenizer = TokenizerWrapper(
        tokenizer_path=config.data.tokenizer.path,
        max_length=config.training.stage.seq_len,
    )
    if rank == 0:
        print(f"Tokenizer loaded: {config.data.tokenizer.path}")

    # Create dataset
    if rank == 0:
        print("Creating mixed dataset (this may take a moment for streaming datasets)...")
    dataset = create_mixed_dataset(
        config=OmegaConf.to_container(config.data, resolve=True),
        tokenizer=tokenizer,
        packing=config.data.get("packing", True),
    )
    if rank == 0:
        print("Dataset created!")

    # Create dataloader
    # Auto-set num_workers based on streaming mode and available CPUs
    # Streaming datasets must use num_workers=0 to avoid worker process hangs
    is_streaming = config.data.get("streaming", True)
    if is_streaming:
        num_workers = 0
        if rank == 0:
            print("Using num_workers=0 for streaming datasets")
    else:
        # For non-streaming, use up to 4 workers based on CPU count
        import multiprocessing
        num_workers = min(4, max(1, multiprocessing.cpu_count() // 2))
        if rank == 0:
            print(f"Using num_workers={num_workers} for non-streaming dataset")

    dataloader = create_pretrain_dataloader(
        dataset=dataset,
        batch_size=config.training.stage.batch_size_per_gpu,
        num_workers=num_workers,
    )

    # Create stage config
    stage_config = StageConfig(
        name=config.training.stage.name,
        num_steps=config.training.stage.num_steps,
        batch_size_per_gpu=config.training.stage.batch_size_per_gpu,
        seq_len=config.training.stage.seq_len,
        learning_rate=config.training.stage.learning_rate,
        weight_decay=config.training.stage.weight_decay,
        scheduler_type=config.training.stage.scheduler_type,
        warmup_steps=config.training.stage.warmup_steps,
        lr_decay_ratio=config.training.stage.lr_decay_ratio,
        dtype=config.training.stage.dtype,
        gradient_clip_norm=config.training.gradient.clip_norm,
        gradient_accumulation_steps=config.training.gradient.accumulation_steps,
    )

    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_type=config.optimizer.optimizer.type,
        learning_rate=stage_config.learning_rate,
        weight_decay=stage_config.weight_decay,
        betas=tuple(config.optimizer.optimizer.betas),
        eps=config.optimizer.optimizer.eps,
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=stage_config.scheduler_type,
        warmup_steps=stage_config.warmup_steps,
        total_steps=stage_config.num_steps,
        min_lr_ratio=stage_config.lr_decay_ratio,
    )

    # Create training stage
    from data_handler.training.stages.pretrain import PretrainStage

    stage = PretrainStage(
        config=stage_config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    # Output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    if rank == 0:
        print(f"\nStarting training: {stage_config.name}")
        print(f"Steps: {stage_config.num_steps}")
        print(f"Output: {output_dir}")

    def checkpoint_callback(stage, metrics):
        if rank == 0:
            ckpt_path = output_dir / f"checkpoint_step{stage.global_step}.pt"
            torch.save(stage.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    for metrics in stage.run(
        log_interval=config.logging.log_interval,
        checkpoint_interval=config.checkpoint.save_interval,
        checkpoint_callback=checkpoint_callback,
    ):
        if rank == 0:
            # Print metrics for monitoring
            print(f"Step {metrics.step}: loss={metrics.loss:.4f}, "
                  f"lr={metrics.learning_rate:.2e}, "
                  f"grad_norm={metrics.grad_norm:.4f}, "
                  f"acc={metrics.extra.get('accuracy', 0):.4f}, "
                  f"ppl={metrics.extra.get('perplexity', 0):.2f}")

            if wandb:
                wandb.log({
                    "train/loss": metrics.loss,
                    "train/lr": metrics.learning_rate,
                    "train/grad_norm": metrics.grad_norm,
                    "train/step": metrics.step,
                    **{f"train/{k}": v for k, v in metrics.extra.items()},
                })

    # Save final checkpoint
    if rank == 0:
        final_path = output_dir / "checkpoint_final.pt"
        torch.save(stage.state_dict(), final_path)
        print(f"\nTraining complete! Final checkpoint: {final_path}")

        if wandb:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: wandb cleanup error (can be ignored): {e}")


if __name__ == "__main__":
    main()
