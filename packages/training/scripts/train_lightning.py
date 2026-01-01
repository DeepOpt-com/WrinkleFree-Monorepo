#!/usr/bin/env python3
"""PyTorch Lightning training entry point for WrinkleFree.

This is the new, simplified training script using PyTorch Lightning.
It provides:
- Auto batch size scaling via BatchSizeFinder
- Built-in DDP/FSDP support
- Clean separation of concerns
- All objectives work unchanged (DLM, LRC, distillation)

Usage:
    uv run python scripts/train_lightning.py model=smollm2_135m training=unified
    uv run python scripts/train_lightning.py model=qwen3_4b training=bitdistill_full

With auto batch size:
    uv run python scripts/train_lightning.py model=smollm2_135m training=unified \
        training.auto_batch_size=true
"""

import logging
import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    BatchSizeFinder,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bitnet_arch.conversion import auto_convert_if_needed
from wrinklefree.lightning import (
    GCSCheckpointCallback,
    LambdaWarmupCallback,
    QKClipCallback,
    TokenCountCallback,
    WrinkleFreeDataModule,
    WrinkleFreeLightningModule,
    ZClipCallback,
)
from wrinklefree.objectives import create_objective_manager
from wrinklefree.teachers import HiddenStateTeacher

logger = logging.getLogger(__name__)


def setup_logging(rank: int) -> None:
    """Setup logging configuration."""
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_model_and_tokenizer(cfg: DictConfig, device: str = "cuda"):
    """Load model and tokenizer from config."""
    model_name = cfg.model.name
    # Try 'path' first, then 'pretrained_name', fallback to name
    model_path = cfg.model.get("path", cfg.model.get("pretrained_name", model_name))

    logger.info(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Auto-convert to BitNet if needed
    if cfg.training.get("auto_convert", {}).get("enabled", True):
        exclude_layers = cfg.training.get("auto_convert", {}).get("exclude_layers", None)
        model = auto_convert_if_needed(
            model,
            hidden_size=cfg.model.hidden_size,
            intermediate_size=cfg.model.intermediate_size,
            exclude_layers=exclude_layers,
        )
        logger.info("Model converted to BitNet")

    return model, tokenizer


def load_teacher_model(cfg: DictConfig) -> HiddenStateTeacher | None:
    """Load teacher model if distillation is enabled."""
    if not cfg.training.get("distillation", {}).get("enabled", False):
        return None

    teacher_path = cfg.training.distillation.get("teacher_path")
    if not teacher_path:
        return None

    logger.info(f"Loading teacher model: {teacher_path}")
    teacher = HiddenStateTeacher.from_pretrained(
        teacher_path,
        torch_dtype=torch.bfloat16,
    )
    return teacher


def create_callbacks(cfg: DictConfig) -> list:
    """Create Lightning callbacks from config."""
    callbacks = []

    # Auto batch size finder
    if cfg.training.get("auto_batch_size", False):
        callbacks.append(
            BatchSizeFinder(
                mode="binsearch",
                steps_per_trial=3,
                init_val=cfg.training.get("batch_size", 32),
            )
        )
        logger.info("Auto batch size scaling enabled (BatchSizeFinder)")

    # Checkpointing
    callbacks.append(
        ModelCheckpoint(
            dirpath=cfg.output_dir,
            filename="checkpoint-{step}",
            save_top_k=cfg.training.checkpoint.get("save_top_k", 3),
            every_n_train_steps=cfg.training.checkpoint.get("save_interval", 1000),
            save_last=True,
        )
    )

    # GCS upload
    if cfg.get("gcs", {}).get("enabled", False):
        callbacks.append(
            GCSCheckpointCallback(
                bucket=cfg.gcs.bucket,
                experiment_name=cfg.get("experiment_name", "default"),
                stage="lightning",
            )
        )

    # ZClip adaptive gradient clipping
    zclip_cfg = cfg.training.get("gradient_clipping", {})
    if zclip_cfg.get("type", "fixed") == "zclip":
        callbacks.append(
            ZClipCallback(
                z_threshold=zclip_cfg.get("z_threshold", 3.0),
                ema_decay=zclip_cfg.get("ema_decay", 0.99),
                enabled=True,
            )
        )

    # QK clipping
    qk_cfg = cfg.training.get("qk_clip", {})
    if qk_cfg.get("enabled", False):
        callbacks.append(
            QKClipCallback(
                threshold=qk_cfg.get("threshold", 1.0),
                alpha=qk_cfg.get("alpha", 0.99),
            )
        )

    # Lambda warmup for gradual quantization
    lambda_cfg = cfg.training.get("lambda_warmup", {})
    if lambda_cfg.get("enabled", False):
        callbacks.append(
            LambdaWarmupCallback(
                warmup_steps=lambda_cfg.get("warmup_steps", 1000),
                schedule=lambda_cfg.get("schedule", "linear"),
            )
        )

    # Token counting
    if cfg.training.get("total_tokens", 0) > 0:
        callbacks.append(
            TokenCountCallback(
                max_tokens=cfg.training.total_tokens,
                seq_length=cfg.training.max_seq_length,
            )
        )

    # LR monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Progress bar
    callbacks.append(RichProgressBar())

    return callbacks


def create_trainer(cfg: DictConfig, callbacks: list) -> pl.Trainer:
    """Create Lightning Trainer from config."""
    # Determine strategy
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        strategy = cfg.distributed.get("strategy", "ddp")
    else:
        strategy = "auto"

    # Logger
    wandb_logger = None
    if cfg.training.logging.wandb.get("enabled", False):
        wandb_logger = WandbLogger(
            project=cfg.training.logging.wandb.get("project", "wrinklefree"),
            name=cfg.get("experiment_name"),
            save_dir=cfg.output_dir,
        )

    # Handle gradient_clipping as either float or dict
    grad_clip_cfg = cfg.training.get("gradient_clipping", 1.0)
    if isinstance(grad_clip_cfg, (int, float)):
        gradient_clip_val = float(grad_clip_cfg)
    elif grad_clip_cfg.get("type", "fixed") == "fixed":
        gradient_clip_val = grad_clip_cfg.get("max_norm", 1.0)
    else:
        gradient_clip_val = None  # ZClip callback handles it

    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        precision="bf16-mixed",
        strategy=strategy,
        devices="auto",
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg.training.logging.get("log_interval", 10),
        val_check_interval=cfg.training.get("eval_interval", 1000),
        gradient_clip_val=gradient_clip_val,
        enable_checkpointing=True,
        default_root_dir=cfg.output_dir,
    )

    return trainer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Get rank for distributed
    rank = int(os.environ.get("RANK", 0))
    setup_logging(rank)

    # Log config
    if rank == 0:
        logger.info("=" * 60)
        logger.info("WrinkleFree Lightning Training")
        logger.info("=" * 60)
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Load teacher if needed
    teacher = load_teacher_model(cfg)

    # Create objective manager
    objective_manager = create_objective_manager(
        config=cfg.training,
        total_steps=cfg.training.max_steps,
    )

    # Create data module
    datamodule = WrinkleFreeDataModule(
        tokenizer=tokenizer,
        batch_size=cfg.training.batch_size,
        max_length=cfg.training.max_seq_length,
        config_name=cfg.data.get("config_name", "default"),
        with_probes=cfg.training.get("influence", {}).get("enabled", False),
        packed=cfg.training.get("packing", {}).get("enabled", True),
    )

    # Create Lightning module
    # Handle gradient_clipping as either float or dict with max_norm
    grad_clip_cfg = cfg.training.get("gradient_clipping", 1.0)
    if isinstance(grad_clip_cfg, (int, float)):
        gradient_clipping = float(grad_clip_cfg)
    else:
        gradient_clipping = grad_clip_cfg.get("max_norm", 1.0)

    module = WrinkleFreeLightningModule(
        model=model,
        objective_manager=objective_manager,
        teacher_model=teacher.model if teacher else None,
        optimizer_cfg=cfg.training.get("optimizer", {}),
        scheduler_cfg=cfg.training.get("scheduler", {}),
        gradient_clipping=gradient_clipping,
    )

    # Create callbacks
    callbacks = create_callbacks(cfg)

    # Create trainer
    trainer = create_trainer(cfg, callbacks)

    # Train!
    logger.info("Starting training...")
    trainer.fit(module, datamodule)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
