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
from pytorch_lightning.strategies import FSDPStrategy
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata
from omegaconf._utils import ValueKind
from transformers import AutoModelForCausalLM, AutoTokenizer

# PyTorch 2.6+ requires explicit safe globals for omegaconf types in checkpoints
torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, ValueKind])

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bitnet_arch.conversion import auto_convert_if_needed
from wrinklefree.lightning import (
    GCSCheckpointCallback,
    InfluenceAwareBatchSizeFinder,
    InfluenceTrackerCallback,
    LambdaWarmupCallback,
    MuonClipInitCallback,
    QKClipCallback,
    RunManagerCallback,
    TokenCountCallback,
    WrinkleFreeDataModule,
    WrinkleFreeLightningModule,
    ZClipCallback,
)
from wrinklefree.meta import MetaOptimizerCallback, MetaOptimizationConfig
from wrinklefree.objectives import create_objective_manager
from wrinklefree.teachers import HiddenStateTeacher

logger = logging.getLogger(__name__)


def resolve_checkpoint_path(
    cfg: DictConfig,
    stage: str = "stage2",
) -> Path | str | None:
    """Resolve checkpoint path from local, GCS, or HuggingFace.

    Priority:
    1. Explicit checkpoint path in config (training.checkpoint.path or model.path)
    2. Local checkpoint directory
    3. GCS bucket download
    4. HuggingFace Hub (model.pretrained_name or model.name)

    Args:
        cfg: Hydra config
        stage: Training stage for GCS lookup (e.g., "stage1_9", "stage2")

    Returns:
        Path to checkpoint, HF model name, or None if not found
    """
    # Check for explicit checkpoint path
    ckpt_path = cfg.training.get("checkpoint", {}).get("path")
    if ckpt_path:
        ckpt_path = Path(ckpt_path)
        if ckpt_path.exists():
            logger.info(f"Using explicit checkpoint: {ckpt_path}")
            return ckpt_path

    # Check for model.path (local safetensors)
    model_path = cfg.model.get("path")
    if model_path:
        model_path = Path(model_path)
        if model_path.exists():
            logger.info(f"Using local model path: {model_path}")
            return model_path

    # Try GCS bucket
    gcs_config = cfg.get("gcs", {})
    if gcs_config.get("enabled", False):
        bucket = gcs_config.get("bucket", "wrinklefree-checkpoints")
        experiment_name = cfg.get("experiment_name", "default")
        gcs_prefix = f"checkpoints/{experiment_name}"

        try:
            from wrinklefree.training import download_checkpoint_from_gcs

            cache_dir = Path(cfg.output_dir) / ".checkpoint_cache"
            gcs_path = download_checkpoint_from_gcs(
                bucket_name=bucket,
                stage=f"{stage}_checkpoint",
                local_dir=cache_dir / stage,
                prefix=gcs_prefix,
            )
            if gcs_path:
                logger.info(f"Downloaded checkpoint from GCS: {gcs_path}")
                return gcs_path
        except ImportError:
            logger.debug("GCS download not available (google-cloud-storage not installed)")
        except Exception as e:
            logger.warning(f"GCS checkpoint download failed: {e}")

    # Fall back to HuggingFace Hub
    hf_name = cfg.model.get("pretrained_name") or cfg.model.get("name")
    if hf_name:
        logger.info(f"Using HuggingFace model: {hf_name}")
        return hf_name

    return None


def setup_logging(rank: int) -> None:
    """Setup logging configuration."""
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_model_and_tokenizer(cfg: DictConfig, device: str = "cuda"):
    """Load model and tokenizer from config.

    Uses resolve_checkpoint_path for intelligent checkpoint resolution:
    - Local path → GCS → HuggingFace Hub

    Handles special stages:
    - lrc_calibration: Converts BitLinear → BitLinearLRC and freezes non-LRC params
    """
    # Resolve checkpoint path (local > GCS > HuggingFace)
    stage = cfg.training.get("stage", "stage2")
    model_path = resolve_checkpoint_path(cfg, stage=stage)

    # Fallback to model.name if nothing found
    if model_path is None:
        model_path = cfg.model.name

    logger.info(f"Loading model: {model_path}")

    # Convert Path to string for transformers compatibility
    model_path_str = str(model_path) if isinstance(model_path, Path) else model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path_str)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path_str,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Auto-convert to BitNet if needed
    if cfg.training.get("auto_convert", {}).get("enabled", True):
        exclude_layers = cfg.training.get("auto_convert", {}).get("exclude_layers", None)
        # insert_subln=False by default to preserve pretrained weights
        # Set to True only if running Stage 1.9 layer-wise distillation afterward
        insert_subln = cfg.training.get("auto_convert", {}).get("insert_subln", False)
        model = auto_convert_if_needed(
            model,
            hidden_size=cfg.model.hidden_size,
            intermediate_size=cfg.model.intermediate_size,
            exclude_layers=exclude_layers,
            insert_subln=insert_subln,
        )
        logger.info(f"Model converted to BitNet (insert_subln={insert_subln})")

    # Handle LRC calibration stage
    if stage == "lrc_calibration":
        try:
            from bitnet_arch import convert_bitlinear_to_lrc, freeze_model_except_lrc

            lrc_cfg = cfg.training.get("lrc", {})
            rank_percentage = lrc_cfg.get("rank_percentage", 0.1)
            init_method = lrc_cfg.get("init_method", "zeros")
            keep_original_weight = lrc_cfg.get("keep_original_weight", True)
            trainable_weight = lrc_cfg.get("trainable_weight", False)

            logger.info(
                f"LRC Calibration: Converting BitLinear → BitLinearLRC "
                f"(rank={rank_percentage*100:.0f}%, init={init_method}, "
                f"keep_weight={keep_original_weight}, trainable={trainable_weight})"
            )

            model = convert_bitlinear_to_lrc(
                model,
                rank_percentage=rank_percentage,
                init_method=init_method,
                keep_original_weight=keep_original_weight,
                trainable_weight=trainable_weight,
            )

            # Freeze all parameters except LRC matrices (U, V)
            freeze_stats = freeze_model_except_lrc(model)
            logger.info(
                f"LRC: Trainable={freeze_stats['trainable']:,}, "
                f"Frozen={freeze_stats['frozen']:,}"
            )

        except ImportError as e:
            logger.error(f"LRC calibration requires bitnet_arch package: {e}")
            raise

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

    # Run manager for GCS auto-resume with fingerprinting
    skip_recovery = cfg.get("skip_recovery", False)
    run_manager_cb = RunManagerCallback(
        config=cfg,
        skip_recovery=skip_recovery,
        skip_completed=cfg.get("resume", {}).get("skip_completed", True),
    )
    callbacks.append(run_manager_cb)

    # Influence tracking callback - created early so InfluenceAwareBatchSizeFinder can reference it
    influence_cfg = cfg.training.get("influence", {})
    influence_callback = None
    if influence_cfg.get("enabled", False):
        influence_callback = InfluenceTrackerCallback(config=cfg)
        callbacks.append(influence_callback)
        logger.info(
            f"Influence tracking enabled: update_interval={influence_cfg.get('update_interval', 1000)}, "
            f"warmup={influence_cfg.get('warmup_steps', 500)}"
        )

    # Auto batch size finder
    if cfg.training.get("auto_batch_size", False):
        # Use InfluenceAwareBatchSizeFinder when influence is enabled
        # This initializes influence cache BEFORE batch size search to account for memory
        if influence_callback is not None:
            callbacks.append(
                InfluenceAwareBatchSizeFinder(
                    influence_callback=influence_callback,
                    mode="binsearch",
                    steps_per_trial=3,
                    init_val=cfg.training.get("batch_size", 32),
                )
            )
            logger.info(
                "InfluenceAwareBatchSizeFinder enabled "
                "(initializes influence cache before batch size search)"
            )
        else:
            callbacks.append(
                BatchSizeFinder(
                    mode="binsearch",
                    steps_per_trial=3,
                    init_val=cfg.training.get("batch_size", 32),
                )
            )
            logger.info("Auto batch size scaling enabled (BatchSizeFinder)")

        # MuonClipInitCallback: Required when using MuonClip with BatchSizeFinder
        # BatchSizeFinder cycles model.eval()/train() which breaks MuonClip's hooks
        # due to an upstream bug where is_registered flag isn't reset on remove_hooks()
        # This callback re-registers hooks after BatchSizeFinder completes
        optimizer_type = cfg.training.get("optimizer", {}).get("type", "adamw")
        if optimizer_type.lower() == "muonclip":
            callbacks.append(MuonClipInitCallback())
            logger.info("MuonClipInitCallback added (required for BatchSizeFinder + MuonClip)")

    # Checkpointing - save every N steps
    save_interval = cfg.training.checkpoint.get("save_interval", 500)
    checkpoint_dir = Path(cfg.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint dir: {checkpoint_dir}, save_interval: {save_interval}")

    callbacks.append(
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="step_{step:06d}",
            save_top_k=-1,  # Keep all checkpoints
            every_n_train_steps=save_interval,
            save_last=True,
            verbose=True,  # Log when saving
        )
    )

    # GCS upload with DLM config
    if cfg.get("gcs", {}).get("enabled", False):
        # Build DLM config for inference compatibility
        dlm_cfg = cfg.training.get("objectives", {}).get("dlm", {})
        dlm_config = None
        if dlm_cfg.get("enabled", False):
            dlm_config = {
                "mask_token_id": dlm_cfg.get("mask_token_id", 0),
                "mask_prob": dlm_cfg.get("mask_prob", 0.15),
                "ignore_index": dlm_cfg.get("ignore_index", -100),
                "training_method": cfg.training.get("stage", "unified-dlm"),
            }
            logger.info(f"DLM config for inference: {dlm_config}")

        callbacks.append(
            GCSCheckpointCallback(
                bucket=cfg.gcs.bucket,
                experiment_name=cfg.get("experiment_name", "default"),
                stage="lightning",
                dlm_config=dlm_config,
            )
        )

    # ZClip adaptive gradient clipping (only if gradient_clipping is a dict with type=zclip)
    zclip_cfg = cfg.training.get("gradient_clipping", {})
    if isinstance(zclip_cfg, dict) and zclip_cfg.get("type", "fixed") == "zclip":
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

    # Meta-optimization outer loop (supersedes influence when enabled)
    meta_opt_cfg = cfg.training.get("meta_optimization", {})
    if meta_opt_cfg.get("enabled", False):
        # Build nested configs for LDC-MTL and ODM
        ldc_mtl_cfg = meta_opt_cfg.get("ldc_mtl", {})
        odm_cfg = meta_opt_cfg.get("odm", {})

        from wrinklefree.meta.config import LDCMTLConfig, ODMConfig

        ldc_mtl_config = LDCMTLConfig(
            enabled=ldc_mtl_cfg.get("enabled", True),
            lambda_penalty=ldc_mtl_cfg.get("lambda_penalty", 0.1),
            hidden_dim=ldc_mtl_cfg.get("hidden_dim", 32),
            router_lr=ldc_mtl_cfg.get("router_lr", 1e-3),
        )

        odm_config = ODMConfig(
            enabled=odm_cfg.get("enabled", True),
            reward_smoothing=odm_cfg.get("reward_smoothing", 0.9),
            warmup_ratio=odm_cfg.get("warmup_ratio", 0.01),
            min_weight=odm_cfg.get("min_weight", 0.05),
            max_weight=odm_cfg.get("max_weight", 0.60),
        )

        meta_config = MetaOptimizationConfig(
            enabled=True,
            ldc_mtl=ldc_mtl_config,
            odm=odm_config,
            log_interval=meta_opt_cfg.get("log_interval", 100),
        )

        callbacks.append(MetaOptimizerCallback(config=meta_config))
        logger.info(
            f"Meta-optimization enabled: "
            f"ldc_mtl={meta_config.ldc_mtl.enabled}, "
            f"odm={meta_config.odm.enabled}, "
            f"log_interval={meta_config.log_interval}"
        )
    else:
        # Fallback to influence-based data remixing only
        influence_cfg = cfg.training.get("influence", {})
        if influence_cfg.get("enabled", False):
            callbacks.append(InfluenceTrackerCallback(config=cfg))
            logger.info(
                f"Influence tracking enabled: update_interval={influence_cfg.get('update_interval', 1000)}, "
                f"warmup={influence_cfg.get('warmup_steps', 500)}"
            )
    # LR monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Progress bar - disabled for non-TTY environments to avoid potential hangs
    # callbacks.append(RichProgressBar())

    return callbacks


def create_trainer(cfg: DictConfig, callbacks: list) -> pl.Trainer:
    """Create Lightning Trainer from config."""
    # Determine strategy based on number of available GPUs
    # NOTE: WORLD_SIZE isn't set yet (Lightning sets it later), so use cuda.device_count()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f"Detected {num_gpus} GPUs for training")

    if num_gpus > 1:
        strategy_name = cfg.distributed.get("strategy", "ddp")
        if strategy_name == "fsdp":
            # Create proper FSDPStrategy with mixed precision for bfloat16 training
            # Reference: https://lightning.ai/docs/pytorch/stable/api/lightning_fabric.strategies.FSDPStrategy.html
            from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

            # FSDP mixed precision config (bfloat16 compute, fp32 reduce for stability)
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,  # Reduce in bf16 to match params
                buffer_dtype=torch.bfloat16,
            )

            # Get sharding strategy from config (default to FULL_SHARD = ZeRO-3)
            fsdp_cfg = cfg.distributed.get("fsdp", {})
            sharding_strategy_str = fsdp_cfg.get("sharding_strategy", "FULL_SHARD")
            sharding_strategy = getattr(ShardingStrategy, sharding_strategy_str)

            strategy = FSDPStrategy(
                sharding_strategy=sharding_strategy,
                mixed_precision=mp_policy,
                activation_checkpointing_policy={torch.nn.TransformerEncoderLayer}
                    if fsdp_cfg.get("activation_checkpointing", {}).get("enabled", False)
                    else None,
                limit_all_gathers=fsdp_cfg.get("limit_all_gathers", True),
            )
            logger.info(f"Using FSDPStrategy with {sharding_strategy_str} sharding and bf16 mixed precision")
        else:
            strategy = strategy_name
    else:
        strategy = "auto"

    # Logger
    wandb_logger = None
    if cfg.training.logging.wandb.get("enabled", False):
        # Generate unique run ID to avoid conflicts
        import uuid
        run_id = f"{cfg.get('experiment_name', 'run')}-{uuid.uuid4().hex[:8]}"
        wandb_logger = WandbLogger(
            project=cfg.training.logging.wandb.get("project", "wrinklefree"),
            name=cfg.get("experiment_name"),
            save_dir=cfg.output_dir,
            id=run_id,  # Unique ID ensures fresh run
        )

    # Handle gradient_clipping as either float or dict
    grad_clip_cfg = cfg.training.get("gradient_clipping", 1.0)
    if isinstance(grad_clip_cfg, (int, float)):
        gradient_clip_val = float(grad_clip_cfg)
    elif grad_clip_cfg.get("type", "fixed") == "fixed":
        gradient_clip_val = grad_clip_cfg.get("max_norm", 1.0)
    else:
        gradient_clip_val = None  # ZClip callback handles it

    # Validation config
    val_cfg = cfg.training.get("validation", {})
    val_enabled = val_cfg.get("enabled", False)

    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        precision="bf16-mixed",
        strategy=strategy,
        devices="auto",
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg.training.logging.get("log_interval", 10),
        # Validation settings (C4 perplexity every N steps)
        val_check_interval=val_cfg.get("val_check_interval", 500) if val_enabled else None,
        limit_val_batches=val_cfg.get("limit_val_batches", 50) if val_enabled else 0,
        num_sanity_val_steps=0,  # Disable sanity check to avoid potential hangs
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

    # Enable TF32 for Ampere+ GPUs (A100, H100, RTX 30xx/40xx)
    # TF32 accelerates bf16 matmul by using reduced-precision accumulation internally.
    # This gives 10-20% speedup with negligible accuracy impact for LLM training.
    # See: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if rank == 0:
        logger.info("TF32 enabled for CUDA matmul and cuDNN")

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

    # Create data module with validation support
    val_cfg = cfg.training.get("validation", {})
    val_enabled = val_cfg.get("enabled", False)

    # Determine num_workers: from config, env var, or default
    # For influence + probe gradient caching, use 0 to avoid memory explosion
    num_workers = cfg.data.get("num_workers", None)
    if num_workers is None:
        num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "4"))

    datamodule = WrinkleFreeDataModule(
        tokenizer=tokenizer,
        batch_size=cfg.training.batch_size,
        max_length=cfg.training.max_seq_length,
        config_name=cfg.data.get("config_name", "default"),
        with_probes=cfg.training.get("influence", {}).get("enabled", False),
        packed=cfg.training.get("packing", {}).get("enabled", True),
        num_workers=num_workers,
        # Validation (C4 perplexity)
        val_config_name=val_cfg.get("config_name") if val_enabled else None,
        val_batch_size=val_cfg.get("batch_size", 8),
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
