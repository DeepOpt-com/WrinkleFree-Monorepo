#!/usr/bin/env python
"""Main entry point for distillation training.

Usage:
    # Distill against original model (default)
    python scripts/distill.py student.checkpoint_path=outputs/stage2/checkpoint.pt

    # Distill with different teacher
    python scripts/distill.py \
        student.checkpoint_path=outputs/stage2/checkpoint.pt \
        teacher.model_name=meta-llama/Llama-3.2-3B

    # Logits-only (no attention distillation)
    python scripts/distill.py \
        student.checkpoint_path=outputs/stage2/checkpoint.pt \
        distillation=logits_only

    # With vLLM teacher
    python scripts/distill.py \
        student.checkpoint_path=outputs/stage2/checkpoint.pt \
        teacher.use_vllm=true \
        teacher.vllm_url=http://localhost:8000
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def download_from_gcs(gcs_path: str) -> Path:
    """Download a file from GCS to a local temp file."""
    from google.cloud import storage

    # Parse gs://bucket/path
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Not a GCS path: {gcs_path}")

    path_without_scheme = gcs_path[5:]  # Remove "gs://"
    bucket_name = path_without_scheme.split("/")[0]
    blob_path = "/".join(path_without_scheme.split("/")[1:])

    logger.info(f"Downloading from GCS: gs://{bucket_name}/{blob_path}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Download to temp file
    temp_dir = tempfile.mkdtemp()
    local_path = Path(temp_dir) / Path(blob_path).name
    blob.download_to_filename(str(local_path))

    logger.info(f"Downloaded to: {local_path}")
    return local_path


def load_student_model(
    checkpoint_path: Union[Path, str],
    device: torch.device,
    use_flash_attention: bool = True,
) -> tuple:
    """
    Load student model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (local or gs://)
        device: Device to load model on
        use_flash_attention: Whether to use flash attention

    Returns:
        Tuple of (model, original_teacher_name)
    """
    logger.info(f"Loading student model from {checkpoint_path}")

    # Handle GCS paths
    if isinstance(checkpoint_path, str) and checkpoint_path.startswith("gs://"):
        checkpoint_path = download_from_gcs(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Try to infer model type from checkpoint
    # This is a simplified version - real implementation would need
    # to handle different model architectures
    config = checkpoint.get("config", {})
    model_state = checkpoint.get("model_state_dict", checkpoint)

    # Try to get original teacher name from checkpoint metadata
    teacher_name = checkpoint.get("teacher_model_name", config.get("teacher_model", None))

    # Infer model architecture from state dict keys
    first_key = next(iter(model_state.keys()), "")

    # Try different model types
    model = None

    # Try wrinklefree BitNetLlama
    try:
        from wrinklefree.models import BitNetLlama, BitNetConfig

        # Extract config from checkpoint if available
        model_config_dict = checkpoint.get("model_config", {})
        if model_config_dict:
            model_config = BitNetConfig(**model_config_dict)
        else:
            # Try to infer from state dict
            # This is a heuristic - may need adjustment for different models
            embed_weight = model_state.get("model.embed_tokens.weight", None)
            if embed_weight is not None:
                vocab_size = embed_weight.shape[0]
                hidden_size = embed_weight.shape[1]
                model_config = BitNetConfig(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    use_flash_attention=use_flash_attention,
                )
            else:
                raise ValueError("Cannot infer model config from checkpoint")

        model = BitNetLlama(model_config)
        model.load_state_dict(model_state, strict=False)
        logger.info("Loaded BitNetLlama model")

    except ImportError:
        logger.warning("wrinklefree not available, trying transformers")

    # Fallback to transformers
    if model is None:
        try:
            from transformers import AutoModelForCausalLM

            # This would need more context about model type
            # For now, raise an error
            raise NotImplementedError(
                "Generic transformers model loading not yet implemented. "
                "Please use wrinklefree BitNetLlama models."
            )
        except Exception as e:
            raise ValueError(f"Could not load model from checkpoint: {e}")

    model = model.to(device)

    return model, teacher_name


def run_distillation(
    cfg: DictConfig,
    checkpoint_path: Path,
    output_dir: Path,
    teacher_model_name: Optional[str] = None,
) -> None:
    """
    Run knowledge distillation training.

    Args:
        cfg: Hydra configuration
        checkpoint_path: Path to student checkpoint
        output_dir: Output directory for checkpoints
        teacher_model_name: Optional teacher model name (None = infer from checkpoint)
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load student model
    needs_attention = cfg.distillation.gamma_attention > 0
    use_flash = not needs_attention  # Disable flash if attention distillation enabled

    student, inferred_teacher = load_student_model(
        checkpoint_path,
        device,
        use_flash_attention=use_flash,
    )

    # Determine teacher model
    if teacher_model_name is None:
        teacher_model_name = inferred_teacher
        if teacher_model_name is None:
            raise ValueError(
                "Could not infer teacher model from checkpoint. "
                "Please specify teacher.model_name explicitly."
            )
    logger.info(f"Using teacher model: {teacher_model_name}")

    # Create teacher
    from distillation.teachers import create_teacher

    teacher = create_teacher(
        model_name=teacher_model_name,
        use_vllm=cfg.teacher.use_vllm,
        vllm_base_url=cfg.teacher.vllm_url,
        device=device,
        use_eager_attention=needs_attention,
        load_in_4bit=cfg.teacher.load_in_4bit,
        offload_to_cpu=cfg.teacher.offload_to_cpu,
    )

    # Create dataloaders from data_handler
    try:
        from data_handler.data import create_dataloader, load_data_config
        from transformers import AutoTokenizer

        # Load tokenizer from teacher
        tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load data config
        data_config = load_data_config(cfg.data.config_name)

        # Create dataloader
        train_dataloader, mixed_dataset = create_dataloader(
            config=data_config,
            tokenizer=tokenizer,
            batch_size=cfg.training.batch_size,
            max_length=cfg.data.max_seq_length,
            num_workers=cfg.data.num_workers,
        )

    except ImportError:
        logger.error(
            "cheapertraining not available. Install it with: "
            "cd ../cheapertraining && uv sync"
        )
        raise

    # Create trainer config
    from distillation.training.config import DistillationConfig, LossConfig, TeacherConfig

    distill_config = DistillationConfig(
        student_checkpoint_path=str(checkpoint_path),
        teacher=TeacherConfig(
            model_name=teacher_model_name,
            use_vllm=cfg.teacher.use_vllm,
            vllm_url=cfg.teacher.vllm_url,
            load_in_4bit=cfg.teacher.load_in_4bit,
        ),
        loss=LossConfig(
            lambda_logits=cfg.distillation.lambda_logits,
            gamma_attention=cfg.distillation.gamma_attention,
            temperature=cfg.distillation.temperature,
            use_relation_distill=cfg.distillation.attention.use_relation_distill,
            distill_layer=cfg.distillation.attention.distill_layer,
        ),
        max_steps=cfg.training.max_steps,
        batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_clipping=cfg.training.gradient_clipping,
        optimizer_type=cfg.optimizer.type,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        scheduler_type=cfg.scheduler.type,
        warmup_steps=cfg.scheduler.warmup_steps,
        min_lr_ratio=cfg.scheduler.min_lr_ratio,
        save_interval=cfg.checkpoint.save_interval,
        keep_last_n=cfg.checkpoint.keep_last_n,
        output_dir=str(output_dir),
        log_interval=cfg.logging.log_interval,
        eval_interval=cfg.logging.eval_interval,
        wandb_enabled=cfg.logging.wandb.enabled,
        wandb_project=cfg.logging.wandb.project,
        influence_enabled=cfg.influence.enabled,
        influence_update_interval=cfg.influence.update_interval,
        influence_learning_rate=cfg.influence.learning_rate,
        data_config_name=cfg.data.config_name,
        max_seq_length=cfg.data.max_seq_length,
    )

    # Create trainer
    from distillation.training import DistillationTrainer

    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        train_dataloader=train_dataloader,
        config=distill_config,
        mixed_dataset=mixed_dataset,
        device=device,
    )

    # Train
    metrics = trainer.train()

    logger.info(f"Distillation complete: {metrics}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for distillation training."""
    # Log config
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Validate required fields
    if not cfg.student.checkpoint_path or cfg.student.checkpoint_path == "???":
        raise ValueError(
            "student.checkpoint_path is required. "
            "Usage: python scripts/distill.py student.checkpoint_path=path/to/checkpoint.pt"
        )

    checkpoint_path_str = cfg.student.checkpoint_path
    # Handle GCS paths (gs://) - don't validate locally
    if checkpoint_path_str.startswith("gs://"):
        checkpoint_path = checkpoint_path_str  # Keep as string for GCS
    else:
        checkpoint_path = Path(checkpoint_path_str)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get teacher model name (None = infer from checkpoint)
    teacher_model_name = cfg.teacher.model_name

    # Run distillation
    run_distillation(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        teacher_model_name=teacher_model_name,
    )


if __name__ == "__main__":
    main()
