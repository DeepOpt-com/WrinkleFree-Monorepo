#!/usr/bin/env python3
"""Main conversion script with Hydra configuration.

Usage:
    uv run python scripts/convert.py model=qwen3_4b source.path=hf://org/model
    uv run python scripts/convert.py model=smollm2_135m conversion.total_tokens=500000000
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run DLM conversion with Hydra configuration."""
    logger.info("Starting DLM conversion")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Validate required fields
    if cfg.source.path is None:
        raise ValueError("source.path must be specified")

    # Import here to avoid slow startup
    from wf_dlm_converter.models import load_bitnet_checkpoint, BlockDiffusionAdapter
    from wf_dlm_converter.conversion import DiffusionFineTuner, save_dlm_checkpoint, DLMConfig
    from wf_dlm_converter.conversion.training import TrainingConfig

    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    # Set seed
    torch.manual_seed(cfg.seed)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {cfg.source.path}")
    model, tokenizer, model_config = load_bitnet_checkpoint(
        cfg.source.path,
        device="cuda",
        dtype=torch.bfloat16,
    )

    logger.info(f"Model config: {model_config}")

    # Adapt for block diffusion
    logger.info("Adapting model for block diffusion")
    adapter = BlockDiffusionAdapter(
        block_size=cfg.block_diffusion.block_size,
        num_diffusion_steps=cfg.block_diffusion.num_diffusion_steps,
        noise_schedule=cfg.block_diffusion.noise_schedule,
        preserve_bitlinear=cfg.block_diffusion.preserve_bitlinear,
    )
    adapted_model = adapter.adapt_model(model)

    # Create training config from Hydra config
    conversion_cfg = cfg.conversion if "conversion" in cfg else cfg.get("conversion", {})
    training_config = TrainingConfig(
        total_tokens=conversion_cfg.get("total_tokens", 1_000_000_000),
        max_seq_length=conversion_cfg.get("max_seq_length", 512),
        batch_size=conversion_cfg.get("batch_size", 8),
        gradient_accumulation_steps=conversion_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=conversion_cfg.get("optimizer", {}).get("lr", 5e-5),
        weight_decay=conversion_cfg.get("optimizer", {}).get("weight_decay", 0.01),
        warmup_steps=conversion_cfg.get("scheduler", {}).get("warmup_steps", 1000),
        block_size=cfg.block_diffusion.block_size,
        num_diffusion_steps=cfg.block_diffusion.num_diffusion_steps,
        output_dir=cfg.output_dir,
        wandb_project=cfg.logging.wandb.project if cfg.logging.wandb.enabled else None,
    )

    # Load training data
    logger.info("Loading training data")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=training_config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    dataloader = DataLoader(
        tokenized,
        batch_size=training_config.batch_size,
    )

    # Fine-tune
    logger.info(f"Starting fine-tuning for {training_config.total_tokens:,} tokens")
    finetuner = DiffusionFineTuner(
        model=adapted_model,
        config=training_config,
        tokenizer=tokenizer,
    )

    trained_model = finetuner.train(dataloader)

    # Save checkpoint
    output_path = Path(cfg.output_dir) / cfg.model.name
    dlm_config = DLMConfig(
        block_size=cfg.block_diffusion.block_size,
        num_diffusion_steps=cfg.block_diffusion.num_diffusion_steps,
        noise_schedule=cfg.block_diffusion.noise_schedule,
        source_model=cfg.model.name,
        source_checkpoint=str(cfg.source.path),
        total_tokens_trained=finetuner.tokens_seen,
        training_loss=finetuner.best_loss,
    )

    final_path = save_dlm_checkpoint(
        model=trained_model,
        output_path=output_path,
        tokenizer=tokenizer,
        dlm_config=dlm_config,
    )

    logger.info(f"Conversion complete! Output: {final_path}")


if __name__ == "__main__":
    main()
