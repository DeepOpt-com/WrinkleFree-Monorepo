#!/usr/bin/env python3
"""Modal training for WrinkleFree BitNet.

Simple, self-contained Modal training script. Mounts local code directly
(no git cloning) for reliability.

Usage:
    # Full pipeline on A10G (all stages)
    modal run modal_train.py::run_full_pipeline

    # Single stage
    modal run modal_train.py --model smollm2_135m --stage 2

    # With Hydra overrides
    modal run modal_train.py --model smollm2_135m --stage 2 --overrides "training.max_steps=100"

Setup:
    # Create secrets (one-time)
    modal secret create wandb-api-key WANDB_API_KEY=<your-key>
    modal secret create hf-token HF_TOKEN=<your-token>
"""
from __future__ import annotations

import modal

# =============================================================================
# Modal App Configuration
# =============================================================================

app = modal.App("wrinklefree-training")

# Persistent volumes
checkpoints_volume = modal.Volume.from_name(
    "wrinklefree-checkpoints", create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name(
    "wrinklefree-hf-cache", create_if_missing=True
)

# Stage to config mapping
STAGE_CONFIGS = {
    1: "stage1_subln",
    1.9: "stage1_9_layerwise",
    2: "stage2_pretrain",
    3: "stage3_distill",
}

# A10G batch sizes for full GPU utilization (from configs/gpu/a10g_24gb.yaml)
A10G_BATCH_SIZES = {
    "smollm2_135m": {
        1.9: {"batch_size": 8, "gradient_accumulation_steps": 8},
        2: {"batch_size": 8, "gradient_accumulation_steps": 8},  # Reduced from 32 to avoid OOM
        3: {"batch_size": 8, "gradient_accumulation_steps": 8},
    },
    "qwen3_4b": {
        1.9: {"batch_size": 2, "gradient_accumulation_steps": 32},
        2: {"batch_size": 4, "gradient_accumulation_steps": 16},
    },
}

# Build training image with all dependencies
training_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        # Core ML
        "torch>=2.5.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.27.0",
        "safetensors>=0.4.0",
        # Config
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        # Training
        "bitsandbytes>=0.43.0",
        "muon-optimizer>=0.1.0",
        "torchao>=0.7.0",
        # Logging
        "wandb>=0.16.0",
        "tqdm>=4.66.0",
        # Data
        "numpy>=1.26.0",
        "einops>=0.7.0",
        "zstandard>=0.22.0",
        # Cloud/HF
        "huggingface_hub>=0.20.0",
        "hf_transfer>=0.1.0",
        "google-cloud-storage>=2.14.0",
        # CheaperTraining dependencies
        "datasketch>=1.6.0",  # MinHash for influence functions
        "sentencepiece>=0.2.0",
        "tiktoken>=0.7.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Use PyTorch SDPA instead of flash-attn (almost as fast, no build issues)
        "TRANSFORMERS_ATTN_IMPLEMENTATION": "sdpa",
    })
    # Mount CheaperTraining (influence functions dependency)
    .add_local_dir(
        "../WrinkleFree-CheaperTraining",
        "/root/cheapertraining",
        copy=True,
        ignore=[".git", "__pycache__", ".venv", "outputs", ".pytest_cache", "*.egg-info"],
    )
    # Mount main wrinklefree code
    .add_local_dir(
        ".",
        "/root/wrinklefree",
        copy=True,
        ignore=[".git", "__pycache__", ".venv", "outputs", ".pytest_cache", "*.egg-info", ".ruff_cache", ".mypy_cache", "wandb"],
    )
    # Install both packages (cheapertraining first, then wrinklefree)
    .run_commands(
        "pip install -e /root/cheapertraining && pip install -e /root/wrinklefree",
    )
)


# =============================================================================
# A10G Training Function (24GB VRAM)
# =============================================================================

@app.function(
    image=training_image,
    gpu="A10G",
    memory=65536,  # 64GB RAM for influence functions and probe data
    timeout=24 * 60 * 60,  # 24 hours
    volumes={
        "/checkpoints": checkpoints_volume,
        "/hf_cache": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-api-key"),
        modal.Secret.from_name("hf-token"),
    ],
)
def train_a10g(
    model: str = "smollm2_135m",
    stage: float = 2,
    overrides: list[str] | None = None,
    data: str | None = None,
    max_steps: int | None = None,
    wandb_api_key: str | None = None,
):
    """
    Run WrinkleFree BitNet training on A10G with optimized batch sizes.

    Args:
        model: Model config name (smollm2_135m, qwen3_4b)
        stage: Training stage (1, 1.9, 2, 3)
        overrides: List of Hydra overrides
        data: Data config override (default: mixed_pretrain for stage 2+)
        max_steps: Optional max steps override
        wandb_api_key: Optional W&B API key (overrides Modal secret)
    """
    import os
    import subprocess
    import sys

    # Set environment
    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache"
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Set wandb API key if provided via CLI (overrides Modal secret)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    # Verify credentials
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    print(f"WandB API key: {'configured' if wandb_key else 'MISSING!'}")
    print(f"HF Token: {'configured' if hf_token else 'MISSING!'}")

    # Build training command
    stage_config = STAGE_CONFIGS.get(stage)
    if not stage_config:
        raise ValueError(f"Unknown stage: {stage}. Valid: {list(STAGE_CONFIGS.keys())}")

    cmd = [
        sys.executable,
        "scripts/train.py",
        f"model={model}",
        f"training={stage_config}",
        "output_dir=/checkpoints",
    ]

    # Add data config (mixed_pretrain for stage 2+ to enable influence)
    if data:
        cmd.append(f"data={data}")
    elif stage >= 2:
        cmd.append("data=mixed_pretrain")

    # Add A10G-optimized batch sizes for full GPU utilization
    if model in A10G_BATCH_SIZES and stage in A10G_BATCH_SIZES[model]:
        batch_cfg = A10G_BATCH_SIZES[model][stage]
        cmd.append(f"training.batch_size={batch_cfg['batch_size']}")
        cmd.append(f"training.gradient_accumulation_steps={batch_cfg['gradient_accumulation_steps']}")
        print(f"A10G batch config: batch_size={batch_cfg['batch_size']}, grad_accum={batch_cfg['gradient_accumulation_steps']}")

    # Add max_steps if specified
    if max_steps:
        cmd.append(f"training.max_steps={max_steps}")

    # Add any additional overrides
    if overrides:
        cmd.extend(overrides)

    # Print command
    print("\n" + "=" * 60)
    print("WRINKLEFREE MODAL TRAINING (A10G)")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Stage: {stage} ({stage_config})")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    # Run training
    result = subprocess.run(
        cmd,
        cwd="/root/wrinklefree",
        check=False,
    )

    # Commit volumes
    print("\nCommitting volumes...")
    checkpoints_volume.commit()
    hf_cache_volume.commit()
    print("Volumes committed.")

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    return {"status": "success", "model": model, "stage": stage}


# =============================================================================
# Full Pipeline Function
# =============================================================================

@app.function(
    image=training_image,
    gpu="A10G",
    memory=65536,  # 64GB RAM for influence functions and probe data
    timeout=24 * 60 * 60,  # 24 hours max (Modal limit)
    volumes={
        "/checkpoints": checkpoints_volume,
        "/hf_cache": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-api-key"),
        modal.Secret.from_name("hf-token"),
    ],
)
def train_full_pipeline(
    model: str = "smollm2_135m",
    stage2_max_steps: int | None = None,
    stage3_max_steps: int | None = None,
    wandb_api_key: str | None = None,
):
    """
    Run full training pipeline: Stage 1 -> 1.9 -> 2 -> 3.

    Uses mixed_pretrain data with influence functions for Stage 2.
    Optimized batch sizes for A10G (full GPU utilization).

    Args:
        model: Model config name
        stage2_max_steps: Optional max steps for stage 2 (default: full 10B tokens)
        stage3_max_steps: Optional max steps for stage 3
        wandb_api_key: Optional W&B API key (overrides Modal secret)
    """
    import os
    import subprocess
    import sys

    # Set wandb API key if provided via CLI (overrides Modal secret)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    # Set environment
    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache"
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Verify credentials
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    print(f"WandB API key: {'configured' if wandb_key else 'MISSING!'}")
    print(f"HF Token: {'configured' if hf_token else 'MISSING!'}")

    def run_stage(stage: float, extra_args: list[str] | None = None):
        """Run a single training stage."""
        stage_config = STAGE_CONFIGS[stage]

        cmd = [
            sys.executable,
            "scripts/train.py",
            f"model={model}",
            f"training={stage_config}",
            "output_dir=/checkpoints",
        ]

        # Add appropriate data config for each stage
        if stage == 1.9:
            cmd.append("data=fineweb")  # Layer-wise distillation uses fineweb
        elif stage == 2:
            cmd.append("data=mixed_pretrain")  # Stage 2 uses mixed with influence
        elif stage == 3:
            cmd.append("data=downstream")  # Stage 3 uses fine-tuning data

        # Add A10G batch sizes
        if model in A10G_BATCH_SIZES and stage in A10G_BATCH_SIZES[model]:
            batch_cfg = A10G_BATCH_SIZES[model][stage]
            cmd.append(f"training.batch_size={batch_cfg['batch_size']}")
            cmd.append(f"training.gradient_accumulation_steps={batch_cfg['gradient_accumulation_steps']}")

        if extra_args:
            cmd.extend(extra_args)

        print("\n" + "=" * 60)
        print(f"STAGE {stage}: {stage_config}")
        print("=" * 60)
        print(f"Command: {' '.join(cmd)}")
        print("=" * 60 + "\n")

        result = subprocess.run(cmd, cwd="/root/wrinklefree", check=False)

        # Commit volumes after each stage
        checkpoints_volume.commit()
        hf_cache_volume.commit()

        if result.returncode != 0:
            raise RuntimeError(f"Stage {stage} failed with exit code {result.returncode}")

        print(f"\n[OK] Stage {stage} complete!\n")

    print("\n" + "#" * 60)
    print("# WRINKLEFREE FULL TRAINING PIPELINE")
    print("# Model:", model)
    print("# GPU: A10G (24GB)")
    print("# Data: mixed_pretrain with influence functions")
    print("#" * 60 + "\n")

    # Stage 1: SubLN Insertion (conversion only, no training)
    run_stage(1)

    # Stage 1.9: Layer-wise Distillation
    run_stage(1.9)

    # Stage 2: Continue Pre-training with influence-based data selection
    stage2_args = []
    if stage2_max_steps:
        stage2_args.append(f"training.max_steps={stage2_max_steps}")
    run_stage(2, stage2_args)

    # Stage 3: Distillation Fine-tuning
    stage3_args = []
    if stage3_max_steps:
        stage3_args.append(f"training.max_steps={stage3_max_steps}")
    run_stage(3, stage3_args)

    print("\n" + "#" * 60)
    print("# FULL PIPELINE COMPLETE!")
    print("#" * 60 + "\n")

    return {"status": "success", "model": model, "stages": [1, 1.9, 2, 3]}


# =============================================================================
# CLI Entrypoints
# =============================================================================

@app.local_entrypoint()
def main(
    model: str = "smollm2_135m",
    stage: float = 2,
    overrides: str = "",
    data: str = "",
    max_steps: int = 0,
):
    """
    Launch single-stage Modal training on A10G.

    Args:
        model: Model config name
        stage: Training stage (1, 1.9, 2, 3)
        overrides: Hydra overrides (space-separated)
        data: Data config override
        max_steps: Max training steps (0 = use config default)

    W&B logging: Set WANDB_API_KEY environment variable locally, or
    create Modal secret: modal secret create wandb-api-key WANDB_API_KEY=<key>
    """
    import os

    # Read WANDB_API_KEY from local environment and pass to remote
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    print(f"\nLaunching training on Modal (A10G)...")
    print(f"  Model: {model}")
    print(f"  Stage: {stage}")
    print(f"  W&B: {'enabled (key from local env)' if wandb_api_key else 'using Modal secret'}")
    if overrides:
        print(f"  Overrides: {overrides}")
    if data:
        print(f"  Data: {data}")
    if max_steps:
        print(f"  Max steps: {max_steps}")
    print()

    override_list = overrides.split() if overrides else None

    result = train_a10g.remote(
        model=model,
        stage=stage,
        overrides=override_list,
        data=data if data else None,
        max_steps=max_steps if max_steps else None,
        wandb_api_key=wandb_api_key,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Result: {result}")


@app.local_entrypoint()
def run_full_pipeline(
    model: str = "smollm2_135m",
    stage2_max_steps: int = 0,
    stage3_max_steps: int = 0,
):
    """
    Run full training pipeline (all stages) on A10G.

    Args:
        model: Model config name
        stage2_max_steps: Max steps for stage 2 (0 = full run)
        stage3_max_steps: Max steps for stage 3 (0 = full run)

    W&B logging: Set WANDB_API_KEY environment variable locally, or
    create Modal secret: modal secret create wandb-api-key WANDB_API_KEY=<key>
    """
    import os

    # Read WANDB_API_KEY from local environment and pass to remote
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    print(f"\nLaunching FULL PIPELINE on Modal (A10G)...")
    print(f"  Model: {model}")
    print(f"  Stages: 1 -> 1.9 -> 2 -> 3")
    print(f"  Data: mixed_pretrain with influence functions")
    print(f"  W&B: {'enabled (key from local env)' if wandb_api_key else 'using Modal secret'}")
    if stage2_max_steps:
        print(f"  Stage 2 max steps: {stage2_max_steps}")
    if stage3_max_steps:
        print(f"  Stage 3 max steps: {stage3_max_steps}")
    print()

    result = train_full_pipeline.remote(
        model=model,
        stage2_max_steps=stage2_max_steps if stage2_max_steps else None,
        stage3_max_steps=stage3_max_steps if stage3_max_steps else None,
        wandb_api_key=wandb_api_key,
    )

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Result: {result}")
