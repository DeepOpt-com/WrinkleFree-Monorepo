#!/usr/bin/env python3
"""Dispatch training with appropriate config and overrides.

Maps training config to train_lightning.py command.
Called from train.yaml:
    python packages/deployer/scripts/dispatch_train.py \
        --training-config $TRAINING_CONFIG --model $MODEL

Usage:
    python scripts/dispatch_train.py --training-config base --model qwen3_0.6b
    python scripts/dispatch_train.py -t lrc_run -m smollm2_135m --total-tokens 1B
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Available training configs with descriptions
TRAINING_CONFIGS = {
    "base": "Combined CE + DLM (recommended for most training)",
    "bitdistill_full": "Knowledge distillation (BitDistill)",
    "lrc_run": "Low-Rank Correction for quantization error recovery",
    "salient_run": "AWQ-style salient columns",
    "salient_lora_run": "Salient + LoRA combined",
    "salient_lora_hadamard_run": "Salient + LoRA + Hadamard",
    "salient_lora_ce_only": "Salient + LoRA (CE only)",
    "salient_muonclip": "Salient + MuonClip (experimental)",
    "sft_run": "Supervised fine-tuning",
    "pretrain_then_sft": "Pretrain then SFT curriculum",
    "full_run": "Full pipeline (all objectives)",
    "dlm_distill": "DLM distillation",
    "smoke_test": "Quick 30-step validation",
}


def parse_tokens(value: str) -> int:
    """Parse token count with K/M/B suffixes."""
    value = value.upper().replace("_", "")
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    for suffix, mult in multipliers.items():
        if value.endswith(suffix):
            return int(float(value[:-1]) * mult)
    return int(value)


def main():
    parser = argparse.ArgumentParser(
        description="Dispatch training job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k}: {v}" for k, v in TRAINING_CONFIGS.items()
        ),
    )
    parser.add_argument(
        "--training-config",
        "-t",
        required=True,
        choices=list(TRAINING_CONFIGS.keys()),
        help="Training config name",
    )
    parser.add_argument(
        "--model", "-m", default="smollm2_135m", help="Model config name"
    )
    parser.add_argument(
        "--total-tokens",
        help="Total tokens to train (e.g., 1B, 10B, 100M)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max_steps (takes precedence over total-tokens)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/tmp/checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--experiment-name",
        help="Experiment name (defaults to training_config_model)",
    )
    parser.add_argument("--gcs-bucket", default="wrinklefree-checkpoints")
    parser.add_argument("--wandb-project", default="wrinklefree_v2")
    parser.add_argument(
        "--distributed",
        choices=["single_gpu", "fsdp_multi"],
        help="Distributed config (auto-detected if not set)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command only")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides (positional)",
    )
    args = parser.parse_args()

    # Auto-detect distributed config based on GPU count
    if not args.distributed:
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        args.distributed = "fsdp_multi" if gpu_count > 1 else "single_gpu"

    # Generate experiment name if not provided
    if not args.experiment_name:
        args.experiment_name = f"{args.training_config}_{args.model}"

    print("=" * 60)
    print(f"WrinkleFree Training: {args.training_config}")
    print("=" * 60)
    print(f"Description: {TRAINING_CONFIGS[args.training_config]}")
    print(f"Model: {args.model}")
    print(f"Distributed: {args.distributed}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Experiment: {args.experiment_name}")
    print("=" * 60)

    # Build command
    cmd = [
        "uv",
        "run",
        "--package",
        "wf-train",
        "python",
        "-u",  # Force unbuffered stdout/stderr for real-time logs
        "packages/training/scripts/train_lightning.py",
        f"model={args.model}",
        f"training={args.training_config}",
        f"distributed={args.distributed}",
        f"output_dir={args.checkpoint_dir}",
        f"experiment_name={args.experiment_name}",
        f"training.logging.wandb.project={args.wandb_project}",
        "gcs.enabled=true",
        f"gcs.bucket={args.gcs_bucket}",
    ]

    # Add optional overrides
    if args.total_tokens:
        tokens = parse_tokens(args.total_tokens)
        cmd.append(f"training.total_tokens={tokens}")
        cmd.append("training.max_steps=null")  # Use token-based termination

    if args.max_steps:
        cmd.append(f"training.max_steps={args.max_steps}")

    # Add user overrides
    cmd.extend(args.overrides)

    print("\nCommand:")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        print("[DRY RUN] Would execute above command")
        return 0

    # Execute
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    monorepo_root = Path(__file__).parent.parent.parent.parent

    # Pre-install BitNet transformers fork if using bitnet model
    # We must sync first, then install fork, then run without uv run (which would reinstall)
    if "bitnet" in args.model.lower():
        print("\n[BitNet] Installing transformers fork with BitNet support...")
        # First run uv sync to create venv
        sync_cmd = ["uv", "sync", "--package", "wf-train"]
        subprocess.run(sync_cmd, cwd=monorepo_root, check=True)
        # Install transformers fork (overwrites PyPI version)
        install_cmd = [
            "uv", "pip", "install",
            "git+https://github.com/huggingface/transformers.git@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4",
        ]
        subprocess.run(install_cmd, cwd=monorepo_root, check=True)
        print("[BitNet] Transformers fork installed successfully\n")

        # Run directly with venv python to avoid uv run reinstalling packages
        venv_python = monorepo_root / ".venv" / "bin" / "python"
        cmd = [
            str(venv_python),
            "-u",  # Force unbuffered stdout/stderr
            "packages/training/scripts/train_lightning.py",
            f"model={args.model}",
            f"training={args.training_config}",
            f"distributed={args.distributed}",
            f"output_dir={args.checkpoint_dir}",
            f"experiment_name={args.experiment_name}",
            f"training.logging.wandb.project={args.wandb_project}",
            "gcs.enabled=true",
            f"gcs.bucket={args.gcs_bucket}",
        ]
        if args.total_tokens:
            tokens = parse_tokens(args.total_tokens)
            cmd.append(f"training.total_tokens={tokens}")
            cmd.append("training.max_steps=null")
        if args.max_steps:
            cmd.append(f"training.max_steps={args.max_steps}")
        cmd.extend(args.overrides)
        print("Command (direct venv):")
        print(" ".join(cmd))
        print()

    result = subprocess.run(
        cmd,
        cwd=monorepo_root,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},  # Ensure unbuffered output
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
