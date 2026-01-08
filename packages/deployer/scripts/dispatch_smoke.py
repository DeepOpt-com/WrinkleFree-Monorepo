#!/usr/bin/env python3
"""Dispatch smoke test with appropriate training config and overrides.

Maps OBJECTIVE env var to training config + Hydra overrides.
Called from smoke_test.yaml:
    python scripts/dispatch_smoke.py --objective $OBJECTIVE --model $MODEL

Usage:
    python scripts/dispatch_smoke.py --objective dlm --model smollm2_135m
    python scripts/dispatch_smoke.py --objective bitdistill --model qwen3_0.6b --steps 50
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Mapping of objectives to training configs and overrides
OBJECTIVE_CONFIGS = {
    "ce": {
        "training_config": "smoke_test",
        "overrides": [
            "training.objectives.continue_pretrain.enabled=true",
            "training.objectives.dlm.enabled=false",
        ],
        "description": "Cross-entropy only",
    },
    "dlm": {
        "training_config": "smoke_test",
        "overrides": [
            "training.objectives.continue_pretrain.enabled=true",
            "training.objectives.dlm.enabled=true",
            "training.objectives.dlm.weight=0.5",
        ],
        "description": "CE + DLM (diffusion language model)",
    },
    "bitdistill": {
        "training_config": "bitdistill_full",
        "overrides": [
            "training.max_steps=30",
            "training.checkpoint.save_interval=10",
            "training.logging.log_interval=1",
        ],
        "description": "BitDistill (logits + attention distillation)",
    },
    "lrc": {
        "training_config": "lrc_run",
        "overrides": [
            "training.max_steps=100",
            "training.checkpoint.save_interval=20",
        ],
        "description": "LRC (Low-Rank Correction)",
    },
    "salient": {
        "training_config": "salient_run",
        "overrides": [
            "training.max_steps=100",
            "training.salient.calibration_samples=16",
            "training.checkpoint.save_interval=20",
        ],
        "description": "AWQ-style salient columns",
    },
    "salient_lora": {
        "training_config": "salient_lora_run",
        "overrides": [
            "training.max_steps=100",
            "training.checkpoint.save_interval=20",
        ],
        "description": "Salient + LoRA combined",
    },
    "hadamard": {
        "training_config": "smoke_test",
        "overrides": [
            "training.auto_convert.use_hadamard=true",
            "training.lambda_warmup.enabled=false",
        ],
        "description": "BitNet v2 Hadamard transform",
    },
    "sft": {
        "training_config": "sft_run",
        "overrides": [
            "training.max_steps=50",
            "training.checkpoint.save_interval=10",
        ],
        "description": "Supervised fine-tuning",
    },
    "meta_opt": {
        "training_config": "smoke_test",
        "overrides": [
            "training.meta_optimization.enabled=true",
            "training.meta_optimization.ldc_mtl.enabled=true",
            "training.meta_optimization.odm.enabled=true",
        ],
        "description": "Meta-optimization (LDC-MTL + ODM)",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Dispatch smoke test")
    parser.add_argument(
        "--objective",
        "-o",
        required=True,
        choices=list(OBJECTIVE_CONFIGS.keys()),
        help="Smoke test objective",
    )
    parser.add_argument(
        "--model", "-m", default="smollm2_135m", help="Model config name"
    )
    parser.add_argument("--steps", type=int, help="Override max_steps")
    parser.add_argument("--dry-run", action="store_true", help="Print command only")
    parser.add_argument(
        "--checkpoint-dir", default="/tmp/checkpoints", help="Checkpoint directory"
    )
    parser.add_argument("--gcs-bucket", default="wrinklefree-checkpoints")
    parser.add_argument("--wandb-project", default="wrinklefree")
    args = parser.parse_args()

    config = OBJECTIVE_CONFIGS[args.objective]
    training_config = config["training_config"]
    overrides = list(config["overrides"])

    # Add step override if specified
    if args.steps:
        overrides.append(f"training.max_steps={args.steps}")

    print("=" * 60)
    print(f"WrinkleFree Smoke Test: {args.objective}")
    print("=" * 60)
    print(f"Description: {config['description']}")
    print(f"Model: {args.model}")
    print(f"Training config: {training_config}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 60)

    # Build command
    cmd = [
        "uv",
        "run",
        "--package",
        "wf-train",
        "python",
        "packages/training/scripts/train_lightning.py",
        f"model={args.model}",
        f"training={training_config}",
        f"output_dir={args.checkpoint_dir}",
        f"experiment_name=smoke_{args.objective}",
        f"training.logging.wandb.project={args.wandb_project}",
        "gcs.enabled=true",
        f"gcs.bucket={args.gcs_bucket}",
    ]
    cmd.extend(overrides)

    print("\nCommand:")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        print("[DRY RUN] Would execute above command")
        return 0

    # Execute
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent.parent)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
