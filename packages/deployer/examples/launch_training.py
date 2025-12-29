#!/usr/bin/env python3
"""Example: Launch a training job using the wf_deployer library.

This example shows how to:
1. Load credentials from a .env file
2. Configure a training job
3. Launch and monitor the job

Usage:
    python examples/launch_training.py
"""

from wf_deployer import TrainingConfig, Trainer, Credentials


def main():
    # Load credentials from .env file (or environment)
    creds = Credentials.from_env_file(".env")

    # Configure the training job
    config = TrainingConfig(
        name="qwen3-stage2-training",
        model="qwen3_4b",
        stage=2,
        # Checkpoint storage
        checkpoint_bucket="wrinklefree-checkpoints",
        checkpoint_store="s3",  # or "gcs", "r2"
        # Resources
        accelerators="H100:4",
        cloud="runpod",
        use_spot=True,
        # W&B
        wandb_project="wrinklefree",
    )

    print(f"Launching training job: {config.name}")
    print(f"Model: {config.model}, Stage: {config.stage}")
    print(f"Accelerators: {config.accelerators}")
    print(f"Checkpoint: {config.checkpoint_store}://{config.checkpoint_bucket}")

    # Create trainer
    trainer = Trainer(config, creds)

    # Launch (async - returns immediately)
    job_id = trainer.launch(detach=True)
    print(f"Job launched: {job_id}")

    # Check status
    print("\nChecking status...")
    status = trainer.status()
    print(f"Status: {status}")

    # To stream logs:
    # trainer.logs(follow=True)

    # To cancel:
    # trainer.cancel()


if __name__ == "__main__":
    main()
