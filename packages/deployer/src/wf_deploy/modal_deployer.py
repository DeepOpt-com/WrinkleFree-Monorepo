"""Modal backend for WrinkleFree training jobs.

Provides Modal-based training infrastructure with:
- Fast cold starts via cached image layers
- Persistent volumes for HF cache and checkpoints
- Local code mounting for fast iteration

Example usage:
    # Via CLI
    wf train -m qwen3_4b -t base --backend modal
    wf smoke -o dlm --backend modal

    # Via Python API
    from wf_deploy.modal_deployer import ModalTrainer

    trainer = ModalTrainer()
    run_id = trainer.launch(
        model="qwen3_4b",
        training_config="base",
    )
    trainer.logs(run_id, follow=True)

Prerequisites:
    1. Install Modal: pip install modal && modal setup
    2. Create secrets (one-time):
        modal secret create wandb-api-key WANDB_API_KEY=$WANDB_API_KEY
        modal secret create gcp-credentials GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat credentials/gcp-service-account.json)"
        modal secret create hf-token HF_TOKEN=$HF_TOKEN  # Optional
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

from wf_deploy.constants import (
    MODAL_APP_NAME,
    MODAL_VOLUME_CHECKPOINTS,
    MODAL_VOLUME_HF_CACHE,
    SCALES,
    TRAINING_CONFIGS,
    SMOKE_OBJECTIVES,
    DEFAULT_WANDB_PROJECT,
    GCP_PROJECT_ID,
    RunIdPrefix,
    get_scale_for_model,
)

# =============================================================================
# Modal App Configuration
# =============================================================================

app = modal.App(MODAL_APP_NAME)

# Persistent volumes for caching
checkpoints_volume = modal.Volume.from_name(
    MODAL_VOLUME_CHECKPOINTS, create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name(
    MODAL_VOLUME_HF_CACHE, create_if_missing=True
)

# =============================================================================
# Image Configuration
# =============================================================================

# Get monorepo root (modal_deployer.py is at packages/deployer/src/wf_deploy/)
MONOREPO_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Base image with all dependencies pre-installed (cached aggressively)
base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "curl", "build-essential")
    # Install uv for fast package management
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc",
    )
    # Install ML dependencies using uv (cached layer)
    # NOTE: Using transformers from HuggingFace fork for BitNet support
    .run_commands(
        "/root/.local/bin/uv pip install --system "
        "torch>=2.5.0 "
        "git+https://github.com/huggingface/transformers.git@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4 "
        "datasets>=2.18.0 "
        "accelerate>=0.27.0 "
        "safetensors>=0.4.0 "
        "pytorch-lightning>=2.6.0 "
        "hydra-core>=1.3.0 "
        "omegaconf>=2.3.0 "
        "bitsandbytes>=0.43.0 "
        "torchao>=0.7.0 "
        "wandb>=0.16.0 "
        "google-cloud-storage>=2.14.0 "
        "gcsfs>=2024.2.0 "
        "huggingface_hub>=0.20.0 "
        "hf_transfer>=0.1.0 "
        "tqdm>=4.66.0 "
        "numpy>=1.26.0 "
        "einops>=0.7.0 "
        "zstandard>=0.22.0 "
        "muon-fsdp2 "
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PATH": "/root/.local/bin:$PATH",
    })
)


def get_training_image() -> modal.Image:
    """Create training image with local monorepo code mounted.

    Uses add_local_dir for fast development iteration:
    - Files are synced at container start
    - Changes to local code are reflected on next run
    - No image rebuild needed for code changes
    """
    return (
        base_image
        .add_local_dir(
            str(MONOREPO_ROOT),
            remote_path="/app/monorepo",
            copy=True,  # Copy into image to allow run_commands after
            # Exclude large/unnecessary directories
            ignore=[
                ".git",
                ".venv",
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                "htmlcov",
                "*.egg-info",
                "node_modules",
                "extern/BitNet",  # Large submodule
                ".mypy_cache",
                "*.so",
                "*.o",
            ],
        )
        .run_commands(
            # Install workspace packages in editable mode
            "cd /app/monorepo && /root/.local/bin/uv pip install --system -e packages/math-utils",
            "cd /app/monorepo && /root/.local/bin/uv pip install --system -e packages/architecture",
            "cd /app/monorepo && /root/.local/bin/uv pip install --system -e packages/data_handler",
            "cd /app/monorepo && /root/.local/bin/uv pip install --system -e packages/training",
            # Install BitNet-compatible transformers fork (overwrites the one from training)
            "/root/.local/bin/uv pip install --system 'git+https://github.com/huggingface/transformers.git@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4'",
        )
    )


def _get_secrets() -> list[modal.Secret]:
    """Get Modal secrets for training.

    Required: wandb-api-key
    Optional: gcp-credentials, hf-token
    """
    secrets = []

    # Required secrets
    try:
        secrets.append(modal.Secret.from_name("wandb-api-key"))
    except modal.exception.NotFoundError:
        raise RuntimeError(
            "Modal secret 'wandb-api-key' not found.\n"
            "Create it with: modal secret create wandb-api-key WANDB_API_KEY=$WANDB_API_KEY"
        )

    # Optional secrets (fail silently)
    for secret_name in ["gcp-credentials", "hf-token"]:
        try:
            secrets.append(modal.Secret.from_name(secret_name))
        except modal.exception.NotFoundError:
            pass  # Optional, continue without

    return secrets


# =============================================================================
# Training Functions
# =============================================================================


@app.function(
    image=get_training_image(),
    gpu="H100",  # Default, can be overridden
    timeout=24 * 60 * 60,  # 24 hours
    volumes={
        "/checkpoints": checkpoints_volume,
        "/hf_cache": hf_cache_volume,
    },
    secrets=_get_secrets(),
)
def run_training(
    model: str,
    training_config: str,
    overrides: list[str] | None = None,
    resume_checkpoint: str | None = None,
    scale: str = "small",
    wandb_project: str = DEFAULT_WANDB_PROJECT,
) -> dict[str, Any]:
    """Run WrinkleFree training on Modal.

    This function runs inside the Modal container. It uses dispatch_train.py
    for consistency with the SkyPilot training path.

    Args:
        model: Model config name (e.g., "qwen3_4b", "smollm2_135m")
        training_config: Training config name (e.g., "base", "lrc_run")
        overrides: Hydra config overrides
        resume_checkpoint: Path to resume from (gs:// or local)
        scale: GPU scale profile
        wandb_project: W&B project name

    Returns:
        Dict with run results: run_id, status, return_code, etc.
    """
    import torch

    # Configure environment
    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache"

    # Setup GCP credentials if available (from Modal secret)
    gcp_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if gcp_creds_json:
        creds_path = Path("/tmp/gcp-credentials.json")
        creds_path.write_text(gcp_creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
        os.environ["GOOGLE_CLOUD_PROJECT"] = GCP_PROJECT_ID
        print(f"[Modal] GCP credentials configured")
    else:
        print("[Modal] WARNING: GCP credentials not found - checkpoint sync disabled")

    # Validate training config
    if training_config not in TRAINING_CONFIGS:
        return {
            "status": "error",
            "error": f"Unknown training_config: {training_config}. Valid: {sorted(TRAINING_CONFIGS)}"
        }

    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"[Modal] GPU: {gpu_count}x {gpu_name}")
    else:
        print("[Modal] WARNING: No GPU available!")

    # Build training command using dispatch_train.py
    cmd = [
        sys.executable,
        "/app/monorepo/packages/deployer/scripts/dispatch_train.py",
        f"--training-config={training_config}",
        f"--model={model}",
        "--checkpoint-dir=/checkpoints",
        "--gcs-bucket=wrinklefree-checkpoints",
        f"--wandb-project={wandb_project}",
    ]

    # Add overrides
    if overrides:
        cmd.extend(overrides)

    if resume_checkpoint:
        cmd.append(f"training.resume.checkpoint_path={resume_checkpoint}")

    print(f"[Modal] Working directory: /app/monorepo")
    print(f"[Modal] Command: {' '.join(cmd)}")

    # Run training
    start_time = datetime.now()
    result = subprocess.run(
        cmd,
        cwd="/app/monorepo",
        capture_output=False,  # Stream output directly
    )
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Commit volume changes to persist checkpoints
    checkpoints_volume.commit()

    run_id = f"{RunIdPrefix.MODAL.value}{model}-{training_config}"
    status = "completed" if result.returncode == 0 else "failed"

    print(f"[Modal] Training {status} in {duration:.1f}s")

    return {
        "run_id": run_id,
        "status": status,
        "return_code": result.returncode,
        "duration_seconds": duration,
        "config": {
            "model": model,
            "training_config": training_config,
            "scale": scale,
            "overrides": overrides,
        },
    }


@app.function(
    image=get_training_image(),
    gpu="L40S",  # Cheaper GPU for smoke tests
    timeout=30 * 60,  # 30 minutes
    volumes={
        "/checkpoints": checkpoints_volume,
        "/hf_cache": hf_cache_volume,
    },
    secrets=_get_secrets(),
)
def run_smoke_test(
    model: str,
    objective: str,
    gpu_type: str = "L40S",
    gpu_count: int = 1,
) -> dict[str, Any]:
    """Run smoke test on Modal.

    Uses dispatch_smoke.py for consistency with SkyPilot path.

    Args:
        model: Model config name
        objective: Smoke test objective (ce, dlm, bitdistill, etc.)
        gpu_type: GPU type
        gpu_count: Number of GPUs

    Returns:
        Dict with test results
    """
    # Configure environment
    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache"

    # Setup GCP credentials
    gcp_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if gcp_creds_json:
        creds_path = Path("/tmp/gcp-credentials.json")
        creds_path.write_text(gcp_creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
        os.environ["GOOGLE_CLOUD_PROJECT"] = GCP_PROJECT_ID

    # Validate objective
    if objective not in SMOKE_OBJECTIVES:
        return {
            "status": "error",
            "error": f"Unknown objective: {objective}. Valid: {sorted(SMOKE_OBJECTIVES)}"
        }

    # Build smoke test command
    cmd = [
        sys.executable,
        "/app/monorepo/packages/deployer/scripts/dispatch_smoke.py",
        f"--model={model}",
        f"--objective={objective}",
        "--checkpoint-dir=/checkpoints",
        "--gcs-bucket=wrinklefree-checkpoints",
    ]

    print(f"[Modal] Smoke test: {model} / {objective}")
    print(f"[Modal] Command: {' '.join(cmd)}")

    start_time = datetime.now()
    result = subprocess.run(
        cmd,
        cwd="/app/monorepo",
        capture_output=False,
    )
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    checkpoints_volume.commit()

    run_id = f"{RunIdPrefix.MODAL.value}smoke-{objective}"
    status = "completed" if result.returncode == 0 else "failed"

    return {
        "run_id": run_id,
        "status": status,
        "return_code": result.returncode,
        "duration_seconds": duration,
        "config": {
            "model": model,
            "objective": objective,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
        },
    }


@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/checkpoints": checkpoints_volume},
)
def list_checkpoints(limit: int = 20) -> list[dict[str, Any]]:
    """List checkpoints in the Modal volume.

    Args:
        limit: Maximum number of checkpoints to return

    Returns:
        List of checkpoint info dicts
    """
    checkpoints = []
    checkpoint_dir = Path("/checkpoints")

    for path in checkpoint_dir.rglob("*.ckpt"):
        stat = path.stat()
        checkpoints.append({
            "path": str(path.relative_to(checkpoint_dir)),
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    # Sort by modified time, newest first
    checkpoints.sort(key=lambda x: x["modified"], reverse=True)
    return checkpoints[:limit]


# =============================================================================
# Client Interface
# =============================================================================


class ModalTrainer:
    """High-level trainer interface for Modal backend.

    Provides a consistent interface for launching and managing training jobs.

    Example:
        trainer = ModalTrainer()

        # Launch training
        run_id = trainer.launch(
            model="qwen3_4b",
            training_config="base",
        )

        # Check status
        status = trainer.status(run_id)

        # Stream logs
        trainer.logs(run_id, follow=True)
    """

    def __init__(self):
        """Initialize the Modal trainer."""
        self._app = app

    def launch(
        self,
        model: str,
        training_config: str,
        scale: str | None = None,
        overrides: list[str] | None = None,
        detach: bool = True,
        resume_checkpoint: str | None = None,
        wandb_project: str = DEFAULT_WANDB_PROJECT,
    ) -> str | dict[str, Any]:
        """Launch a training run on Modal.

        Args:
            model: Model config name (e.g., "qwen3_4b", "smollm2_135m")
            training_config: Training config (e.g., "base", "lrc_run")
            scale: GPU scale profile (dev, small, medium, large, xlarge)
            overrides: Hydra config overrides
            detach: Return immediately (True) or wait for completion (False)
            resume_checkpoint: Path to resume from (gs:// or local)
            wandb_project: W&B project name

        Returns:
            If detach=True: run_id string
            If detach=False: full result dict
        """
        # Determine scale and GPU config
        scale = scale or get_scale_for_model(model)
        scale_config = SCALES[scale]
        gpu_count = scale_config["gpus"]
        gpu_type = scale_config["type"]

        print(f"[Modal] Launching {model} (training={training_config})")
        print(f"   Scale: {scale} ({gpu_count}x {gpu_type})")
        if overrides:
            print(f"   Overrides: {overrides}")
        if resume_checkpoint:
            print(f"   Resume: {resume_checkpoint}")

        kwargs = {
            "model": model,
            "training_config": training_config,
            "overrides": overrides or [],
            "resume_checkpoint": resume_checkpoint,
            "scale": scale,
            "wandb_project": wandb_project,
        }

        if detach:
            # Async execution - return immediately
            with app.run():
                handle = run_training.spawn(**kwargs)
                run_id = f"{RunIdPrefix.MODAL.value}{model}-{training_config}"
                print(f"[Modal] Launched! Run ID: {run_id}")
                print(f"   Function call ID: {handle.object_id}")
                print(f"   Dashboard: https://modal.com/apps/{MODAL_APP_NAME}")
                # Store function call ID for later retrieval
                self._last_function_call_id = handle.object_id
            return run_id
        else:
            # Blocking execution - wait for result
            with app.run():
                result = run_training.remote(**kwargs)
            return result

    def smoke_test(
        self,
        model: str,
        objective: str = "dlm",
        gpu_type: str = "L40S",
        gpu_count: int = 1,
        detach: bool = True,
    ) -> str | dict[str, Any]:
        """Run a smoke test on Modal.

        Args:
            model: Model config name
            objective: Smoke test objective
            gpu_type: GPU type
            gpu_count: Number of GPUs
            detach: Return immediately or wait

        Returns:
            run_id string or result dict
        """
        print(f"[Modal] Launching smoke test: {model} / {objective}")
        print(f"   GPU: {gpu_count}x {gpu_type}")

        kwargs = {
            "model": model,
            "objective": objective,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
        }

        if detach:
            with app.run():
                handle = run_smoke_test.spawn(**kwargs)
                run_id = f"{RunIdPrefix.MODAL.value}smoke-{objective}"
                print(f"[Modal] Launched! Run ID: {run_id}")
                self._last_function_call_id = handle.object_id
            return run_id
        else:
            with app.run():
                result = run_smoke_test.remote(**kwargs)
            return result

    def logs(self, run_id: str, follow: bool = False) -> None:
        """Stream logs from a training run.

        Args:
            run_id: The run identifier
            follow: If True, continuously stream logs
        """
        # Try to get logs from the function call
        if hasattr(self, '_last_function_call_id'):
            try:
                fc = modal.FunctionCall.from_id(self._last_function_call_id)
                for log in fc.get_logs():
                    print(log, end="")
                return
            except Exception as e:
                print(f"Could not get logs: {e}")

        print(f"To view logs, visit: https://modal.com/apps/{MODAL_APP_NAME}")
        print(f"Or use: modal app logs {MODAL_APP_NAME}")

    def cancel(self, run_id: str) -> bool:
        """Cancel a running training job.

        Args:
            run_id: The run identifier

        Returns:
            True if cancelled successfully
        """
        if hasattr(self, '_last_function_call_id'):
            try:
                fc = modal.FunctionCall.from_id(self._last_function_call_id)
                fc.cancel()
                print(f"[Modal] Cancelled: {run_id}")
                return True
            except Exception as e:
                print(f"Could not cancel: {e}")
                return False

        print(f"No active function call found for {run_id}")
        return False

    def status(self, run_id: str) -> dict[str, Any]:
        """Get status of a training run.

        Args:
            run_id: The run identifier

        Returns:
            Dict with run status
        """
        if hasattr(self, '_last_function_call_id'):
            try:
                fc = modal.FunctionCall.from_id(self._last_function_call_id)
                return {
                    "run_id": run_id,
                    "status": str(fc.status),
                    "function_call_id": self._last_function_call_id,
                }
            except Exception as e:
                return {"run_id": run_id, "status": "unknown", "error": str(e)}

        return {"run_id": run_id, "status": "unknown"}

    def list_checkpoints(self, limit: int = 20) -> list[dict[str, Any]]:
        """List checkpoints in the Modal volume.

        Args:
            limit: Maximum number of checkpoints

        Returns:
            List of checkpoint info dicts
        """
        with app.run():
            return list_checkpoints.remote(limit=limit)


# =============================================================================
# CLI Entry Point (for direct modal run)
# =============================================================================


@app.local_entrypoint()
def main(
    model: str = "smollm2_135m",
    training_config: str = "base",
    smoke: bool = False,
    objective: str = "dlm",
    list_ckpts: bool = False,
):
    """CLI entry point for Modal training.

    Examples:
        # Launch training
        modal run modal_deployer.py --model qwen3_4b --training-config base

        # Run smoke test
        modal run modal_deployer.py --smoke --objective dlm

        # List checkpoints
        modal run modal_deployer.py --list-ckpts
    """
    import json

    if list_ckpts:
        ckpts = list_checkpoints.remote()
        print(json.dumps(ckpts, indent=2))
        return

    if smoke:
        print(f"Running smoke test: {model} / {objective}")
        result = run_smoke_test.remote(model=model, objective=objective)
        print(json.dumps(result, indent=2))
        return

    print(f"Launching training: {model} / {training_config}")
    result = run_training.remote(
        model=model,
        training_config=training_config,
    )
    print(json.dumps(result, indent=2))
