"""Modal backend for WrinkleFree training jobs.

Provides Modal-based training infrastructure designed for AI tool control.
Modal is the default backend; SkyPilot is available as fallback.

Key features:
- Automatic checkpoint resumption via fingerprint-based run IDs
- Persistent Modal volumes for checkpoints and HF cache
- Simple Python API for AI tools

Example usage:
    from wf_deployer import ModalTrainer, TrainingConfig

    config = TrainingConfig(
        name="qwen3-stage2",
        model="qwen3_4b",
        stage=2,
    )
    trainer = ModalTrainer(config)
    run_id = trainer.launch()
    print(trainer.status(run_id))
    trainer.logs(run_id, follow=True)

For AI tools (simple JSON API):
    trainer.launch_json({
        "model": "qwen3_4b",
        "stage": 2,
        "max_steps": 1000,
    })

CLI usage:
    # Via unified CLI
    wf train --model qwen3_4b --stage 2

    # Or directly via modal
    modal run src/wf_deployer/modal_deployer.py --model qwen3_4b --stage 2

Checkpoint resumption:
    The training code uses fingerprint-based run IDs (SHA256 of config + git commit).
    When you launch a run with the same config, it automatically resumes from
    the last checkpoint. This works because:
    1. Modal volumes persist checkpoints across runs
    2. The training code's RunManager detects existing fingerprints
    3. Training resumes from the latest checkpoint step
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import modal

from wf_deployer.constants import (
    MODAL_APP_NAME,
    MODAL_VOLUME_CHECKPOINTS,
    MODAL_VOLUME_HF_CACHE,
    REPO_1_58_QUANT,
    REPO_CHEAPER_TRAINING,
    DEFAULT_DATA,
    DEFAULT_WANDB_PROJECT,
    DEFAULT_SMOKE_TEST_MODEL,
    TRAINING_TIMEOUT,
    SMOKE_TEST_TIMEOUT,
    DEBUG_TIMEOUT,
    STAGE_CONFIG_MAP,
    RunIdPrefix,
    EnvVars,
)

# ============================================================================
# Scale Profiles: T-shirt sizing for GPU resources
# ============================================================================
# Users specify --scale instead of raw GPU counts
# Each scale maps to GPU type and count for both Modal and SkyPilot

SCALES = {
    "dev": {"gpus": 1, "type": "A10G", "gpu_profile": "a10g_24gb"},
    "small": {"gpus": 1, "type": "H100", "gpu_profile": "h100_80gb"},
    "medium": {"gpus": 2, "type": "H100", "gpu_profile": "h100_80gb"},
    "large": {"gpus": 4, "type": "H100", "gpu_profile": "h100_80gb"},
    "xlarge": {"gpus": 8, "type": "H100", "gpu_profile": "h100_80gb"},
}

# Model-specific default scales (use balanced resources by default)
MODEL_SCALES = {
    "smollm2_135m": "dev",     # Small model, single cheap GPU
    "qwen3_4b": "medium",      # 4B model, 2x H100 recommended
}

DEFAULT_SCALE = "dev"  # Use A10G by default (cheaper)

# ============================================================================
# GPU Configuration (set at deploy time via environment variables)
# ============================================================================
# Usage:
#   modal deploy src/wf_deployer/modal_deployer.py                    # dev (1x A10G)
#   MODAL_SCALE=large modal deploy ...                                # 4x H100
#   MODAL_GPU_TYPE=A100 MODAL_GPU_COUNT=8 modal deploy ...           # custom

MODAL_SCALE = os.environ.get("MODAL_SCALE", DEFAULT_SCALE)
GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", SCALES.get(MODAL_SCALE, SCALES[DEFAULT_SCALE])["type"])
GPU_COUNT = int(os.environ.get("MODAL_GPU_COUNT", SCALES.get(MODAL_SCALE, SCALES[DEFAULT_SCALE])["gpus"]))
GPU_SPEC = f"{GPU_TYPE}:{GPU_COUNT}" if GPU_COUNT > 1 else GPU_TYPE


def get_scale_for_model(model: str) -> str:
    """Get the recommended scale for a model."""
    return MODEL_SCALES.get(model, DEFAULT_SCALE)


def get_scale_config(scale: str) -> dict[str, Any]:
    """Get GPU configuration for a scale profile."""
    if scale not in SCALES:
        raise ValueError(f"Unknown scale '{scale}'. Valid: {', '.join(SCALES.keys())}")
    return SCALES[scale]


def get_deployed_scale() -> str:
    """Get the scale the Modal app was deployed with."""
    return MODAL_SCALE


def check_scale_compatibility(requested_scale: str) -> tuple[bool, str]:
    """Check if requested scale is compatible with deployed Modal app.

    Returns:
        (is_compatible, message)
    """
    if requested_scale not in SCALES:
        return False, f"Unknown scale '{requested_scale}'. Valid: {', '.join(SCALES.keys())}"

    requested = SCALES[requested_scale]
    deployed_gpus = GPU_COUNT
    deployed_type = GPU_TYPE

    if requested["gpus"] > deployed_gpus:
        return False, (
            f"Scale '{requested_scale}' requires {requested['gpus']}x {requested['type']}, "
            f"but Modal app deployed with {deployed_gpus}x {deployed_type}.\n"
            f"Redeploy with: MODAL_SCALE={requested_scale} uv run modal deploy src/wf_deployer/modal_deployer.py"
        )

    if requested["gpus"] == deployed_gpus and requested["type"] == deployed_type:
        return True, f"Using {deployed_gpus}x {deployed_type}"

    # Requested fewer GPUs than deployed - that's fine, we'll use fewer
    return True, f"Using {requested['gpus']}x {requested['type']} (deployed: {deployed_gpus}x {deployed_type})"

# ============================================================================
# Modal App Definition
# ============================================================================

app = modal.App(MODAL_APP_NAME)

# Persistent volumes
checkpoints_volume = modal.Volume.from_name(
    MODAL_VOLUME_CHECKPOINTS, create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name(
    MODAL_VOLUME_HF_CACHE, create_if_missing=True
)

# Training image with all dependencies
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
        # Training
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "bitsandbytes>=0.43.0",
        "muon-optimizer>=0.1.0",
        "torchao>=0.7.0",
        # Logging & storage
        "wandb>=0.16.0",
        "google-cloud-storage>=2.14.0",
        "huggingface_hub>=0.20.0",
        "hf_transfer>=0.1.0",
        # Utilities
        "tqdm>=4.66.0",
        "numpy>=1.26.0",
        "einops>=0.7.0",
        "zstandard>=0.22.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WF_VERSION": "2025-12-25-v1",  # Cache buster
    })
    # Explicitly copy wf_deployer package (src layout requires add_local_dir)
    .add_local_dir(
        Path(__file__).parent,  # src/wf_deployer/
        remote_path="/root/wf_deployer",
    )
)


def _get_secrets() -> list[modal.Secret]:
    """Get available Modal secrets.

    Only includes secrets that actually exist - others are optional.
    """
    # Required secrets for training
    existing_secrets = ["wandb-api-key", "github-token", "gcp-credentials"]
    return [modal.Secret.from_name(name) for name in existing_secrets]


# ============================================================================
# Modal Training Functions
# ============================================================================

# =============================================================================
# GPU Verification
# =============================================================================


def verify_gpu_allocation(requested_type: str) -> dict[str, Any]:
    """Verify allocated GPU matches the requested type.

    Called at training start to ensure we got the correct GPU.
    Raises RuntimeError on mismatch to prevent wasted compute.

    Args:
        requested_type: The GPU type we requested (e.g., "H100", "A10G")

    Returns:
        Dict with GPU info: name, memory_gb, verified

    Raises:
        RuntimeError: If no GPU available or wrong GPU type allocated
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU verification failed: No CUDA GPU available. "
            "Check Modal GPU allocation and CUDA installation."
        )

    actual_name = torch.cuda.get_device_name(0)
    actual_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    device_count = torch.cuda.device_count()

    print(f"[GPU Verify] Requested: {requested_type}")
    print(f"[GPU Verify] Allocated: {actual_name} ({actual_memory_gb:.1f} GB)")
    print(f"[GPU Verify] Device count: {device_count}")

    # Check if requested GPU type is in the actual name
    # e.g., "H100" should be in "NVIDIA H100 80GB HBM3"
    # Special handling: A10G and A10 are treated as equivalent (same 24GB VRAM)
    requested_lower = requested_type.lower()
    actual_lower = actual_name.lower()

    # Normalize A10G -> A10 for comparison (both are 24GB GPUs)
    if requested_lower == "a10g":
        gpu_match = "a10" in actual_lower
    else:
        gpu_match = requested_lower in actual_lower

    if not gpu_match:
        raise RuntimeError(
            f"GPU MISMATCH: Requested {requested_type} but got {actual_name}. "
            "This may indicate Modal allocated a different GPU type. "
            "Check your Modal deployment scale settings. Aborting to prevent "
            "wasted compute on wrong hardware."
        )

    print(f"[GPU Verify] SUCCESS - GPU type verified")
    return {
        "gpu_name": actual_name,
        "gpu_memory_gb": actual_memory_gb,
        "gpu_count": device_count,
        "verified": True,
    }


@app.function(
    image=training_image,
    gpu=GPU_SPEC,
    timeout=TRAINING_TIMEOUT,
    volumes={
        "/checkpoints": checkpoints_volume,
        "/hf_cache": hf_cache_volume,
    },
    secrets=_get_secrets(),
)
def run_training(
    model: str,
    stage: int | float,
    data: str = DEFAULT_DATA,
    max_steps: int | None = None,
    max_tokens: int | None = None,
    wandb_enabled: bool = True,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    hydra_overrides: list[str] | None = None,
    repo_url: str = REPO_1_58_QUANT,
    skip_recovery: bool = False,
    gpu_count: int | None = None,  # Override GPU count (defaults to GPU_COUNT)
    gpu_profile: str | None = None,  # Hydra GPU profile (e.g., "h100_80gb")
) -> dict[str, Any]:
    """Run WrinkleFree training on Modal.

    This is the core training function that runs on Modal infrastructure.
    Designed for programmatic control by AI tools.

    The training code uses fingerprint-based run IDs (SHA256 of config + git commit).
    Runs with the same config automatically resume from the last checkpoint.

    Args:
        model: Model config name (e.g., "qwen3_4b", "smollm2_135m")
        stage: Training stage (1, 1.9, 2, or 3)
        data: Data config name (e.g., "fineweb")
        max_steps: Maximum training steps (None = use config default)
        max_tokens: Maximum tokens (None = use config default)
        wandb_enabled: Enable W&B logging
        wandb_project: W&B project name
        hydra_overrides: Additional Hydra CLI overrides
        repo_url: Git repo URL for training code
        skip_recovery: If True, start fresh instead of resuming

    Returns:
        Dict with run results: run_id, status, checkpoint_path, duration, metrics
    """
    # Map stage to training config (see constants.STAGE_CONFIG_MAP for details)
    # Stage 1: Convert FP16 model to initial 1.58-bit (SubLN initialization)
    # Stage 1.9: Layer-wise distillation to refine initial conversion
    # Stage 2: Pre-training on large corpus (main training)
    # Stage 3: Distillation from teacher model (final refinement)
    training_config = STAGE_CONFIG_MAP.get(stage)
    if not training_config:
        return {"status": "error", "error": f"Unknown stage: {stage}. Valid: {list(STAGE_CONFIG_MAP.keys())}"}

    # Get GitHub token for cloning private repos
    gh_token = os.environ.get(EnvVars.GH_TOKEN)

    # Setup GCP credentials from Modal secret (REQUIRED for checkpoint sync)
    gcp_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if gcp_creds_json:
        gcp_creds_path = Path("/tmp/gcp-credentials.json")
        gcp_creds_path.write_text(gcp_creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(gcp_creds_path)
        print(f"[Modal] GCP credentials configured: {gcp_creds_path}")
    else:
        raise RuntimeError(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "!!! FATAL: GCP credentials not found!\n"
            "!!!\n"
            "!!! GCP credentials are REQUIRED for checkpoint sync.\n"
            "!!! Create the Modal secret with:\n"
            "!!!   modal secret create gcp-credentials GOOGLE_APPLICATION_CREDENTIALS_JSON=\"$(cat path/to/key.json)\"\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

    # Verify GPU allocation before starting (fails fast if wrong GPU)
    gpu_info = verify_gpu_allocation(GPU_TYPE)

    # 1. Install data_handler dependency first
    cheaper_dir = Path("/app/WrinkleFree-CheaperTraining")
    if not cheaper_dir.exists():
        cheaper_url = REPO_CHEAPER_TRAINING
        if gh_token:
            cheaper_clone_url = cheaper_url.replace("https://", f"https://{gh_token}@")
        else:
            cheaper_clone_url = cheaper_url
        print("[Modal] Cloning WrinkleFree-CheaperTraining...")
        subprocess.run(
            ["git", "clone", "--depth=1", cheaper_clone_url, str(cheaper_dir)],
            check=True,
        )
        print("[Modal] Installing data_handler...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(cheaper_dir)],
            check=True,
        )

    # 2. Clone and install main training code (always pull latest)
    work_dir = Path("/app/WrinkleFree-1.58Quant")
    clone_url = repo_url
    if gh_token and "github.com" in repo_url:
        clone_url = repo_url.replace("https://", f"https://{gh_token}@")
        print("[Modal] Using authenticated URL")

    if work_dir.exists():
        # Pull latest changes if already cloned
        print("[Modal] Updating WrinkleFree-1.58Quant...")
        subprocess.run(
            ["git", "-C", str(work_dir), "fetch", "--depth=1", "origin"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(work_dir), "reset", "--hard", "origin/master"],
            check=True,
        )
    else:
        print("[Modal] Cloning WrinkleFree-1.58Quant...")
        subprocess.run(
            ["git", "clone", "--depth=1", clone_url, str(work_dir)],
            check=True,
        )

    print("[Modal] Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(work_dir)],
        check=True,
    )

    os.chdir(work_dir)

    # Set up environment
    env = os.environ.copy()
    env[EnvVars.HF_HOME] = "/hf_cache"
    env[EnvVars.TRANSFORMERS_CACHE] = "/hf_cache"

    # Determine GPU count (parameter overrides deploy-time config)
    num_gpus = gpu_count if gpu_count is not None else GPU_COUNT

    # The output_dir is where checkpoints are stored
    # The training code will generate a fingerprint-based run ID
    # and auto-resume if a matching checkpoint exists
    checkpoint_base = "/checkpoints"

    # Build training command
    # Use torchrun for multi-GPU, python for single-GPU
    if num_gpus > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--standalone",
            "scripts/train.py",
        ]
    else:
        cmd = [sys.executable, "scripts/train.py"]

    # Add Hydra config overrides
    cmd.extend([
        f"model={model}",
        f"training={training_config}",
        f"data={data}",
        f"output_dir={checkpoint_base}",
        # Enable GCS checkpoint sync (required for reproducibility)
        "gcs.enabled=true",
        "gcs.bucket=wrinklefree-checkpoints",
    ])

    # Stage 1 is model conversion only (no logging config), skip wandb for it
    if stage > 1:
        cmd.append(f"training.logging.wandb.enabled={str(wandb_enabled).lower()}")
        if wandb_enabled:
            cmd.append(f"training.logging.wandb.project={wandb_project}")

    # Add GPU profile for batch size configuration (use + to add to defaults list)
    if gpu_profile:
        cmd.append(f"+gpu={gpu_profile}")

    if max_steps is not None:
        cmd.append(f"training.max_steps={max_steps}")

    if max_tokens is not None:
        cmd.append(f"training.max_tokens={max_tokens}")

    if skip_recovery:
        cmd.append("+skip_recovery=true")

    if hydra_overrides:
        cmd.extend(hydra_overrides)

    # Generate a descriptive run ID with prefix for easy backend detection
    # Format: modal-{model}-s{stage} (e.g., modal-qwen3_4b-s2)
    run_desc = f"{RunIdPrefix.MODAL.value}{model}-s{stage}"
    print(f"[Modal] Starting training: {run_desc}")
    print(f"[Modal] GPUs: {num_gpus}x {GPU_TYPE}")
    print(f"[Modal] Checkpoint directory: {checkpoint_base}")
    print(f"[Modal] Command: {' '.join(cmd)}")
    print(f"[Modal] Auto-resume: {'disabled' if skip_recovery else 'enabled (fingerprint-based)'}")

    # Run training (capture output for debugging)
    start_time = datetime.now()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Print captured output
    if result.stdout:
        print("[Modal] Training stdout:")
        print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
    if result.stderr:
        print("[Modal] Training stderr:")
        print(result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)

    # Commit volume changes to persist checkpoints
    checkpoints_volume.commit()

    status = "completed" if result.returncode == 0 else "failed"

    return {
        "run_id": run_desc,  # Prefixed ID for backend detection (modal-{model}-s{stage})
        "status": status,
        "return_code": result.returncode,
        "duration_seconds": duration,
        "checkpoint_path": checkpoint_base,
        "auto_resume": not skip_recovery,
        "config": {
            "model": model,
            "stage": stage,
            "data": data,
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "gpu_count": num_gpus,
            "gpu_type": GPU_TYPE,
        },
        # Actual GPU info from verification (not just requested)
        "gpu_actual": gpu_info["gpu_name"],
        "gpu_memory_gb": gpu_info["gpu_memory_gb"],
        "gpu_verified": gpu_info["verified"],
        "stdout_tail": result.stdout[-2000:] if result.stdout else None,
        "stderr_tail": result.stderr[-2000:] if result.stderr else None,
    }


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=SMOKE_TEST_TIMEOUT,
    volumes={
        "/checkpoints": checkpoints_volume,
        "/hf_cache": hf_cache_volume,
    },
    secrets=_get_secrets(),
)
def smoke_test(model: str = DEFAULT_SMOKE_TEST_MODEL) -> dict[str, Any]:
    """Quick smoke test to verify training pipeline.

    Args:
        model: Model to test (default: smollm2_135m for fast testing)

    Returns:
        Dict with test results
    """
    print(f"[Modal] Running smoke test with {model}...")

    # Stage 1 (conversion only, should be fast)
    stage1 = run_training.local(
        model=model,
        stage=1,
        wandb_enabled=False,
    )
    if stage1["status"] != "completed":
        return {"status": "failed", "failed_at": "stage1", "details": stage1}

    # Stage 1.9 (quick distillation, limited steps)
    # save_interval=5 for testing GCS checkpoint uploads
    stage1_9 = run_training.local(
        model=model,
        stage=1.9,
        max_steps=10,
        wandb_enabled=False,
        hydra_overrides=["training.checkpoint.save_interval=5"],
    )

    return {
        "status": "completed" if stage1_9["status"] == "completed" else "failed",
        "stage1": stage1,
        "stage1_9": stage1_9,
    }


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=DEBUG_TIMEOUT,
    secrets=_get_secrets(),
)
def debug_clone() -> dict[str, Any]:
    """Debug: check environment and attempt git clone."""
    import os
    import subprocess

    result = {
        "env_vars": {},
        "clone_result": None,
    }

    # Check environment
    for k, v in os.environ.items():
        if "TOKEN" in k.upper() or "GH" in k.upper() or "WANDB" in k.upper():
            result["env_vars"][k] = v[:15] + "..." if len(v) > 15 else v

    # Try clone
    gh_token = os.environ.get(EnvVars.GH_TOKEN)
    repo_url = REPO_1_58_QUANT
    if gh_token:
        clone_url = repo_url.replace("https://", f"https://{gh_token}@")
        result["using_token"] = True
    else:
        clone_url = repo_url
        result["using_token"] = False

    try:
        proc = subprocess.run(
            ["git", "clone", "--depth=1", clone_url, "/tmp/test-clone"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        result["clone_result"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout[:500] if proc.stdout else None,
            "stderr": proc.stderr[:500] if proc.stderr else None,
        }
    except Exception as e:
        result["clone_result"] = {"error": str(e)}

    return result


@app.function(
    image=modal.Image.debian_slim().pip_install("pydantic"),
    volumes={"/checkpoints": checkpoints_volume},
)
def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """List training runs with their status.

    Args:
        limit: Maximum number of runs to return

    Returns:
        List of run metadata dicts, sorted by start time (newest first)
    """
    runs = []
    checkpoint_dir = Path("/checkpoints")

    for run_dir in checkpoint_dir.iterdir():
        if not run_dir.is_dir():
            continue

        metadata_file = run_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                runs.append(json.load(f))
        else:
            runs.append({
                "run_id": run_dir.name,
                "status": "unknown",
                "path": str(run_dir),
            })

    # Sort by start time, newest first
    runs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return runs[:limit]


@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/checkpoints": checkpoints_volume},
)
def get_run(run_id: str) -> dict[str, Any]:
    """Get status and details of a specific run.

    Args:
        run_id: The run identifier

    Returns:
        Dict with run status, metrics, and checkpoint info
    """
    run_dir = Path(f"/checkpoints/{run_id}")

    if not run_dir.exists():
        return {"error": f"Run {run_id} not found", "status": "not_found"}

    metadata_file = run_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
    else:
        metadata = {"run_id": run_id, "status": "unknown"}

    # List checkpoint files
    checkpoints = []
    for f in run_dir.glob("**/*.pt"):
        checkpoints.append(str(f.relative_to(run_dir)))
    for f in run_dir.glob("**/*.safetensors"):
        checkpoints.append(str(f.relative_to(run_dir)))

    metadata["checkpoints"] = checkpoints
    return metadata


# ============================================================================
# GCS Checkpoint Import
# ============================================================================

gcs_import_image = modal.Image.debian_slim().pip_install("google-cloud-storage>=2.14.0")


@app.function(
    image=gcs_import_image,
    volumes={"/checkpoints": checkpoints_volume},
    timeout=1800,  # 30 min for large checkpoints
)
def import_from_gcs(
    gcs_path: str,
    dest_run_id: str,
    stage: str = "stage1_9_checkpoint",
) -> dict:
    """Import checkpoint from GCS to Modal volume.

    Args:
        gcs_path: GCS path like gs://bucket/path/to/checkpoint/
        dest_run_id: Destination run ID in Modal volume
        stage: Stage subdirectory (stage1_checkpoint, stage1_9_checkpoint)

    Returns:
        Dict with import status and paths
    """
    from google.cloud import storage
    from pathlib import Path

    # Parse GCS path
    if not gcs_path.startswith("gs://"):
        return {"error": f"Invalid GCS path: {gcs_path}"}

    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1].rstrip("/") + "/" if len(parts) > 1 else ""

    # Setup destination
    dest_dir = Path(f"/checkpoints/{dest_run_id}/{stage}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download from GCS (anonymous access for public buckets)
    try:
        client = storage.Client()
    except Exception:
        # Fallback to anonymous access
        client = storage.Client.create_anonymous_client()

    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    downloaded = []
    for blob in blobs:
        # Get relative path from prefix
        rel_path = blob.name[len(prefix):].lstrip("/")
        if not rel_path:
            continue

        dest_file = dest_dir / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(dest_file))
        downloaded.append(str(rel_path))
        print(f"Downloaded: {rel_path}")

    # Commit volume
    checkpoints_volume.commit()

    return {
        "status": "success",
        "gcs_path": gcs_path,
        "dest_path": str(dest_dir),
        "run_id": dest_run_id,
        "files": downloaded,
    }


# ============================================================================
# Local Client for AI Tools
# ============================================================================

@dataclass
class ModalTrainingConfig:
    """Simple config for Modal training.

    Designed for easy JSON serialization for AI tool control.
    """
    model: str = "qwen3_4b"
    stage: int | float = 2
    data: str = DEFAULT_DATA
    max_steps: int | None = None
    max_tokens: int | None = None
    wandb_enabled: bool = True
    wandb_project: str = DEFAULT_WANDB_PROJECT
    gpu: Literal["H100", "A100-80GB", "A100", "A10G", "L4", "T4"] = "H100"
    gpu_count: int = 1
    timeout_hours: int = TRAINING_TIMEOUT // 3600  # Convert seconds to hours
    hydra_overrides: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "model": self.model,
            "stage": self.stage,
            "data": self.data,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "wandb_enabled": self.wandb_enabled,
            "wandb_project": self.wandb_project,
            "gpu": self.gpu,
            "gpu_count": self.gpu_count,
            "timeout_hours": self.timeout_hours,
            "hydra_overrides": self.hydra_overrides,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModalTrainingConfig":
        """Create from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModalTrainer:
    """High-level trainer interface for Modal backend.

    Designed for easy use by AI tools with simple JSON inputs/outputs.

    Example:
        trainer = ModalTrainer()

        # Launch training
        run_id = trainer.launch("qwen3_4b", stage=2)

        # Check status
        status = trainer.status(run_id)

        # Stream logs
        trainer.logs(run_id, follow=True)

        # List all runs
        runs = trainer.list_runs()
    """

    def __init__(self):
        """Initialize the Modal trainer."""
        self._app = app

    def launch(
        self,
        model: str = "qwen3_4b",
        stage: int | float = 2,
        data: str = DEFAULT_DATA,
        max_steps: int | None = None,
        max_tokens: int | None = None,
        wandb_enabled: bool = True,
        wandb_project: str = DEFAULT_WANDB_PROJECT,
        hydra_overrides: list[str] | None = None,
        gpu_count: int | None = None,
        detach: bool = True,
    ) -> str | dict[str, Any]:
        """Launch a training run.

        Args:
            model: Model config name
            stage: Training stage (1, 1.9, 2, or 3)
            data: Data config name
            max_steps: Maximum training steps
            max_tokens: Maximum tokens
            wandb_enabled: Enable W&B logging
            wandb_project: W&B project name
            hydra_overrides: Additional Hydra overrides
            gpu_count: Number of GPUs (defaults to deploy-time config)
            detach: If True, return run_id immediately. If False, wait for completion.

        Returns:
            If detach=True: run_id string
            If detach=False: full result dict
        """
        kwargs = dict(
            model=model,
            stage=stage,
            data=data,
            max_steps=max_steps,
            max_tokens=max_tokens,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            hydra_overrides=hydra_overrides or [],
            gpu_count=gpu_count,
        )

        if detach:
            # Use spawn for async execution
            handle = run_training.spawn(**kwargs)
            # Return the function call ID as the run identifier
            return handle.object_id
        else:
            # Blocking call
            return run_training.remote(**kwargs)

    def launch_json(self, config: dict[str, Any], detach: bool = True) -> str | dict[str, Any]:
        """Launch training from JSON config.

        Convenience method for AI tools that prefer JSON interfaces.

        Args:
            config: Dict with training config (model, stage, etc.)
            detach: If True, return run_id immediately.

        Returns:
            run_id or full result dict
        """
        cfg = ModalTrainingConfig.from_dict(config)
        return self.launch(
            model=cfg.model,
            stage=cfg.stage,
            data=cfg.data,
            max_steps=cfg.max_steps,
            max_tokens=cfg.max_tokens,
            wandb_enabled=cfg.wandb_enabled,
            wandb_project=cfg.wandb_project,
            hydra_overrides=cfg.hydra_overrides,
            gpu_count=cfg.gpu_count,
            detach=detach,
        )

    def status(self, run_id: str) -> dict[str, Any]:
        """Get status of a training run.

        Args:
            run_id: The run identifier

        Returns:
            Dict with run status and metrics
        """
        return get_run.remote(run_id)

    def logs(self, run_id: str, follow: bool = False) -> None:
        """Stream logs from a training run.

        Args:
            run_id: The run identifier
            follow: If True, continuously stream logs
        """
        # For Modal, we can use the function call ID to get logs
        try:
            fc = modal.FunctionCall.from_id(run_id)
            for log in fc.get_logs():
                print(log, end="")
        except Exception as e:
            print(f"Could not get logs for {run_id}: {e}")

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent training runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of run metadata dicts
        """
        return list_runs.remote(limit=limit)

    def cancel(self, run_id: str) -> bool:
        """Cancel a running training job.

        Args:
            run_id: The run identifier

        Returns:
            True if cancelled successfully
        """
        try:
            fc = modal.FunctionCall.from_id(run_id)
            fc.cancel()
            return True
        except Exception as e:
            print(f"Could not cancel {run_id}: {e}")
            return False

    def smoke_test(self, model: str = DEFAULT_SMOKE_TEST_MODEL) -> dict[str, Any]:
        """Run a smoke test to verify the pipeline.

        Args:
            model: Model to test

        Returns:
            Dict with test results
        """
        return smoke_test.remote(model=model)


# ============================================================================
# CLI Entry Point
# ============================================================================

@app.local_entrypoint()
def main(
    model: str = "qwen3_4b",
    stage: float = 2,
    data: str = DEFAULT_DATA,
    max_steps: int | None = None,
    smoke: bool = False,
    list_jobs: bool = False,
    hydra_overrides: str | None = None,
):
    """CLI entry point for Modal training.

    Examples:
        # Launch Stage 2 training
        modal run wf_deployer/modal_deployer.py --model qwen3_4b --stage 2

        # Run smoke test
        modal run wf_deployer/modal_deployer.py --smoke

        # List recent runs
        modal run wf_deployer/modal_deployer.py --list-jobs

        # Limited training steps
        modal run wf_deployer/modal_deployer.py --model smollm2_135m --stage 1.9 --max-steps 100
    """
    if list_jobs:
        runs = list_runs.remote()
        print(json.dumps(runs, indent=2))
        return

    if smoke:
        print("Running smoke test...")
        result = smoke_test.remote()
        print(json.dumps(result, indent=2))
        return

    print(f"Launching training: model={model}, stage={stage}, data={data}")

    # Parse hydra overrides from space-separated string
    overrides_list = hydra_overrides.split() if hydra_overrides else None
    if overrides_list:
        print(f"Hydra overrides: {overrides_list}")

    result = run_training.remote(
        model=model,
        stage=int(stage) if stage == int(stage) else stage,
        data=data,
        max_steps=max_steps,
        hydra_overrides=overrides_list,
    )
    print(json.dumps(result, indent=2))
