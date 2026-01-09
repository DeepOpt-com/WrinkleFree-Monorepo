"""Core training launcher - the simple way.

This is the ONLY file you need to understand to launch training.

The deployer is a "delivery truck" - it doesn't care what's in the boxes (config),
it just needs to know which boxes to load (model, stage) and runs them on SkyPilot.

Everything else is passed through as Hydra overrides to the training script.

Example:
    # Python
    from wf_deploy import train
    train("qwen3_4b", stage=2)
    train("qwen3_4b", stage=2, overrides=["training.lr=1e-4"])
    train("qwen3_4b", stage=2, scale="large")  # 4x H100

    # CLI
    wf train -m qwen3_4b -s 2
    wf train -m qwen3_4b -s 2 --scale large
    wf train -m qwen3_4b -s 2 training.lr=1e-4 training.batch_size=8
"""

import base64
import os
import subprocess
from pathlib import Path
from typing import Literal

from wf_deploy.constants import (
    DEFAULT_SMOKE_TEST_MODEL,
    DEFAULT_WANDB_PROJECT,
    DEFAULT_CONTEXT_SIZE,
    DOCKER_IMAGE,
    GCP_PROJECT_ID,
    GAR_REGION,
    GAR_REPO,
    SCALES,
    get_scale_for_model,
)

# Deployer directory for finding credentials
DEPLOYER_DIR = Path(__file__).parent.parent.parent
GCP_SA_PATH = DEPLOYER_DIR / "credentials" / "gcp-service-account.json"

# Valid scales (re-exported for convenience)
Scale = Literal["dev", "small", "medium", "large", "xlarge"]


def _prepare_docker_secrets(cloud: str) -> dict[str, str]:
    """Prepare Docker registry secrets for non-GCP clouds.

    Non-GCP clouds (Nebius, RunPod) need explicit credentials to pull
    from Google Artifact Registry. This function reads the GCP service
    account JSON and formats it appropriately for each cloud.

    Args:
        cloud: Cloud provider name ("gcp", "nebius", "runpod")

    Returns:
        Dict with SKYPILOT_DOCKER_PASSWORD set appropriately

    Raises:
        FileNotFoundError: If GCP service account file is missing
    """
    if cloud == "gcp":
        return {}  # GCP VMs use IAM automatically

    if not GCP_SA_PATH.exists():
        raise FileNotFoundError(
            f"GCP service account not found: {GCP_SA_PATH}\n"
            f"Required to pull Docker images on {cloud}.\n"
            "See: credentials/README.md for setup instructions."
        )

    sa_json = GCP_SA_PATH.read_text()

    if cloud == "runpod":
        # RunPod requires base64-encoded credentials
        password = base64.b64encode(sa_json.encode()).decode()
    else:
        # Nebius and others use raw JSON
        password = sa_json

    return {"SKYPILOT_DOCKER_PASSWORD": password}


def train(
    model: str,
    stage: float = 2.0,  # Deprecated, ignored
    scale: Scale | None = None,
    overrides: list[str] | None = None,
    cloud: str = "nebius",
    detach: bool = True,
    resume_checkpoint: str | None = None,
    gpu_type: str | None = None,
    training_config: str | None = None,
) -> str:
    """Launch a training job on SkyPilot.

    Args:
        model: Model config name (e.g., "qwen3_4b", "smollm2_135m")
        stage: [DEPRECATED] Ignored, kept for backward compatibility
        scale: GPU scale profile ("dev", "small", "medium", "large", "xlarge").
               If None, uses model-specific defaults.
        overrides: Hydra config overrides (e.g., ["training.lr=1e-4"])
        cloud: Cloud provider ("nebius", "gcp", "runpod", or "vast")
        detach: Return immediately (True) or wait for completion (False)
        resume_checkpoint: Path to checkpoint to resume from (local or gs://)
        gpu_type: Override GPU type (e.g., "H100", "L40S", "A10G")
        training_config: Training config name (e.g., "base", "lrc_run", "bitdistill_full")
                        If not provided, extracts from "training=" override

    Returns:
        Run ID (use with `wf logs <run_id>` to see logs)

    Example:
        >>> from wf_deploy import train
        >>> run_id = train("qwen3_4b", training_config="base")
        >>> run_id = train("qwen3_4b", training_config="base", scale="large")
        >>> run_id = train("qwen3_4b", training_config="lrc_run", overrides=["training.batch_size=8"])
    """
    overrides = overrides or []
    return _train_skypilot(model, scale, overrides, cloud, detach, resume_checkpoint, gpu_type, training_config)


def _train_skypilot(
    model: str,
    scale: Scale | None,
    overrides: list[str],
    cloud: str,
    detach: bool,
    resume_checkpoint: str | None,
    gpu_type_override: str | None = None,
    training_config: str | None = None,
) -> str:
    """Launch training on SkyPilot."""
    try:
        import sky
    except ImportError:
        raise ImportError(
            "SkyPilot not installed. Install with: uv add 'skypilot[all]'"
        )

    # Extract training_config from overrides if not explicitly provided
    if training_config is None:
        for override in overrides:
            if override.startswith("training="):
                training_config = override.split("=", 1)[1]
                # Remove from overrides since we pass via TRAINING_CONFIG
                overrides = [o for o in overrides if not o.startswith("training=")]
                break
    if training_config is None:
        raise ValueError(
            "training_config is required. Either pass training_config='base' or "
            "include 'training=base' in overrides."
        )

    # Determine scale (use model default if not specified)
    if scale is None:
        scale = get_scale_for_model(model)

    scale_config = SCALES[scale]
    gpu_count = scale_config["gpus"]
    # Use override GPU type if provided, otherwise use scale default
    gpu_type = gpu_type_override or scale_config["type"]
    accelerators = f"{gpu_type}:{gpu_count}"

    print(f"🚀 Launching {model} (training={training_config}) on SkyPilot")
    print(f"   Cloud: {cloud}")
    if cloud == "vast":
        print("   ⚠️  Vast.ai: Marketplace pricing, variable reliability. Use --cloud nebius for critical runs.")
    print(f"   Scale: {scale} ({accelerators})")
    if overrides:
        print(f"   Overrides: {overrides}")

    # Build environment variables
    envs = {
        "MODEL": model,
        "TRAINING_CONFIG": training_config,
    }

    # Pass through secrets from local environment
    # FAIL LOUDLY if W&B key is not set - training will fail anyway
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        raise RuntimeError(
            "WANDB_API_KEY not set! Training requires W&B logging.\n"
            "Fix: source credentials/.env or export WANDB_API_KEY=your_key"
        )
    # Pass W&B key as env var (not using secrets: block for Python API)
    envs["WANDB_API_KEY"] = wandb_key

    # HF_TOKEN is optional but useful for gated models
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        envs["HF_TOKEN"] = hf_token

    # Resume checkpoint (passed via env var to avoid Hydra parsing issues)
    if resume_checkpoint:
        envs["RESUME_CHECKPOINT"] = resume_checkpoint
        print(f"   Resume from: {resume_checkpoint}")

    # Add distributed config based on GPU count
    # dispatch_train.py auto-detects, but we also pass for visibility
    distributed_config = "fsdp_multi" if gpu_count > 1 else "single_gpu"
    base_overrides = [f"distributed={distributed_config}"]

    # Combine with user overrides
    all_overrides = base_overrides + (overrides or [])
    envs["HYDRA_OVERRIDES"] = " ".join(all_overrides)

    # Create task from YAML
    task = sky.Task.from_yaml("skypilot/train.yaml")
    task.update_envs(envs)

    # Update accelerators based on scale
    # Note: set_resources replaces everything, so we preserve use_spot from YAML
    # Get use_spot from existing YAML resources (defaults to False for on-demand)
    existing_resources = list(task.resources)[0] if task.resources else None
    use_spot = existing_resources.use_spot if existing_resources else False

    # Select cloud provider
    if cloud == "gcp":
        cloud_obj = sky.GCP()
    elif cloud == "runpod":
        cloud_obj = sky.RunPod()
    elif cloud == "vast":
        cloud_obj = sky.Vast()
    else:
        cloud_obj = sky.Nebius()

    task.set_resources(
        sky.Resources(accelerators=accelerators, cloud=cloud_obj, use_spot=use_spot)
    )

    # Job name comes from the YAML (wrinklefree-train)
    job_name = "wrinklefree-train"

    # Launch as managed job
    if detach:
        request_id = sky.jobs.launch(task)
        job_id, _ = sky.get(request_id)
        print(f"✓ Launched! Job ID: {job_id}")
        print(f"  View logs: uv run wf logs {job_name}")
        print(f"  Cancel:    uv run wf cancel {job_name}")
        return job_name
    else:
        print("   Waiting for completion...")
        request_id = sky.jobs.launch(task)
        job_id, _ = sky.get(request_id)
        # Wait for completion by tailing logs
        log_request = sky.jobs.tail_logs(name=job_name)
        sky.stream_and_get(log_request)
        print(f"✓ Completed!")
        return job_name


def logs(run_id: str, follow: bool = False) -> None:
    """View logs for a training run.

    Args:
        run_id: The run ID from train()
        follow: Stream logs continuously
    """
    import sky

    # Extract job_name from run_id format: "sky-{model}-s{stage}:{job_id}"
    if ":" in run_id:
        job_name, _ = run_id.rsplit(":", 1)
        request_id = sky.jobs.tail_logs(name=job_name, follow=follow)
        sky.stream_and_get(request_id)
    else:
        # Assume it's just a job name
        request_id = sky.jobs.tail_logs(name=run_id, follow=follow)
        sky.stream_and_get(request_id)


def cancel(run_id: str) -> bool:
    """Cancel a training run.

    Args:
        run_id: The run ID from train()

    Returns:
        True if cancelled successfully
    """
    import sky

    try:
        # Extract job_name from run_id format: "sky-{model}-s{stage}:{job_id}"
        if ":" in run_id:
            job_name, _ = run_id.rsplit(":", 1)
        else:
            job_name = run_id

        request_id = sky.jobs.cancel(name=job_name)
        sky.get(request_id)
        print(f"✓ Cancelled {run_id}")
        return True
    except Exception as e:
        print(f"✗ Failed to cancel: {e}")
        return False


def list_runs(limit: int = 10) -> list[dict]:
    """List recent training runs.

    Args:
        limit: Maximum number of runs

    Returns:
        List of run info dicts
    """
    import sky

    request_id = sky.jobs.queue(refresh=True)
    jobs = sky.get(request_id)
    return jobs[:limit]


def smoke_test(model: str = DEFAULT_SMOKE_TEST_MODEL) -> dict:
    """Run a quick smoke test to verify the pipeline (legacy).

    Args:
        model: Model to test with (default: smollm2_135m, smallest)

    Returns:
        Test results dict
    """
    print(f"🧪 Running smoke test with {model} on SkyPilot...")

    # Launch a quick job with minimal steps
    run_id = train(
        model=model,
        stage=1.9,
        overrides=["training.max_steps=10"],
        detach=False,
    )
    return {"status": "success", "run_id": run_id}


def smoke_test_unified(
    model: str = DEFAULT_SMOKE_TEST_MODEL,
    objective: str = "dlm",
    gpu_type: str = "L40S",
    gpu_count: int = 1,
    cloud: str = "nebius",
    extra_overrides: list[str] | None = None,
) -> dict:
    """Run unified smoke test with specific objective.

    Uses the new unified smoke_test.yaml with dispatch_smoke.py.

    Args:
        model: Model config name (default: smollm2_135m)
        objective: Smoke test objective (ce, dlm, bitdistill, lrc, etc.)
        gpu_type: GPU type (H100, L40S, A10G)
        gpu_count: Number of GPUs
        cloud: Cloud provider (nebius, runpod, vast)
        extra_overrides: Additional Hydra overrides

    Returns:
        Test results dict with job name
    """
    try:
        import sky
    except ImportError:
        raise ImportError(
            "SkyPilot not installed. Install with: uv add 'skypilot[all]'"
        )

    accelerators = f"{gpu_type}:{gpu_count}"

    print(f"🧪 Launching unified smoke test")
    print(f"   Model: {model}")
    print(f"   Objective: {objective}")
    print(f"   GPU: {accelerators}")
    print(f"   Cloud: {cloud}")

    # Build environment variables
    envs = {
        "MODEL": model,
        "OBJECTIVE": objective,
        "GPU_TYPE": gpu_type,
        "GPU_COUNT": str(gpu_count),
        "CLOUD": cloud,
    }

    # Pass through W&B API key
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        envs["WANDB_API_KEY"] = wandb_key
    else:
        print("   ⚠️  WANDB_API_KEY not set - logging may be disabled")

    # HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        envs["HF_TOKEN"] = hf_token

    # Create task from unified smoke test YAML
    task = sky.Task.from_yaml("skypilot/smoke_test.yaml")
    task.update_envs(envs)

    # Select cloud provider
    if cloud == "runpod":
        cloud_obj = sky.RunPod()
    elif cloud == "vast":
        cloud_obj = sky.Vast()
    else:
        cloud_obj = sky.Nebius()

    # Update resources
    task.set_resources(
        sky.Resources(accelerators=accelerators, cloud=cloud_obj, use_spot=False)
    )

    job_name = f"wf-smoke-{objective}"

    # Launch as managed job (detached)
    request_id = sky.jobs.launch(task)
    job_id, _ = sky.get(request_id)

    print(f"✓ Launched! Job ID: {job_id}")
    print(f"  View logs: sky logs {job_name}")
    print(f"  Cancel:    sky jobs cancel {job_name}")

    return {"status": "launched", "job_name": job_name, "job_id": job_id}


# =============================================================================
# Fairy2 Training (Complex-Valued Quantization)
# =============================================================================


def train_fairy2(
    model: str,
    mode: str = "w2",
    scale: Scale | None = None,
    overrides: list[str] | None = None,
    detach: bool = True,
) -> str:
    """Launch a Fairy2i training job on SkyPilot.

    Trains complex-valued quantized models using the Fairy2i algorithm.
    Weights are quantized to {+1, -1, +i, -i} (fourth roots of unity).

    Args:
        model: Model config name (e.g., "smollm2_135m", "qwen3_4b")
        mode: "w1" for 1-bit (1-stage), "w2" for 2-bit (2-stage)
        scale: GPU scale profile ("dev", "small", etc.)
        overrides: Hydra config overrides
        detach: Return immediately (True) or wait for completion (False)

    Returns:
        Run ID

    Example:
        >>> from wf_deploy import train_fairy2
        >>> run_id = train_fairy2("smollm2_135m", mode="w2")
        >>> run_id = train_fairy2("qwen3_4b", mode="w1", scale="large")
    """
    overrides = overrides or []
    return _train_fairy2_skypilot(model, mode, scale, overrides, detach)


def _train_fairy2_skypilot(
    model: str,
    mode: str,
    scale: Scale | None,
    overrides: list[str],
    detach: bool,
) -> str:
    """Launch Fairy2 training on SkyPilot."""
    try:
        import sky
    except ImportError:
        raise ImportError(
            "SkyPilot not installed. Install with: uv add 'skypilot[all]'"
        )

    # Validate mode
    if mode not in ("w1", "w2"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'w1' or 'w2'")

    # Determine scale (use model default if not specified)
    if scale is None:
        scale = get_scale_for_model(model)

    scale_config = SCALES[scale]
    gpu_count = scale_config["gpus"]
    gpu_type = scale_config["type"]
    accelerators = f"{gpu_type}:{gpu_count}"

    print(f"🚀 Launching Fairy2 training for {model} ({mode}) on SkyPilot")
    print(f"   Scale: {scale} ({accelerators})")
    print(f"   Mode: {mode} ({'1-bit' if mode == 'w1' else '2-bit'})")
    if overrides:
        print(f"   Overrides: {overrides}")

    # Build environment variables
    envs = {
        "MODEL": model,
        "MODE": mode,
    }

    # Pass through secrets from local environment
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        envs["WANDB_API_KEY"] = wandb_key
    else:
        print("   ⚠️  WANDB_API_KEY not set - logging disabled")

    # HF_TOKEN is optional but useful for gated models
    if os.environ.get("HF_TOKEN"):
        envs["HF_TOKEN"] = os.environ["HF_TOKEN"]

    # Hydra overrides
    if overrides:
        envs["HYDRA_OVERRIDES"] = " ".join(overrides)

    # Create task from Fairy2 YAML
    task = sky.Task.from_yaml("skypilot/fairy2_train.yaml")
    task.update_envs(envs)

    # Update accelerators based on scale (preserve use_spot from YAML)
    existing_resources = list(task.resources)[0] if task.resources else None
    use_spot = existing_resources.use_spot if existing_resources else False
    task.set_resources(
        sky.Resources(accelerators=accelerators, cloud=sky.Nebius(), use_spot=use_spot)
    )

    job_name = "wf-fairy2-train"

    # Launch as managed job
    if detach:
        request_id = sky.jobs.launch(task)
        job_id, _ = sky.get(request_id)
        print(f"✓ Launched! Job ID: {job_id}")
        print(f"  View logs: uv run wf logs {job_name}")
        print(f"  Cancel:    uv run wf cancel {job_name}")
        return job_name
    else:
        print("   Waiting for completion...")
        request_id = sky.jobs.launch(task)
        job_id, _ = sky.get(request_id)
        log_request = sky.jobs.tail_logs(name=job_name)
        sky.stream_and_get(log_request)
        print(f"✓ Completed!")
        return job_name


# =============================================================================
# Distillation Training (BitDistill)
# =============================================================================


def train_distill(
    model: str,
    checkpoint: str,
    teacher: str | None = None,
    config: str = "bitdistill",
    scale: Scale | None = None,
    overrides: list[str] | None = None,
    cloud: str = "nebius",
    detach: bool = True,
) -> str:
    """Launch a distillation training job on SkyPilot.

    Distills a BitNet student model against a teacher using BitDistill-style
    distillation (logits + attention relation loss).

    Args:
        model: Model config name (e.g., "qwen3_4b", "smollm2_135m")
        checkpoint: Path to student checkpoint (gs:// or local)
        teacher: Teacher model name (default: same as student's original model)
        config: Distillation config ("bitdistill", "logits_only", "classification")
        scale: GPU scale profile ("dev", "small", etc.)
        overrides: Hydra config overrides
        cloud: Cloud provider ("gcp", "nebius", "runpod", "vast")
        detach: Return immediately (True) or wait for completion (False)

    Returns:
        Run ID

    Example:
        >>> from wf_deploy import train_distill
        >>> run_id = train_distill("qwen3_4b", "gs://bucket/stage2/checkpoint.pt")
        >>> run_id = train_distill("qwen3_4b", "gs://bucket/checkpoint.pt", teacher="meta-llama/Llama-3.2-3B")
    """
    overrides = overrides or []
    return _train_distill_skypilot(model, checkpoint, teacher, config, scale, overrides, cloud, detach)


def _train_distill_skypilot(
    model: str,
    checkpoint: str,
    teacher: str | None,
    config: str,
    scale: Scale | None,
    overrides: list[str],
    cloud: str,
    detach: bool,
) -> str:
    """Launch distillation training on SkyPilot."""
    try:
        import sky
    except ImportError:
        raise ImportError(
            "SkyPilot not installed. Install with: uv add 'skypilot[all]'"
        )

    # Determine scale (use model default if not specified)
    if scale is None:
        scale = get_scale_for_model(model)

    scale_config = SCALES[scale]
    gpu_count = scale_config["gpus"]
    gpu_type = scale_config["type"]
    accelerators = f"{gpu_type}:{gpu_count}"

    print(f"🚀 Launching distillation for {model} on SkyPilot")
    print(f"   Cloud: {cloud}")
    if cloud == "vast":
        print("   ⚠️  Vast.ai: Marketplace pricing, variable reliability. Use --cloud nebius for critical runs.")
    print(f"   Scale: {scale} ({accelerators})")
    print(f"   Checkpoint: {checkpoint}")
    if teacher:
        print(f"   Teacher: {teacher}")
    print(f"   Config: {config}")
    if overrides:
        print(f"   Overrides: {overrides}")

    # Build environment variables
    envs = {
        "MODEL": model,
        "STUDENT_CHECKPOINT": checkpoint,
        "TEACHER_MODEL": teacher or "",
        "DISTILLATION_CONFIG": config,
    }

    # Pass through secrets from local environment
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        envs["WANDB_API_KEY"] = wandb_key
    else:
        print("   ⚠️  WANDB_API_KEY not set - logging disabled")

    # HF_TOKEN is optional but useful for gated models
    if os.environ.get("HF_TOKEN"):
        envs["HF_TOKEN"] = os.environ["HF_TOKEN"]

    # GCS credentials for non-GCP clouds (Vast.ai, RunPod, Nebius)
    # Vast.ai doesn't support file_mounts, so we pass credentials as base64 env var
    if cloud != "gcp" and GCP_SA_PATH.exists():
        sa_json = GCP_SA_PATH.read_text()
        envs["GCS_CREDENTIALS_B64"] = base64.b64encode(sa_json.encode()).decode()
        print(f"   GCS auth: Using service account (base64)")

    # Hydra overrides
    if overrides:
        envs["HYDRA_OVERRIDES"] = " ".join(overrides)

    # Create task from distillation YAML
    task = sky.Task.from_yaml("skypilot/distill_train.yaml")
    task.update_envs(envs)

    # Select cloud provider
    if cloud == "gcp":
        cloud_obj = sky.GCP()
    elif cloud == "runpod":
        cloud_obj = sky.RunPod()
    elif cloud == "vast":
        cloud_obj = sky.Vast()
    else:
        cloud_obj = sky.Nebius()

    # Update accelerators based on scale (preserve use_spot from YAML)
    existing_resources = list(task.resources)[0] if task.resources else None
    use_spot = existing_resources.use_spot if existing_resources else False
    task.set_resources(
        sky.Resources(accelerators=accelerators, cloud=cloud_obj, use_spot=use_spot)
    )

    job_name = "wf-distill-train"

    # Launch as managed job
    if detach:
        request_id = sky.jobs.launch(task)
        job_id, _ = sky.get(request_id)
        print(f"✓ Launched! Job ID: {job_id}")
        print(f"  View logs: uv run wf logs {job_name}")
        print(f"  Cancel:    uv run wf cancel {job_name}")
        return job_name
    else:
        print("   Waiting for completion...")
        request_id = sky.jobs.launch(task)
        job_id, _ = sky.get(request_id)
        log_request = sky.jobs.tail_logs(name=job_name)
        sky.stream_and_get(log_request)
        print(f"✓ Completed!")
        return job_name


# =============================================================================
# TCS Distillation (DLM Students)
# =============================================================================


def train_tcs_distill(
    checkpoint: str,
    teacher: str | None = None,
    scale: Scale | None = None,
    overrides: list[str] | None = None,
    cloud: str = "nebius",
    detach: bool = True,
) -> str:
    """Launch TCS distillation training for DLM students.

    Distills a DLM (Diffusion Language Model) student against an AR teacher
    using Target Concrete Score (TCS) with block-wise attention distillation.

    Args:
        checkpoint: Path to DLM checkpoint (gs:// or local)
        teacher: Teacher model name (default: inferred from dlm_config.json)
        scale: GPU scale profile ("dev", "small", etc.)
        overrides: Hydra config overrides
        cloud: Cloud provider ("nebius", "runpod", "vast")
        detach: Return immediately (True) or wait for completion (False)

    Returns:
        Run ID

    Example:
        >>> from wf_deploy import train_tcs_distill
        >>> run_id = train_tcs_distill("gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/")
        >>> run_id = train_tcs_distill("gs://...", teacher="1bitLLM/bitnet_b1_58-2B")
    """
    overrides = overrides or []
    return _train_tcs_distill_skypilot(checkpoint, teacher, scale, overrides, cloud, detach)


def _train_tcs_distill_skypilot(
    checkpoint: str,
    teacher: str | None,
    scale: Scale | None,
    overrides: list[str],
    cloud: str,
    detach: bool,
) -> str:
    """Launch TCS distillation on SkyPilot."""
    try:
        import sky
    except ImportError:
        raise ImportError(
            "SkyPilot not installed. Install with: uv add 'skypilot[all]'"
        )

    # Default scale for TCS distillation (2B model fits on single H100)
    if scale is None:
        scale = "small"

    scale_config = SCALES[scale]
    gpu_count = scale_config["gpus"]
    gpu_type = scale_config["type"]
    accelerators = f"{gpu_type}:{gpu_count}"

    print(f"🚀 Launching TCS Distillation on SkyPilot")
    print(f"   Cloud: {cloud}")
    if cloud == "vast":
        print("   ⚠️  Vast.ai: Marketplace pricing, variable reliability. Use --cloud nebius for critical runs.")
    print(f"   Scale: {scale} ({accelerators})")
    print(f"   DLM Checkpoint: {checkpoint}")
    if teacher:
        print(f"   Teacher: {teacher}")
    else:
        print(f"   Teacher: (from dlm_config.json)")
    print(f"   Block attention: ENABLED")
    if overrides:
        print(f"   Overrides: {overrides}")

    # Build environment variables
    envs = {
        "DLM_CHECKPOINT": checkpoint,
        "TEACHER_MODEL": teacher or "",
    }

    # Pass through secrets from local environment
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        envs["WANDB_API_KEY"] = wandb_key
    else:
        print("   ⚠️  WANDB_API_KEY not set - logging disabled")

    # HF_TOKEN is optional but useful for gated models
    if os.environ.get("HF_TOKEN"):
        envs["HF_TOKEN"] = os.environ["HF_TOKEN"]

    # GCS credentials for non-GCP clouds
    if cloud != "gcp" and GCP_SA_PATH.exists():
        sa_json = GCP_SA_PATH.read_text()
        envs["GCS_CREDENTIALS_B64"] = base64.b64encode(sa_json.encode()).decode()
        print(f"   GCS auth: Using service account (base64)")

    # Hydra overrides
    if overrides:
        envs["HYDRA_OVERRIDES"] = " ".join(overrides)

    # Create task from TCS distillation YAML
    task = sky.Task.from_yaml("skypilot/tcs_distill_train.yaml")
    task.update_envs(envs)

    # Select cloud provider
    if cloud == "gcp":
        cloud_obj = sky.GCP()
    elif cloud == "runpod":
        cloud_obj = sky.RunPod()
    elif cloud == "vast":
        cloud_obj = sky.Vast()
    else:
        cloud_obj = sky.Nebius()

    # Update accelerators based on scale
    existing_resources = list(task.resources)[0] if task.resources else None
    use_spot = existing_resources.use_spot if existing_resources else False
    task.set_resources(
        sky.Resources(accelerators=accelerators, cloud=cloud_obj, use_spot=use_spot)
    )

    job_name = "wf-tcs-distill"

    # Launch as managed job
    if detach:
        request_id = sky.jobs.launch(task)
        job_id, _ = sky.get(request_id)
        print(f"✓ Launched! Job ID: {job_id}")
        print(f"  View logs: uv run wf logs {job_name}")
        print(f"  Cancel:    uv run wf cancel {job_name}")
        return job_name
    else:
        print("   Waiting for completion...")
        request_id = sky.jobs.launch(task)
        job_id, _ = sky.get(request_id)
        log_request = sky.jobs.tail_logs(name=job_name)
        sky.stream_and_get(log_request)
        print(f"✓ Completed!")
        return job_name


# =============================================================================
# Inference / Serving
# =============================================================================


def serve(
    model_path: str,
    name: str | None = None,
    backend: str = "bitnet",
    context_size: int = DEFAULT_CONTEXT_SIZE,
) -> str:
    """Deploy a model for inference.

    Uses SkyPilot SkyServe for deployment with auto-scaling.

    Args:
        model_path: Path to model (gs://, s3://, hf://, or local)
        name: Service name (auto-generated if not provided)
        backend: Inference backend ("bitnet" or "vllm")
        context_size: Context window size

    Returns:
        Service endpoint URL

    Example:
        >>> from wf_deploy import serve
        >>> endpoint = serve("gs://my-bucket/model.gguf")
        >>> endpoint = serve("hf://Qwen/Qwen3-4B", backend="vllm")
    """
    try:
        import sky
    except ImportError:
        raise ImportError(
            "SkyPilot not installed. Install with: pip install 'skypilot[all]'"
        )

    # Auto-generate name from model path
    if name is None:
        name = model_path.split("/")[-1].replace(".", "-").replace("_", "-")[:20]
        name = f"wf-{name}"

    print(f"🚀 Deploying {model_path}")
    print(f"   Service: {name}")
    print(f"   Backend: {backend}")

    # Build environment
    envs = {
        "MODEL_PATH": model_path,
        "BACKEND": backend,
        "CONTEXT_SIZE": str(context_size),
    }

    # Create task from YAML
    task = sky.Task.from_yaml("skypilot/service.yaml")
    task.update_envs(envs)

    # Launch service
    request_id = sky.serve.up(task, service_name=name)
    service_name, endpoint = sky.get(request_id)

    print(f"✓ Deployed!")
    print(f"  Endpoint: {endpoint}")
    print(f"  Status: wf serve-status {name}")
    return endpoint


def serve_down(name: str) -> None:
    """Stop and remove a deployed service.

    Args:
        name: Service name
    """
    try:
        import sky
    except ImportError:
        raise ImportError("SkyPilot not installed.")

    print(f"🛑 Stopping service {name}...")
    request_id = sky.serve.down(name)
    sky.get(request_id)
    print(f"✓ Service {name} stopped")


def serve_status(name: str) -> dict:
    """Get status of a deployed service.

    Args:
        name: Service name

    Returns:
        Status dict
    """
    try:
        import sky
    except ImportError:
        raise ImportError("SkyPilot not installed.")

    request_id = sky.serve.status(name)
    return sky.get(request_id)


# =============================================================================
# Wandb Monitoring
# =============================================================================


def wandb_status(
    entity: str | None = None,
    project: str = DEFAULT_WANDB_PROJECT,
    limit: int = 10,
) -> None:
    """Check status of recent Wandb runs.

    Args:
        entity: Wandb entity (user or team name). Defaults to current user.
        project: Wandb project name.
        limit: Number of runs to check.
    """
    import os
    import time

    try:
        import wandb
    except ImportError:
        print("❌ wandb not installed. Run: pip install wandb")
        return

    # Set API key if available
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        # Try to load from common locations
        env_file = os.path.expanduser("~/.config/.env.global")
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    if line.startswith("WANDB_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip('"\'')
                        os.environ["WANDB_API_KEY"] = api_key
                        break

    api = wandb.Api()

    # Construct path
    path = f"{entity}/{project}" if entity else project

    print(f"📊 Checking last {limit} runs in {path}...")
    print()

    try:
        runs = list(api.runs(path, order="-created_at", per_page=limit))
    except Exception as e:
        print(f"❌ Error accessing Wandb API: {e}")
        print(f"   Try: wf wandb-status -e <your-entity> -p {project}")
        return

    if not runs:
        print("No runs found.")
        return

    # Current time for comparison
    now = time.time()

    for run in runs:
        # State indicator
        if run.state == "running":
            state_icon = "🟢"
        elif run.state == "finished":
            state_icon = "✅"
        elif run.state in ["failed", "crashed"]:
            state_icon = "❌"
        else:
            state_icon = "⚪"

        print(f"{state_icon} {run.name}")
        print(f"   State: {run.state}")
        print(f"   URL: {run.url}")

        # Metrics
        summary = run.summary or {}
        step = summary.get("_step", summary.get("train/step", "-"))
        loss = summary.get("train/loss", summary.get("loss", "-"))
        lr = summary.get("train/lr", summary.get("lr", "-"))

        if isinstance(loss, float):
            loss = f"{loss:.4f}"
        if isinstance(lr, float):
            lr = f"{lr:.2e}"

        print(f"   Step: {step} | Loss: {loss} | LR: {lr}")

        # Heartbeat check for running runs
        if run.state == "running":
            heartbeat = getattr(run, "heartbeatAt", None)
            if heartbeat:
                print(f"   Last heartbeat: {heartbeat}")

        # Tokens processed if available
        tokens = summary.get("train/tokens_processed")
        if tokens:
            print(f"   Tokens: {tokens:,}")

        print()


# =============================================================================
# Docker Image Building
# =============================================================================

# Image configuration (using GAR - Google Artifact Registry)
IMAGE_NAME = f"{GAR_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/{GAR_REPO}/wf-train"


def build_image(push: bool = True, tag: str | None = None) -> str:
    """Build and optionally push training Docker image to GAR.

    Args:
        push: Whether to push to GCR after building
        tag: Custom tag (default: YYYYMMDD-<git-hash>)

    Returns:
        The full image URL with tag

    Example:
        >>> from wf_deploy import build_image
        >>> image_url = build_image()
        >>> image_url = build_image(push=False, tag="test")
    """
    import datetime

    # Get the deployer directory
    deployer_dir = Path(__file__).parent.parent.parent

    # Generate tag if not provided
    if tag is None:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=deployer_dir,
            )
            git_hash = result.stdout.strip() if result.returncode == 0 else "local"
        except Exception:
            git_hash = "local"
        tag = f"{date_str}-{git_hash}"

    image_url = f"{IMAGE_NAME}:{tag}"
    image_latest = f"{IMAGE_NAME}:latest"

    print(f"🐳 Building WrinkleFree training image...")
    print(f"   Tag: {tag}")
    print(f"   Image: {image_url}")
    print()

    # Build the image
    dockerfile_path = deployer_dir / "docker" / "Dockerfile.train"
    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

    build_cmd = [
        "docker", "build",
        "-f", str(dockerfile_path),
        "-t", image_url,
        "-t", image_latest,
        "--build-arg", f"WF_VERSION={tag}",
        str(deployer_dir),
    ]

    print(f"Running: {' '.join(build_cmd[:4])} ...")
    result = subprocess.run(build_cmd)
    if result.returncode != 0:
        raise RuntimeError("Docker build failed")

    print()
    print(f"✓ Build complete!")
    print(f"   Tagged: {image_url}")
    print(f"   Tagged: {image_latest}")

    if not push:
        print()
        print("Skipping push (--no-push specified)")
        print(f"To push manually:")
        print(f"  docker push {image_url}")
        print(f"  docker push {image_latest}")
        return image_url

    # Push to GAR
    print()
    print("📤 Pushing to Google Artifact Registry...")

    for img in [image_url, image_latest]:
        push_cmd = ["docker", "push", img]
        print(f"   Pushing {img}...")
        result = subprocess.run(push_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to push {img}")

    print()
    print(f"✓ Successfully pushed to GAR!")
    print(f"   {image_url}")
    print(f"   {image_latest}")
    print()
    print("Your SkyPilot YAMLs are already configured to use this image.")

    return image_url
