"""WrinkleFree CLI - Simple training launcher.

The CLI is the main interface for launching training jobs on SkyPilot.
Any argument not recognized is passed directly to the training script
as a Hydra override.

Examples:
    # Basic training
    wf train -m qwen3_4b -s 2

    # With overrides (passed to Hydra)
    wf train -m qwen3_4b -s 2 training.lr=1e-4 training.batch_size=8

    # Quick smoke test
    wf smoke

    # View logs
    wf logs <run_id>
"""

from pathlib import Path

import click
from dotenv import load_dotenv

# Auto-load .env files at CLI startup
load_dotenv()  # .env in current directory
load_dotenv(Path.home() / ".config" / ".env.global")  # Global fallback

import sys
import warnings

from wf_deploy import core
from wf_deploy.constants import (
    DEFAULT_SMOKE_TEST_MODEL,
    DEFAULT_WANDB_PROJECT,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_BACKEND,
    TRAINING_CONFIGS,
    SMOKE_OBJECTIVES,
    SCALES,
    Backend,
    get_wandb_entity,
)


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """WrinkleFree - Train 1.58-bit LLMs.

    Simple launcher for training jobs on SkyPilot.
    """
    pass


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--model", "-m", required=True, help="Model config (e.g., qwen3_4b)")
@click.option("--training", "-t", default=None, help="Training config (e.g., base, bitdistill_full, lrc_run)")
@click.option("--stage", "-s", default=None, type=float, help="[DEPRECATED] Training stage (1, 1.9, 2, 3). Use --training instead.")
@click.option("--scale", default=None,
              type=click.Choice(["dev", "small", "medium", "large", "xlarge"]),
              help="GPU scale: dev (1xH100), small (1xH100), medium (2xH100), large (4xH100), xlarge (8xH100)")
@click.option("--gpu-type", default=None,
              type=click.Choice(["H100", "A100", "A100-80GB", "L40S", "A10G"]),
              help="Override GPU type (default: H100)")
@click.option("--resume", "-r", default=None, help="Resume from checkpoint (local path or gs://)")
@click.option("--cloud", "-c", default="nebius", type=click.Choice(["gcp", "nebius", "runpod", "vast"]), help="Cloud provider (SkyPilot only)")
@click.option("--backend", "-b", default=DEFAULT_BACKEND,
              type=click.Choice([b.value for b in Backend]),
              help="Deployment backend: skypilot (default) or modal")
@click.option("--detach/--no-detach", default=True, help="Return immediately or wait")
@click.option("--dry-run", is_flag=True, help="Print config without launching")
@click.pass_context
def train(ctx, model: str, training: str | None, stage: float | None, scale: str, gpu_type: str | None, resume: str, cloud: str, backend: str, detach: bool, dry_run: bool):
    """Launch a training job.

    Any extra arguments are passed directly to Hydra.

    \b
    Training Configs (recommended):
        base:           CE training (default)
        bitdistill_full: Knowledge distillation
        lrc_run:        Low-Rank Correction
        salient_run:    AWQ-style salient columns
        sft_run:        Supervised fine-tuning

    \b
    Scales (GPU profiles):
        dev:    1x H100  (cheap, for testing)
        small:  1x H100  (default)
        medium: 2x H100
        large:  4x H100
        xlarge: 8x H100

    \b
    Examples:
        wf train -m qwen3_4b -t base
        wf train -m qwen3_4b -t bitdistill_full --scale large
        wf train -m qwen3_4b -t base training.lr=1e-4
        wf train -m smollm2_135m -t lrc_run --no-detach
        wf train -m qwen3_4b -t base --resume gs://bucket/checkpoint.pt
        wf train -m qwen3_4b -t base --dry-run  # Preview without launching
    """
    # Handle deprecation of --stage
    if stage is not None and training is None:
        warnings.warn(
            "--stage/-s is deprecated. Use --training/-t instead.\n"
            "  Example: wf train -m qwen3_4b -t base",
            DeprecationWarning,
            stacklevel=2,
        )
        click.echo("⚠️  --stage is deprecated. Mapping to training config...", err=True)
        # Map stage to training config
        stage_to_training = {
            1: "stage1_subln",
            1.9: "stage1_9_layerwise",
            2: "stage2_pretrain",
            3: "stage3_distill",
        }
        training = stage_to_training.get(stage, "base")
        click.echo(f"   Using training={training}", err=True)
    elif stage is not None and training is not None:
        click.echo("⚠️  Both --stage and --training provided. Using --training.", err=True)
    elif training is None:
        click.echo("Error: --training/-t is required.", err=True)
        click.echo("  Example: wf train -m qwen3_4b -t base", err=True)
        sys.exit(1)

    # Validate training config
    if training not in TRAINING_CONFIGS:
        click.echo(f"Error: Unknown training config '{training}'", err=True)
        click.echo(f"Available configs: {', '.join(sorted(TRAINING_CONFIGS))}", err=True)
        sys.exit(1)

    # Extra args are Hydra overrides
    overrides = list(ctx.args)

    if dry_run:
        click.echo("🔍 DRY RUN - would launch with:")
        click.echo(f"   Model: {model}")
        click.echo(f"   Training: {training}")
        click.echo(f"   Backend: {backend}")
        click.echo(f"   Scale: {scale or 'auto'}")
        if gpu_type:
            click.echo(f"   GPU Type: {gpu_type}")
        if backend == Backend.SKYPILOT.value:
            click.echo(f"   Cloud: {cloud}")
        if overrides:
            click.echo(f"   Overrides: {overrides}")
        if resume:
            click.echo(f"   Resume: {resume}")
        return

    core.train(
        model=model,
        training_config=training,
        scale=scale,
        overrides=overrides,
        cloud=cloud,
        backend=backend,
        detach=detach,
        resume_checkpoint=resume,
        gpu_type=gpu_type,
    )


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--model", "-m", required=True, help="Model config (e.g., smollm2_135m)")
@click.option("--mode", default="w2", type=click.Choice(["w1", "w2"]),
              help="Quantization mode: w1 (1-bit) or w2 (2-bit)")
@click.option("--scale", default=None,
              type=click.Choice(["dev", "small", "medium", "large", "xlarge"]),
              help="GPU scale profile")
@click.option("--detach/--no-detach", default=True, help="Return immediately or wait")
@click.pass_context
def fairy2(ctx, model: str, mode: str, scale: str, detach: bool):
    """Launch Fairy2i complex-valued quantization training.

    Trains models with weights quantized to {+1, -1, +i, -i} using
    the Fairy2i algorithm (arxiv:2512.02901).

    \b
    Modes:
        w1: 1-bit (single-stage, most aggressive)
        w2: 2-bit (two-stage residual, better quality)

    \b
    Examples:
        wf fairy2 -m smollm2_135m --mode w2
        wf fairy2 -m qwen3_4b --mode w1 --scale large
        wf fairy2 -m smollm2_135m training.max_steps=1000
    """
    overrides = ctx.args

    core.train_fairy2(
        model=model,
        mode=mode,
        scale=scale,
        overrides=overrides,
        detach=detach,
    )


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--model", "-m", required=True, help="Model config (e.g., qwen3_4b)")
@click.option("--checkpoint", "-ckpt", required=True, help="Student checkpoint path (gs:// or local)")
@click.option("--teacher", "-t", default=None, help="Teacher model (default: same as student original)")
@click.option("--config", "-cfg", default="bitdistill",
              type=click.Choice(["bitdistill", "logits_only", "classification"]),
              help="Distillation config")
@click.option("--scale", "-s", default=None,
              type=click.Choice(["dev", "small", "medium", "large", "xlarge"]),
              help="GPU scale profile")
@click.option("--cloud", "-c", default="nebius", type=click.Choice(["gcp", "nebius", "runpod", "vast"]), help="Cloud provider")
@click.option("--detach/--no-detach", default=True, help="Return immediately or wait")
@click.pass_context
def distill(ctx, model: str, checkpoint: str, teacher: str | None, config: str, scale: str, cloud: str, detach: bool):
    """Launch distillation training on cloud GPU.

    Distills a BitNet student model against a teacher using BitDistill-style
    distillation (logits + attention relation loss).

    \b
    Configs:
        bitdistill:     Logits + attention distillation (default)
        logits_only:    Logits KL divergence only (no attention)
        classification: For classification tasks

    \b
    Examples:
        wf distill -m qwen3_4b -ckpt gs://bucket/stage2/checkpoint.pt
        wf distill -m qwen3_4b -ckpt gs://bucket/checkpoint.pt --cloud vast
        wf distill -m qwen3_4b -ckpt gs://bucket/checkpoint.pt -t meta-llama/Llama-3.2-3B
        wf distill -m qwen3_4b -ckpt gs://bucket/checkpoint.pt --config logits_only
        wf distill -m smollm2_135m -ckpt gs://bucket/checkpoint.pt training.max_steps=1000
    """
    overrides = list(ctx.args)

    core.train_distill(
        model=model,
        checkpoint=checkpoint,
        teacher=teacher,
        config=config,
        scale=scale,
        overrides=overrides,
        cloud=cloud,
        detach=detach,
    )


@cli.command("tcs-distill",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--checkpoint", "-ckpt", required=True, help="DLM checkpoint path (gs:// or local)")
@click.option("--teacher", "-t", default=None, help="Teacher model (default: from dlm_config.json)")
@click.option("--scale", "-s", default=None,
              type=click.Choice(["dev", "small", "medium", "large", "xlarge"]),
              help="GPU scale profile")
@click.option("--cloud", "-c", default="nebius", type=click.Choice(["nebius", "runpod", "vast"]), help="Cloud provider")
@click.option("--detach/--no-detach", default=True, help="Return immediately or wait")
@click.pass_context
def tcs_distill(ctx, checkpoint: str, teacher: str | None, scale: str, cloud: str, detach: bool):
    """Launch TCS distillation for DLM students (block-wise attention enabled).

    Distills a DLM (Diffusion Language Model) student against an AR teacher
    using Target Concrete Score (TCS) with block-wise attention distillation.

    \b
    Key features:
        - NO logit shifting (DLM predicts masked tokens, not next tokens)
        - Top-K TCS estimation for sparse distribution matching
        - Block-wise attention distillation (matches within bd_size blocks)
        - GCS checkpoint uploads enabled by default

    \b
    Examples:
        wf tcs-distill --checkpoint gs://wrinklefree-checkpoints/dlm/bitnet-b1.58-2B-4T-bf16/
        wf tcs-distill --checkpoint gs://... --teacher 1bitLLM/bitnet_b1_58-2B
        wf tcs-distill --checkpoint gs://... --cloud runpod --scale medium
        wf tcs-distill --checkpoint gs://... training.max_steps=1000
    """
    overrides = list(ctx.args)

    core.train_tcs_distill(
        checkpoint=checkpoint,
        teacher=teacher,
        scale=scale,
        overrides=overrides,
        cloud=cloud,
        detach=detach,
    )


@cli.command()
@click.argument("job_name")
@click.option("--follow", "-f", is_flag=True, help="Stream logs continuously")
def logs(job_name: str, follow: bool):
    """View logs for a training run.

    \b
    Examples:
        wf logs wrinklefree-train
        wf logs wrinklefree-train -f
    """
    core.logs(job_name, follow=follow)


@cli.command()
@click.argument("job_name")
def cancel(job_name: str):
    """Cancel a training run.

    \b
    Example:
        wf cancel wrinklefree-train
    """
    core.cancel(job_name)


@cli.command()
@click.option("--limit", "-n", default=10, type=int, help="Number of runs to show")
def runs(limit: int):
    """List recent training runs.

    \b
    Examples:
        wf runs
        wf runs -n 20
    """
    results = core.list_runs(limit=limit)

    if not results:
        click.echo("No runs found.")
        return

    click.echo("Recent jobs:")
    click.echo(f"{'ID':<4} {'Name':<20} {'Status':<12} {'Resources':<15}")
    click.echo("-" * 55)
    for run in results:
        # Handle both dict and object responses from SkyPilot
        if isinstance(run, dict):
            job_id = run.get("job_id", "?")
            name = run.get("job_name", run.get("name", "?"))
            status = str(run.get("status", "unknown"))
            resources = run.get("resources", "-")
        else:
            job_id = getattr(run, "job_id", "?")
            name = getattr(run, "job_name", "?")
            status = str(getattr(run, "status", "unknown"))
            resources = getattr(run, "resources", "-")

        # Clean up status display
        if "." in status:
            status = status.split(".")[-1].strip("'>")

        click.echo(f"{job_id:<4} {name:<20} {status:<12} {resources:<15}")


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--model", "-m", default=DEFAULT_SMOKE_TEST_MODEL, help="Model to test (default: smollm2_135m)")
@click.option("--objective", "-o", default="ce",
              type=click.Choice(list(SMOKE_OBJECTIVES)),
              help="Smoke test objective (default: ce)")
@click.option("--gpu-type", default="L40S",
              type=click.Choice(["H100", "A100", "L40S", "A10G"]),
              help="GPU type (default: L40S)")
@click.option("--gpu-count", default=1, type=int, help="Number of GPUs (default: 1)")
@click.option("--cloud", "-c", default="nebius",
              type=click.Choice(["nebius", "runpod", "vast"]),
              help="Cloud provider (SkyPilot only, default: nebius)")
@click.option("--backend", "-b", default=DEFAULT_BACKEND,
              type=click.Choice([b.value for b in Backend]),
              help="Deployment backend: skypilot (default) or modal")
@click.option("--dry-run", is_flag=True, help="Print config without launching")
@click.pass_context
def smoke(ctx, model: str, objective: str, gpu_type: str, gpu_count: int, cloud: str, backend: str, dry_run: bool):
    """Run a quick smoke test on cloud GPU.

    Uses the unified smoke_test.yaml with dispatch_smoke.py to run
    various training objectives for validation.

    \b
    Objectives:
        ce:           Cross-entropy only (default)
        bitdistill:   BitDistill distillation
        lrc:          Low-Rank Correction
        salient:      AWQ-style salient columns
        salient_lora: Salient + LoRA
        hadamard:     BitNet v2 Hadamard
        sft:          Supervised fine-tuning
        meta_opt:     Meta-optimization (LDC-MTL + ODM)

    \b
    Examples:
        wf smoke                          # Default: ce on L40S
        wf smoke -o bitdistill            # BitDistill smoke test
        wf smoke -o lrc --gpu-type H100   # LRC on H100
        wf smoke -o meta_opt --gpu-count 2  # Meta-opt with 2 GPUs
        wf smoke --dry-run                # Preview without launching
    """
    # Extra args passed to dispatch_smoke.py
    extra_overrides = list(ctx.args)

    if dry_run:
        click.echo("🔍 DRY RUN - would launch smoke test with:")
        click.echo(f"   Model: {model}")
        click.echo(f"   Objective: {objective}")
        click.echo(f"   Backend: {backend}")
        click.echo(f"   GPU: {gpu_count}x {gpu_type}")
        if backend == Backend.SKYPILOT.value:
            click.echo(f"   Cloud: {cloud}")
        if extra_overrides:
            click.echo(f"   Extra overrides: {extra_overrides}")
        return

    result = core.smoke_test_unified(
        model=model,
        objective=objective,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        cloud=cloud,
        backend=backend,
        extra_overrides=extra_overrides,
    )
    click.echo(f"Result: {result}")


@cli.command()
@click.option("--no-push", is_flag=True, help="Build only, don't push to GCR")
@click.option("--tag", "-t", default=None, help="Custom image tag (default: YYYYMMDD-<git-hash>)")
def build(no_push: bool, tag: str | None):
    """Build and push training Docker image to GCR.

    Creates a Docker image with all training dependencies pre-installed.
    This dramatically reduces job startup time from ~10min to ~30s.

    \b
    Examples:
        wf build              # Build and push with auto-generated tag
        wf build --no-push    # Build locally only
        wf build -t v1.0.0    # Build with custom tag

    \b
    The image is pushed to:
        gcr.io/wrinklefree-481904/wf-train:latest
        gcr.io/wrinklefree-481904/wf-train:<tag>
    """
    core.build_image(push=not no_push, tag=tag)


@cli.command()
@click.argument("model_path")
@click.option("--name", "-n", default=None, help="Service name (auto-generated if not set)")
@click.option("--backend", "-b", default="bitnet", type=click.Choice(["bitnet", "vllm"]))
@click.option("--context", "-c", default=DEFAULT_CONTEXT_SIZE, type=int, help="Context window size")
def serve(model_path: str, name: str, backend: str, context: int):
    """Deploy a model for inference.

    \b
    Examples:
        wf serve gs://my-bucket/model.gguf
        wf serve hf://Qwen/Qwen3-4B -b vllm
    """
    core.serve(model_path, name=name, backend=backend, context_size=context)


@cli.command("serve-down")
@click.argument("name")
def serve_down(name: str):
    """Stop and remove a deployed service.

    \b
    Example:
        wf serve-down my-service
    """
    core.serve_down(name)


@cli.command("serve-status")
@click.argument("name")
def serve_status(name: str):
    """Get status of a deployed service.

    \b
    Example:
        wf serve-status my-service
    """
    status = core.serve_status(name)
    click.echo(f"Status: {status}")


@cli.command("wandb-status")
@click.option("--entity", "-e", default=None, help="Wandb entity (user/team). Uses WANDB_ENTITY env var if not set.")
@click.option("--project", "-p", default=DEFAULT_WANDB_PROJECT, help="Wandb project name")
@click.option("--limit", "-n", default=5, type=int, help="Number of runs to show")
def wandb_status_cmd(entity: str | None, project: str, limit: int):
    """Check status of recent Wandb training runs.

    Shows current state, metrics, and heartbeat for recent runs.

    \b
    Examples:
        wf wandb-status
        wf wandb-status -n 10
        wf wandb-status -e my-team -p my-project

    Note: Set WANDB_ENTITY environment variable to avoid passing -e each time.
    """
    # Use env var if entity not provided on command line
    resolved_entity = entity or get_wandb_entity()
    core.wandb_status(entity=resolved_entity, project=project, limit=limit)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
