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

from wf_deployer import core
from wf_deployer.constants import (
    DEFAULT_SMOKE_TEST_MODEL,
    DEFAULT_WANDB_PROJECT,
    DEFAULT_CONTEXT_SIZE,
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
@click.option("--stage", "-s", required=True, type=float, help="Training stage (1, 1.9, 2, 3)")
@click.option("--scale", default=None,
              type=click.Choice(["dev", "small", "medium", "large", "xlarge"]),
              help="GPU scale: dev (1xA10G), small (1xH100), medium (2xH100), large (4xH100), xlarge (8xH100)")
@click.option("--resume", "-r", default=None, help="Resume from checkpoint (local path or gs://)")
@click.option("--cloud", "-c", default="nebius", type=click.Choice(["gcp", "nebius", "runpod"]), help="Cloud provider")
@click.option("--detach/--no-detach", default=True, help="Return immediately or wait")
@click.pass_context
def train(ctx, model: str, stage: float, scale: str, resume: str, cloud: str, detach: bool):
    """Launch a training job.

    Any extra arguments are passed directly to Hydra.

    \b
    Scales (GPU profiles):
        dev:    1x A10G  (cheap, for testing)
        small:  1x H100  (default)
        medium: 2x H100
        large:  4x H100
        xlarge: 8x H100

    \b
    Examples:
        wf train -m qwen3_4b -s 2
        wf train -m qwen3_4b -s 2 --scale large
        wf train -m qwen3_4b -s 2 training.lr=1e-4
        wf train -m smollm2_135m -s 1.9 --no-detach
        wf train -m qwen3_4b -s 2 --resume gs://bucket/checkpoint.pt
    """
    # Extra args are Hydra overrides
    overrides = list(ctx.args)

    core.train(
        model=model,
        stage=stage,
        scale=scale,
        overrides=overrides,
        cloud=cloud,
        detach=detach,
        resume_checkpoint=resume,  # Pass resume separately to avoid Hydra issues
    )


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--model", "-m", required=True, help="Model config (e.g., qwen3_4b)")
@click.option("--source", "-s", default=None, help="Source checkpoint (hf://org/model, gs://, or local path)")
@click.option("--scale", default=None,
              type=click.Choice(["dev", "small", "medium", "large", "xlarge"]),
              help="GPU scale profile")
@click.option("--detach/--no-detach", default=True, help="Return immediately or wait")
@click.pass_context
def dlm(ctx, model: str, source: str | None, scale: str, detach: bool):
    """Launch DLM (Fast-dLLM v2) training for ~2.5x faster inference.

    Converts a BitNet checkpoint to a Diffusion LLM using the
    Fast-dLLM v2 SFT recipe.

    \b
    Examples:
        wf dlm -m qwen3_4b -s hf://org/checkpoint
        wf dlm -m qwen3_4b -s gs://bucket/checkpoint
        wf dlm -m smollm2_135m --no-detach
        wf dlm -m qwen3_4b conversion.total_tokens=500000000
    """
    overrides = ctx.args

    core.train_dlm(
        model=model,
        source=source,
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


@cli.command()
@click.option("--model", "-m", default=DEFAULT_SMOKE_TEST_MODEL, help="Model to test (default: smollm2_135m)")
def smoke(model: str):
    """Run a quick smoke test.

    \b
    Examples:
        wf smoke
        wf smoke -m qwen3_4b
    """
    result = core.smoke_test(model=model)
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
