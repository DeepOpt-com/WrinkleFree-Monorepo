"""Command-line interface for DLM Converter.

Usage:
    wf-dlm convert -m qwen3_4b -c hf://org/checkpoint
    wf-dlm validate -m ./outputs/dlm/qwen3_4b
    wf-dlm logs <run_id>
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from wf_dlm_converter import core
from wf_dlm_converter.constants import (
    DEFAULT_MODEL,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_TOTAL_TOKENS,
    DEFAULT_LEARNING_RATE,
)

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="wf-dlm")
def cli():
    """WrinkleFree DLM Converter - Convert BitNet models to Diffusion LLMs."""
    pass


@cli.command()
@click.option(
    "--model", "-m",
    default=DEFAULT_MODEL,
    help=f"Model config (default: {DEFAULT_MODEL})",
)
@click.option(
    "--checkpoint", "-c",
    required=True,
    help="Path to BitNet checkpoint (local, hf://, gs://)",
)
@click.option(
    "--output", "-o",
    default="./outputs/dlm",
    help="Output directory for converted model",
)
@click.option(
    "--tokens", "-t",
    default=DEFAULT_TOTAL_TOKENS,
    type=int,
    help=f"Total fine-tuning tokens (default: {DEFAULT_TOTAL_TOKENS:,})",
)
@click.option(
    "--block-size", "-b",
    default=DEFAULT_BLOCK_SIZE,
    type=int,
    help=f"Block size for block diffusion (default: {DEFAULT_BLOCK_SIZE})",
)
@click.option(
    "--steps", "-s",
    default=DEFAULT_DIFFUSION_STEPS,
    type=int,
    help=f"Diffusion steps per block (default: {DEFAULT_DIFFUSION_STEPS})",
)
@click.option(
    "--lr",
    default=DEFAULT_LEARNING_RATE,
    type=float,
    help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
)
@click.option(
    "--gpu", "-g",
    default="A10G",
    type=click.Choice(["H100", "A10G", "L4", "dev"]),
    help="GPU type for Modal (default: A10G, dev=A10G)",
)
@click.option(
    "--backend",
    default="modal",
    type=click.Choice(["modal", "local"]),
    help="Execution backend (default: modal)",
)
def convert(
    model: str,
    checkpoint: str,
    output: str,
    tokens: int,
    block_size: int,
    steps: int,
    lr: float,
    gpu: str,
    backend: str,
):
    """Convert a BitNet checkpoint to Diffusion LLM.

    Examples:

        # Convert Qwen3-4B BitNet model
        wf-dlm convert -m qwen3_4b -c hf://org/qwen3-4b-bitnet

        # Convert with custom settings
        wf-dlm convert -m smollm2_135m -c ./checkpoints/stage2 \\
            --tokens 500000000 --lr 1e-4

        # Run locally (requires GPU)
        wf-dlm convert -m smollm2_135m -c ./ckpt --backend local
    """
    console.print(f"[bold blue]Converting BitNet model to DLM[/bold blue]")
    console.print(f"  Model: {model}")
    console.print(f"  Checkpoint: {checkpoint}")
    console.print(f"  Tokens: {tokens:,}")
    console.print(f"  Block size: {block_size}")
    console.print(f"  Diffusion steps: {steps}")
    console.print(f"  Learning rate: {lr}")
    console.print(f"  GPU: {gpu}")
    console.print(f"  Backend: {backend}")
    console.print()

    result = core.convert(
        model=model,
        checkpoint_path=checkpoint,
        output_path=output,
        total_tokens=tokens,
        block_size=block_size,
        num_diffusion_steps=steps,
        learning_rate=lr,
        backend=backend,
        gpu=gpu,
    )

    if result.get("status") == "success":
        console.print(f"[bold green]Conversion complete![/bold green]")
        console.print(f"  Output: {result.get('output_path')}")
        console.print(f"  Tokens trained: {result.get('tokens_trained', 0):,}")
    else:
        console.print(f"[bold red]Conversion failed[/bold red]")
        if "error" in result:
            console.print(f"  Error: {result['error']}")

    console.print(f"\n  Run ID: {result.get('run_id')}")


@cli.command()
@click.option(
    "--model-path", "-m",
    required=True,
    help="Path to converted DLM model",
)
@click.option(
    "--prompt", "-p",
    default="Hello, how are you today?",
    help="Test prompt for generation",
)
@click.option(
    "--block-size", "-b",
    default=DEFAULT_BLOCK_SIZE,
    type=int,
    help=f"Block size for generation (default: {DEFAULT_BLOCK_SIZE})",
)
@click.option(
    "--steps", "-s",
    default=DEFAULT_DIFFUSION_STEPS,
    type=int,
    help=f"Diffusion steps per block (default: {DEFAULT_DIFFUSION_STEPS})",
)
@click.option(
    "--max-length", "-l",
    default=128,
    type=int,
    help="Maximum generation length (default: 128)",
)
def validate(
    model_path: str,
    prompt: str,
    block_size: int,
    steps: int,
    max_length: int,
):
    """Validate a converted DLM model works correctly.

    Runs inference and reports quality metrics.

    Examples:

        wf-dlm validate -m ./outputs/dlm/qwen3_4b

        wf-dlm validate -m ./outputs/dlm/smollm2_135m \\
            --prompt "Write a haiku about coding"
    """
    console.print(f"[bold blue]Validating DLM model[/bold blue]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Prompt: {prompt}")
    console.print()

    result = core.validate(
        model_path=model_path,
        test_prompt=prompt,
        block_size=block_size,
        diffusion_steps=steps,
        max_length=max_length,
    )

    if result.get("success"):
        console.print(f"[bold green]Validation passed![/bold green]")
        console.print()
        console.print("[bold]Generated text:[/bold]")
        console.print(result.get("generated_text", ""))
        console.print()

        # Show metrics
        table = Table(title="Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Tokens/second", f"{result.get('tokens_per_second', 0):.2f}")
        table.add_row("New tokens", str(result.get("new_tokens", 0)))
        table.add_row("Elapsed (s)", f"{result.get('elapsed_seconds', 0):.2f}")

        console.print(table)
    else:
        console.print(f"[bold red]Validation failed[/bold red]")
        console.print(f"  Error: {result.get('error')}")


@cli.command()
@click.argument("run_id")
@click.option(
    "--follow", "-f",
    is_flag=True,
    help="Stream logs continuously",
)
def logs(run_id: str, follow: bool):
    """View logs for a conversion job.

    Examples:

        wf-dlm logs dlm-convert-qwen3_4b-abc123

        wf-dlm logs dlm-convert-qwen3_4b-abc123 --follow
    """
    core.logs(run_id, follow=follow)


@cli.command()
@click.argument("run_id")
def cancel(run_id: str):
    """Cancel a running conversion job.

    Example:

        wf-dlm cancel dlm-convert-qwen3_4b-abc123
    """
    console.print(f"Cancelling job: {run_id}")

    result = core.cancel(run_id)

    if result.get("success"):
        console.print("[bold green]Job cancelled[/bold green]")
    else:
        console.print(f"[bold red]Cancel failed: {result.get('error')}[/bold red]")


@cli.command()
def info():
    """Show information about supported models and settings."""
    console.print("[bold blue]WrinkleFree DLM Converter[/bold blue]")
    console.print()

    # Supported models
    console.print("[bold]Supported Models:[/bold]")
    models = [
        ("smollm2_135m", "SmolLM2 135M - Fast iteration"),
        ("smollm2_360m", "SmolLM2 360M - Small but capable"),
        ("qwen3_4b", "Qwen3 4B - Production quality"),
    ]
    for name, desc in models:
        console.print(f"  {name}: {desc}")

    console.print()

    # Default settings
    console.print("[bold]Default Settings:[/bold]")
    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Block size", str(DEFAULT_BLOCK_SIZE))
    table.add_row("Diffusion steps", str(DEFAULT_DIFFUSION_STEPS))
    table.add_row("Total tokens", f"{DEFAULT_TOTAL_TOKENS:,}")
    table.add_row("Learning rate", str(DEFAULT_LEARNING_RATE))

    console.print(table)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
