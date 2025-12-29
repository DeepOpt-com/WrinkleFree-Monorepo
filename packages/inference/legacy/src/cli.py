"""CLI entry point for WrinkleFree Inference Engine."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option()
def main():
    """WrinkleFree Inference Engine - BitNet model serving."""
    pass


@main.command()
@click.option(
    "--hf-repo",
    default="microsoft/BitNet-b1.58-2B-4T",
    help="HuggingFace repository ID",
)
@click.option(
    "--quant-type",
    default="i2_s",
    type=click.Choice(["i2_s", "tl1", "tl2"]),
    help="Quantization type",
)
@click.option(
    "--bitnet-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to BitNet installation",
)
def convert(hf_repo: str, quant_type: str, bitnet_path: Optional[Path]):
    """Convert a HuggingFace model to GGUF format."""
    from wrinklefree_inference.converter.hf_to_gguf import (
        ConversionConfig,
        HFToGGUFConverter,
    )

    console.print(f"[bold]Converting {hf_repo}[/bold]")
    console.print(f"Quantization type: {quant_type}")

    try:
        converter = HFToGGUFConverter(bitnet_path)
        config = ConversionConfig(hf_repo=hf_repo, quant_type=quant_type)

        def progress(msg: str):
            console.print(f"  {msg}")

        gguf_path = converter.convert(config, progress_callback=progress)
        console.print(f"\n[green]Success![/green] Model saved to: {gguf_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to GGUF model file",
)
@click.option("--port", "-p", default=8080, help="Server port")
@click.option("--host", "-h", default="0.0.0.0", help="Server host")
@click.option("--threads", "-t", default=0, help="Number of threads (0=auto)")
@click.option("--context-size", "-c", default=4096, help="Context size (KV cache)")
@click.option(
    "--bitnet-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to BitNet installation",
)
def serve(
    model: Path,
    port: int,
    host: str,
    threads: int,
    context_size: int,
    bitnet_path: Optional[Path],
):
    """Start the BitNet inference server."""
    from wrinklefree_inference.server.bitnet_server import (
        BitNetServer,
        get_default_bitnet_path,
    )

    if bitnet_path is None:
        bitnet_path = get_default_bitnet_path()

    console.print(f"[bold]Starting inference server[/bold]")
    console.print(f"Model: {model}")
    console.print(f"Port: {port}")
    console.print(f"Context size: {context_size}")

    server = BitNetServer(
        bitnet_path=bitnet_path,
        model_path=model,
        port=port,
        host=host,
        num_threads=threads,
        context_size=context_size,
        continuous_batching=True,
    )

    try:
        console.print("\n[yellow]Starting server...[/yellow]")
        server.start(wait_for_ready=True, timeout=120)
        console.print(f"[green]Server running at http://{host}:{port}[/green]")
        console.print("Press Ctrl+C to stop")

        # Keep running
        import time
        while server.is_running():
            time.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        server.stop()
        console.print("[green]Server stopped[/green]")


@main.command()
@click.option(
    "--url",
    default="http://localhost:8080",
    help="Inference server URL",
)
@click.option("--prompt", "-p", required=True, help="Prompt to generate from")
@click.option("--max-tokens", "-n", default=128, help="Maximum tokens to generate")
@click.option("--temperature", "-t", default=0.7, help="Sampling temperature")
@click.option("--stream/--no-stream", default=True, help="Stream output")
def generate(url: str, prompt: str, max_tokens: int, temperature: float, stream: bool):
    """Generate text from a prompt."""
    from wrinklefree_inference.client.bitnet_client import BitNetClient

    # Parse URL
    url_clean = url.replace("http://", "").replace("https://", "")
    if ":" in url_clean:
        host, port_str = url_clean.split(":")
        port = int(port_str.split("/")[0])
    else:
        host = url_clean.split("/")[0]
        port = 8080

    client = BitNetClient(host=host, port=port)

    if not client.health_check():
        console.print(f"[red]Error:[/red] Server not available at {url}")
        sys.exit(1)

    if stream:
        for chunk in client.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            console.print(chunk, end="")
        console.print()
    else:
        response = client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        console.print(response)


@main.command()
@click.option(
    "--url",
    default="http://localhost:8080",
    help="Inference server URL",
)
@click.option("--timeout", default=60, help="Request timeout in seconds")
def validate(url: str, timeout: int):
    """Validate KV cache behavior."""
    from wrinklefree_inference.kv_cache.validator import run_kv_cache_validation

    console.print(f"[bold]Validating KV cache at {url}[/bold]\n")

    metrics = run_kv_cache_validation(url, timeout)

    table = Table(title="KV Cache Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Prefix Speedup", f"{metrics.prefix_speedup:.2f}x")
    table.add_row("First Request Latency", f"{metrics.first_request_latency_ms:.1f}ms")
    table.add_row("Second Request Latency", f"{metrics.second_request_latency_ms:.1f}ms")
    table.add_row("Context Limit Handled", str(metrics.context_limit_handled))
    table.add_row("Concurrent Success Rate", f"{metrics.concurrent_success_rate*100:.0f}%")

    console.print(table)

    if metrics.errors:
        console.print("\n[red]Errors:[/red]")
        for error in metrics.errors:
            console.print(f"  - {error}")
        sys.exit(1)

    # Check pass criteria
    if metrics.concurrent_success_rate < 0.8:
        console.print("\n[red]FAIL:[/red] Concurrent success rate too low")
        sys.exit(1)

    console.print("\n[green]PASS:[/green] KV cache validation successful")


@main.command()
@click.option(
    "--url",
    default="http://localhost:8080",
    help="Inference server URL",
)
@click.option("--port", "-p", default=8501, help="Streamlit port")
def chat(url: str, port: int):
    """Launch Streamlit chat interface."""
    import subprocess
    import os

    console.print(f"[bold]Launching chat interface[/bold]")
    console.print(f"Server URL: {url}")
    console.print(f"Chat UI will be available at: http://localhost:{port}")

    # Get the chat module path
    chat_module = Path(__file__).parent / "ui" / "chat.py"

    env = os.environ.copy()
    env["INFERENCE_URL"] = url

    try:
        subprocess.run(
            ["streamlit", "run", str(chat_module), "--server.port", str(port)],
            env=env,
            check=True,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Chat interface stopped[/yellow]")
    except FileNotFoundError:
        console.print("[red]Error:[/red] streamlit not found. Install with: pip install streamlit")
        sys.exit(1)


@main.command()
@click.option(
    "--bitnet-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to BitNet installation",
)
def list_models(bitnet_path: Optional[Path]):
    """List available GGUF models."""
    from wrinklefree_inference.converter.hf_to_gguf import HFToGGUFConverter

    try:
        converter = HFToGGUFConverter(bitnet_path)
        models = converter.list_available_models()

        if not models:
            console.print("[yellow]No models found[/yellow]")
            console.print("Run 'wrinklefree-inference convert' to download a model")
            return

        table = Table(title="Available Models")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="green")

        for model_path in models:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            table.add_row(str(model_path.name), f"{size_mb:.1f} MB")

        console.print(table)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command("benchmark-cost")
@click.option(
    "--url",
    default="http://localhost:8080",
    help="Inference server URL",
)
@click.option(
    "--hardware",
    required=True,
    type=click.Choice(["a40", "l4", "cpu_16", "cpu_32", "cpu_64"]),
    help="Hardware configuration being benchmarked",
)
@click.option(
    "--model",
    default="bitnet-2b-4t",
    help="Model being benchmarked",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("results/raw"),
    help="Output directory for results",
)
@click.option(
    "--duration",
    default=60,
    help="Duration per batch size in seconds",
)
def benchmark_cost(
    url: str,
    hardware: str,
    model: str,
    output_dir: Path,
    duration: int,
):
    """Run cost benchmarking against inference server."""
    import asyncio

    console.print("[bold]Cost Benchmarking[/bold]")
    console.print(f"Server: {url}")
    console.print(f"Hardware: {hardware}")
    console.print(f"Model: {model}")

    try:
        # Import benchmark modules
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "benchmark"))
        from benchmark.runner import BenchmarkConfig, BenchmarkRunner
        from benchmark.cost_tracker import CostTracker

        hardware_type = "gpu" if hardware in ["a40", "l4"] else "cpu"

        config = BenchmarkConfig(
            model=model,
            model_size="2B",
            quantization="native",
            hardware=hardware,
            hardware_type=hardware_type,
            batch_sizes=[1, 4, 8],
            duration_seconds=duration,
            server_url=url,
        )

        async def run():
            runner = BenchmarkRunner(config, CostTracker())
            return await runner.run()

        result = asyncio.run(run())

        # Display results
        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Throughput", f"{result.tokens_per_second:.1f} tok/s")
        table.add_row("TTFT P50", f"{result.ttft_p50_ms:.0f}ms")
        table.add_row("Latency P99", f"{result.latency_p99_ms:.0f}ms")
        table.add_row("Hardware Cost", f"${result.hardware_cost_per_hour:.2f}/hr")
        table.add_row("Cost/1M Tokens", f"${result.cost_per_million_tokens:.4f}")
        table.add_row("Cost/1M @ 70%", f"${result.cost_per_million_at_70pct:.4f}")

        console.print(table)

        # Save result
        output_path = result.save(output_dir)
        console.print(f"\n[green]Saved:[/green] {output_path}")

    except ImportError as e:
        console.print(f"[red]Error:[/red] Missing benchmark dependencies: {e}")
        console.print("Install with: uv sync --extra benchmark")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command("naive-convert")
@click.option(
    "--model-id",
    required=True,
    help="HuggingFace model ID to convert",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("models/naive"),
    help="Output directory for converted model",
)
@click.option(
    "--use-gpu/--no-gpu",
    default=True,
    help="Use GPU for conversion",
)
@click.option(
    "--estimate-only",
    is_flag=True,
    help="Only estimate memory requirements",
)
def naive_convert(
    model_id: str,
    output_dir: Path,
    use_gpu: bool,
    estimate_only: bool,
):
    """Convert model to naive ternary format for benchmarking.

    WARNING: This produces LOW QUALITY outputs.
    Only use for speed/cost benchmarking, NOT production.
    """
    console.print("[bold yellow]WARNING: Naive conversion produces low quality outputs![/bold yellow]")
    console.print("This is only for speed/cost benchmarking.\n")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "benchmark"))
        from benchmark.naive_converter import NaiveConverter, ConversionConfig

        config = ConversionConfig(
            model_id=model_id,
            output_dir=output_dir,
            use_gpu=use_gpu,
        )

        converter = NaiveConverter(config)

        # Estimate memory
        estimates = converter.estimate_memory_requirements()
        if "error" not in estimates:
            console.print(f"Model: {model_id}")
            console.print(f"Estimated memory: {estimates['total_recommended_gb']:.1f} GB")

        if estimate_only:
            return

        console.print("\n[yellow]Starting conversion...[/yellow]")

        result = converter.convert()

        if result.success:
            console.print(f"\n[green]Success![/green]")
            console.print(f"Output: {result.output_path}")
            console.print(f"Compression: {result.compression_ratio:.1f}x")
        else:
            console.print(f"\n[red]Failed:[/red] {result.error}")
            sys.exit(1)

    except ImportError as e:
        console.print(f"[red]Error:[/red] Missing conversion dependencies: {e}")
        console.print("Install with: uv sync --extra convert")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
