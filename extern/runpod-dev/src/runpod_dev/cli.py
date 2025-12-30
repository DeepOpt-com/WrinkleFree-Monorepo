"""CLI interface for runpod-dev."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import DEFAULT_GPU, DEFAULT_IMAGE, DEFAULT_REMOTE_DIR, DEFAULT_VOLUME_SIZE, list_gpu_types
from .pod import PodManager
from .setup import (
    attach_tmux_session,
    create_tmux_session,
    install_dependencies,
    run_uv_sync,
    setup_gcs_credentials,
    wait_for_ssh,
)
from .sync import SyncManager, rsync_git_files, terminate_all_sessions

console = Console()
error_console = Console(stderr=True)


class OutputFormat:
    """Output format helper for human/JSON output."""

    def __init__(self, json_mode: bool = False):
        self.json_mode = json_mode
        self._data: dict = {}

    def print(self, message: str, style: str | None = None) -> None:
        """Print a message (human mode only)."""
        if not self.json_mode:
            console.print(message, style=style)

    def error(self, message: str) -> None:
        """Print an error message."""
        if self.json_mode:
            self._data["error"] = message
        else:
            error_console.print(f"[red]Error:[/red] {message}")

    def success(self, message: str) -> None:
        """Print a success message."""
        if self.json_mode:
            self._data["success"] = True
            self._data["message"] = message
        else:
            console.print(f"[green]{message}[/green]")

    def set(self, key: str, value) -> None:
        """Set a JSON output field."""
        self._data[key] = value

    def output(self) -> None:
        """Output final JSON (json mode only)."""
        if self.json_mode:
            print(json.dumps(self._data, indent=2))


@click.group()
@click.option("--json", "json_output", is_flag=True, help="Output JSON (for AI/programmatic use)")
@click.version_option(version="0.1.0")
@click.pass_context
def cli(ctx: click.Context, json_output: bool) -> None:
    """RunPod development workflow tool.

    Launch, sync, and develop on remote GPU instances.

    \b
    Examples:
      runpod-dev launch my-dev           # Launch H100 instance
      runpod-dev launch my-dev --gpu A100  # Launch A100 instance
      runpod-dev connect my-dev          # Get SSH command
      runpod-dev sync my-dev             # Start live file sync
      runpod-dev stop my-dev             # Stop instance
      runpod-dev list                    # List all instances
    """
    ctx.ensure_object(dict)
    ctx.obj["output"] = OutputFormat(json_output)


@cli.command()
@click.argument("name")
@click.option("--gpu", "-g", default=DEFAULT_GPU, help=f"GPU type (default: {DEFAULT_GPU})")
@click.option("--gpu-count", "-n", default=1, type=int, help="Number of GPUs (1-8, single node)")
@click.option("--image", "-i", default=DEFAULT_IMAGE, help="Docker image")
@click.option("--volume-size", "-v", default=DEFAULT_VOLUME_SIZE, type=int, help=f"Volume size in GB (default: {DEFAULT_VOLUME_SIZE})")
@click.option("--local", "-l", default=".", help="Local directory to sync")
@click.option("--remote", "-r", default=DEFAULT_REMOTE_DIR, help="Remote directory")
@click.option("--no-sync", is_flag=True, help="Skip file sync")
@click.option("--no-setup", is_flag=True, help="Skip dependency installation")
@click.option("--watch/--no-watch", default=False, help="Start live mutagen sync (default: one-time rsync)")
@click.option("--credentials", "-c", default=None, help="Credentials directory to sync")
@click.option("--tmux/--no-tmux", default=True, help="Create/attach tmux session")
@click.pass_context
def launch(
    ctx: click.Context,
    name: str,
    gpu: str,
    gpu_count: int,
    image: str,
    volume_size: int,
    local: str,
    remote: str,
    no_sync: bool,
    no_setup: bool,
    watch: bool,
    credentials: str | None,
    tmux: bool,
) -> None:
    """Launch or connect to a RunPod instance.

    If an instance with NAME already exists and is running, connects to it.
    If stopped, resumes it. Otherwise creates a new instance.

    \b
    Examples:
      runpod-dev launch dev              # Create/connect with defaults
      runpod-dev launch dev --gpu A100   # Use A100 instead of H100
      runpod-dev launch dev --no-sync    # Skip file syncing
      runpod-dev launch dev --no-tmux    # Don't attach tmux
    """
    out = ctx.obj["output"]

    try:
        manager = PodManager()

        # Check for existing pod
        existing = manager.find_by_name(name)

        if existing:
            if existing.status == "RUNNING":
                out.print(f"Found existing running pod '[cyan]{name}[/cyan]'")
                pod = existing
            elif existing.status == "EXITED":
                out.print(f"Resuming stopped pod '[cyan]{name}[/cyan]'...")
                pod = manager.resume(name)
            else:
                out.error(f"Pod '{name}' is in state '{existing.status}'. Please wait or terminate it.")
                out.output()
                sys.exit(1)
        else:
            gpu_desc = f"{gpu_count}x {gpu}" if gpu_count > 1 else gpu
            out.print(f"Creating pod '[cyan]{name}[/cyan]' with [yellow]{gpu_desc}[/yellow]...")
            pod = manager.create(name, gpu=gpu, gpu_count=gpu_count, image=image, volume_size=volume_size)
            out.print(f"Pod created (id: {pod.id}). Waiting for ready", style="dim")

            # Wait for pod to be ready
            pod = manager.wait_for_ready(pod.id)
            out.success("Pod is ready!")

        if not pod.ssh_info:
            out.error("Could not get SSH info for pod")
            out.output()
            sys.exit(1)

        # Set JSON output fields
        out.set("pod", pod.to_dict())
        out.set("ssh_command", pod.ssh_info.ssh_command)

        out.print(f"\n[bold]SSH:[/bold] {pod.ssh_info.ssh_command}")

        # Wait for SSH to be available
        out.print("Waiting for SSH...", style="dim")
        if not wait_for_ssh(pod.ssh_info):
            out.error("SSH connection timed out")
            out.output()
            sys.exit(1)

        out.success("SSH is available!")

        # Install dependencies
        if not no_setup:
            out.print("\n[bold]Installing dependencies...[/bold]")
            success = install_dependencies(
                pod.ssh_info,
                on_progress=lambda msg: out.print(f"  {msg}", style="dim"),
            )
            if not success:
                out.error("Failed to install dependencies")
            else:
                out.success("Dependencies installed!")

        # File sync
        if not no_sync:
            out.print(f"\n[bold]Syncing git-tracked files...[/bold]")
            out.print(f"  Local:  {Path(local).resolve()}")
            out.print(f"  Remote: {pod.ssh_info.host}:{remote}")

            # Use rsync with git ls-files (fast, only tracked files)
            success = rsync_git_files(
                pod.ssh_info,
                local,
                remote,
                credentials_path=credentials,
                verbose=not out.json_mode,
            )

            if success:
                out.success("Files synced!")
            else:
                out.error("File sync failed")

            # Optionally start live mutagen sync
            if watch:
                out.print("Starting live sync with mutagen...")
                sync_manager = SyncManager(pod.ssh_info)
                sync_manager.create_session(name, local, remote)
                out.print("Live sync is active. Changes will sync automatically.")
                out.set("sync_active", True)

            # Set up GCS credentials if they exist
            if credentials:
                out.print("\n[bold]Setting up GCS credentials...[/bold]")
                setup_gcs_credentials(
                    pod.ssh_info,
                    remote,
                    credentials_subdir=Path(credentials).name,
                    on_progress=lambda msg: out.print(f"  {msg}", style="dim"),
                )

        # Run uv sync
        if not no_setup and not no_sync:
            out.print("\n[bold]Running uv sync...[/bold]")
            if run_uv_sync(pod.ssh_info, remote):
                out.success("uv sync completed!")
            else:
                out.print("[yellow]Warning: uv sync may have failed[/yellow]")

        # Attach to tmux
        if tmux and not out.json_mode:
            out.print(f"\n[bold]Attaching to tmux session '{name}'...[/bold]")
            create_tmux_session(pod.ssh_info, name, remote)
            attach_tmux_session(pod.ssh_info, name)

        out.output()

    except Exception as e:
        out.error(str(e))
        out.output()
        sys.exit(1)


@cli.command()
@click.argument("name")
@click.pass_context
def connect(ctx: click.Context, name: str) -> None:
    """Print SSH command to connect to an instance.

    \b
    Examples:
      runpod-dev connect dev
      runpod-dev connect dev --json  # Get JSON with SSH info
    """
    out = ctx.obj["output"]

    try:
        manager = PodManager()
        pod = manager.find_by_name(name)

        if not pod:
            out.error(f"Pod '{name}' not found")
            out.output()
            sys.exit(1)

        if pod.status != "RUNNING":
            out.error(f"Pod '{name}' is not running (status: {pod.status})")
            out.output()
            sys.exit(1)

        if not pod.ssh_info:
            out.error(f"No SSH info available for pod '{name}'")
            out.output()
            sys.exit(1)

        out.set("pod", pod.to_dict())
        out.set("ssh_command", pod.ssh_info.ssh_command)

        if out.json_mode:
            out.output()
        else:
            console.print(pod.ssh_info.ssh_command)

    except Exception as e:
        out.error(str(e))
        out.output()
        sys.exit(1)


@cli.command()
@click.argument("name")
@click.option("--local", "-l", default=".", help="Local directory to sync")
@click.option("--remote", "-r", default=DEFAULT_REMOTE_DIR, help="Remote directory")
@click.option("--watch", "-w", is_flag=True, help="Start live mutagen sync after initial rsync")
@click.option("--monitor", "-m", is_flag=True, help="Monitor sync status")
@click.option("--credentials", "-c", default=None, help="Credentials directory to sync")
@click.pass_context
def sync(
    ctx: click.Context,
    name: str,
    local: str,
    remote: str,
    watch: bool,
    monitor: bool,
    credentials: str | None,
) -> None:
    """Start or manage file sync for an instance.

    By default, syncs only git-tracked files using rsync (fast, minimal).
    Use --watch to additionally start live mutagen sync for ongoing development.

    \b
    Examples:
      runpod-dev sync dev                    # One-time rsync of git files
      runpod-dev sync dev --watch            # Rsync + live mutagen sync
      runpod-dev sync dev --local ./src      # Sync specific directory
      runpod-dev sync dev --monitor          # Monitor live sync status
    """
    out = ctx.obj["output"]

    try:
        manager = PodManager()
        pod = manager.find_by_name(name)

        if not pod:
            out.error(f"Pod '{name}' not found")
            out.output()
            sys.exit(1)

        if not pod.ssh_info:
            out.error(f"No SSH info available for pod '{name}'")
            out.output()
            sys.exit(1)

        if monitor:
            # Check status of live sync
            sync_manager = SyncManager(pod.ssh_info)
            status = sync_manager.get_status(name)
            if status:
                out.set("sync_status", status.to_dict())
                if out.json_mode:
                    out.output()
                else:
                    console.print(f"[bold]Session:[/bold] {name}")
                    console.print(f"[bold]Status:[/bold] {status.status}")
                    console.print(f"[bold]Local:[/bold] {status.local_path}")
                    console.print(f"[bold]Remote:[/bold] {status.remote_path}")
                    if status.conflicts > 0:
                        console.print(f"[yellow]Conflicts: {status.conflicts}[/yellow]")
            else:
                out.error(f"No sync session found for '{name}'")
                out.output()
                sys.exit(1)
            return

        # Default: rsync git-tracked files (fast, minimal)
        out.print(f"[bold]Syncing git-tracked files...[/bold]")
        out.print(f"  Local:  {Path(local).resolve()}")
        out.print(f"  Remote: {pod.ssh_info.host}:{remote}")

        success = rsync_git_files(
            pod.ssh_info,
            local,
            remote,
            credentials_path=credentials,
            verbose=not out.json_mode,
        )

        if success:
            out.success("Git files synced!")
            out.set("sync_complete", True)
        else:
            out.error("Rsync failed")
            out.output()
            sys.exit(1)

        # Set up GCS credentials if they exist
        if credentials:
            out.print("\n[bold]Setting up GCS credentials...[/bold]")
            setup_gcs_credentials(
                pod.ssh_info,
                remote,
                credentials_subdir=Path(credentials).name,
                on_progress=lambda msg: out.print(f"  {msg}", style="dim"),
            )

        # Optionally start live mutagen sync
        if watch:
            out.print("\n[bold]Starting live mutagen sync...[/bold]")
            sync_manager = SyncManager(pod.ssh_info)
            sync_manager.create_session(name, local, remote)
            out.print("Live sync active. Changes will sync automatically.")
            out.set("watch_active", True)

        out.output()

    except Exception as e:
        out.error(str(e))
        out.output()
        sys.exit(1)


@cli.command()
@click.argument("name")
@click.option("--terminate", "-t", is_flag=True, help="Permanently delete the pod")
@click.option("--sync-only", is_flag=True, help="Only stop sync, not the pod")
@click.pass_context
def stop(
    ctx: click.Context,
    name: str,
    terminate: bool,
    sync_only: bool,
) -> None:
    """Stop or terminate a RunPod instance.

    By default, stops the pod (data preserved, can be resumed).
    Use --terminate to permanently delete the pod.

    \b
    Examples:
      runpod-dev stop dev             # Stop (can resume later)
      runpod-dev stop dev --terminate # Delete permanently
      runpod-dev stop dev --sync-only # Only stop file sync
    """
    out = ctx.obj["output"]

    try:
        manager = PodManager()
        pod = manager.find_by_name(name)

        if not pod and not sync_only:
            out.error(f"Pod '{name}' not found")
            out.output()
            sys.exit(1)

        # Stop sync session if it exists
        if pod and pod.ssh_info:
            sync_manager = SyncManager(pod.ssh_info)
            sync_manager.terminate_session(name)
            out.print(f"Sync session terminated")

        if sync_only:
            out.success("Sync stopped")
            out.output()
            return

        # Stop or terminate pod
        if terminate:
            manager.terminate(name)
            out.success(f"Pod '{name}' terminated (deleted)")
            out.set("action", "terminated")
        else:
            manager.stop(name)
            out.success(f"Pod '{name}' stopped (can be resumed)")
            out.set("action", "stopped")

        out.output()

    except Exception as e:
        out.error(str(e))
        out.output()
        sys.exit(1)


@cli.command("list")
@click.pass_context
def list_pods(ctx: click.Context) -> None:
    """List all RunPod instances.

    \b
    Examples:
      runpod-dev list
      runpod-dev list --json
    """
    out = ctx.obj["output"]

    try:
        manager = PodManager()
        pods = manager.list_pods()

        out.set("pods", [p.to_dict() for p in pods])

        if out.json_mode:
            out.output()
            return

        if not pods:
            console.print("No pods found")
            return

        table = Table(title="RunPod Instances")
        table.add_column("Name", style="cyan")
        table.add_column("Status")
        table.add_column("GPU")
        table.add_column("SSH")

        for pod in pods:
            status_style = "green" if pod.status == "RUNNING" else "yellow"
            ssh = pod.ssh_info.ssh_command if pod.ssh_info else "-"
            table.add_row(
                pod.name,
                f"[{status_style}]{pod.status}[/{status_style}]",
                pod.gpu_type or "-",
                ssh,
            )

        console.print(table)

    except Exception as e:
        out.error(str(e))
        out.output()
        sys.exit(1)


@cli.command()
@click.pass_context
def gpus(ctx: click.Context) -> None:
    """List available GPU types.

    \b
    Examples:
      runpod-dev gpus
      runpod-dev gpus --json
    """
    out = ctx.obj["output"]

    gpu_list = list_gpu_types()
    out.set("gpus", gpu_list)

    if out.json_mode:
        out.output()
        return

    table = Table(title="Available GPU Types")
    table.add_column("Name", style="cyan")
    table.add_column("RunPod ID")

    for gpu in gpu_list:
        table.add_row(gpu["name"], gpu["id"])

    console.print(table)


@cli.command()
@click.pass_context
def cleanup(ctx: click.Context) -> None:
    """Terminate all runpod-dev sync sessions.

    \b
    Examples:
      runpod-dev cleanup
    """
    out = ctx.obj["output"]

    count = terminate_all_sessions()
    out.set("terminated_sessions", count)
    out.success(f"Terminated {count} sync session(s)")
    out.output()


if __name__ == "__main__":
    cli()
