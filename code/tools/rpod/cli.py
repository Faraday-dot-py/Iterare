"""rpod CLI — run batch jobs on RunPod GPU instances.

Usage:
    rpod verify                           Test API key and show account info
    rpod gpus                             List available GPU types
    rpod pods                             List your current pods
    rpod run <script.py> [options]        Create pod, run script, terminate
    rpod exec "<code>" [options]          Run inline code on a pod
    rpod terminate <pod_id>               Terminate a pod
    rpod stop <pod_id>                    Stop (not terminate) a pod

Options for run/exec:
    --gpu <type>      GPU type ID (default: NVIDIA H100 80GB HBM3)
    --gpu-count N     Number of GPUs (default: 1)
    --image <name>    Docker image (default: runpod/pytorch:2.4.0-...)
    --disk N          Container disk GB (default: 50)
    --timeout N       Max seconds (default: 3600)
    --keep            Do not terminate pod after completion
    --download <remote>:<local>   Download file after execution (repeatable)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

_ITERARE_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[3]))
load_dotenv(_ITERARE_ROOT / ".env")

from .client import RunPodClient, DEFAULT_GPU, DEFAULT_IMAGE, DEFAULT_DISK_GB
from .jobs import run_script, run_code, gpu_info

console = Console()


def _client() -> RunPodClient:
    return RunPodClient()


def cmd_verify(_args: list[str]) -> None:
    import runpod
    try:
        c = _client()
        user = runpod.get_user()
        console.print(f"[green]✓[/green] Authenticated")
        console.print(f"  User ID:    {user.get('id', '?')}")
        console.print(f"  SSH key:    {'registered' if user.get('pubKey') else '[yellow]not set[/yellow]'}")
        volumes = user.get("networkVolumes", [])
        if volumes:
            console.print(f"  Volumes:    {len(volumes)}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def cmd_gpus(_args: list[str]) -> None:
    import runpod
    try:
        gpus = runpod.get_gpus()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", no_wrap=True)
    table.add_column("Display Name")
    table.add_column("VRAM (GB)", justify="right")
    for g in gpus:
        table.add_row(g["id"], g["displayName"], str(g["memoryInGb"]))
    console.print(table)


def cmd_pods(_args: list[str]) -> None:
    try:
        pods = _client().list_pods()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    if not pods:
        console.print("[dim]No pods.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", no_wrap=True)
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("GPU")
    table.add_column("Image")
    for p in pods:
        gpu = p.get("machine", {}).get("gpuDisplayName", "?")
        table.add_row(p["id"], p.get("name", ""), p.get("desiredStatus", "?"), gpu, p.get("imageName", "?")[:40])
    console.print(table)


def _parse_run_args(args: list[str]) -> tuple[str, dict]:
    """Parse run/exec args. Returns (positional_arg, options_dict)."""
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("target")
    p.add_argument("--gpu", default=DEFAULT_GPU)
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--disk", type=int, default=DEFAULT_DISK_GB)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--download", action="append", default=[])
    opts = p.parse_args(args)
    downloads = []
    for d in opts.download:
        if ":" not in d:
            console.print(f"[red]--download must be remote:local, got: {d}[/red]")
            sys.exit(1)
        remote, local = d.split(":", 1)
        downloads.append((remote, local))
    return opts.target, {
        "gpu_type_id": opts.gpu,
        "gpu_count": opts.gpu_count,
        "image_name": opts.image,
        "container_disk_in_gb": opts.disk,
        "timeout": opts.timeout,
        "terminate_on_complete": not opts.keep,
        "download_paths": downloads,
    }


def cmd_run(args: list[str]) -> None:
    target, kwargs = _parse_run_args(args)

    def on_output(text):
        console.print(text, end="")

    console.print(f"[dim]Creating pod and running {target}...[/dim]")
    try:
        result = run_script(_client(), target, on_output=on_output, **kwargs)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print()
    if result.status == "complete":
        console.print(f"[green]Complete[/green] in {result.elapsed_seconds}s  pod={result.pod_id}")
    else:
        console.print(f"[red]Failed[/red] (exit {result.exit_code}) after {result.elapsed_seconds}s")
        if result.stderr:
            console.print(f"[red]{result.stderr[-2000:]}[/red]")
        sys.exit(1)


def cmd_exec(args: list[str]) -> None:
    target, kwargs = _parse_run_args(args)
    kwargs.pop("download_paths", None)  # not supported for inline exec

    def on_output(text):
        console.print(text, end="")

    console.print(f"[dim]Creating pod and running inline code...[/dim]")
    try:
        result = run_code(_client(), target, on_output=on_output, **kwargs)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print()
    if result.status == "complete":
        console.print(f"[green]Complete[/green] in {result.elapsed_seconds}s")
    else:
        console.print(f"[red]Failed[/red] (exit {result.exit_code})")
        if result.stderr:
            console.print(f"[red]{result.stderr[-2000:]}[/red]")
        sys.exit(1)


def cmd_terminate(args: list[str]) -> None:
    if not args:
        console.print("[red]Usage: rpod terminate <pod_id>[/red]")
        sys.exit(1)
    try:
        _client().terminate_pod(args[0])
        console.print(f"[green]Terminated:[/green] {args[0]}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def cmd_stop(args: list[str]) -> None:
    if not args:
        console.print("[red]Usage: rpod stop <pod_id>[/red]")
        sys.exit(1)
    try:
        _client().stop_pod(args[0])
        console.print(f"[yellow]Stopped:[/yellow] {args[0]}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


_COMMANDS = {
    "verify": cmd_verify,
    "gpus": cmd_gpus,
    "pods": cmd_pods,
    "run": cmd_run,
    "exec": cmd_exec,
    "terminate": cmd_terminate,
    "stop": cmd_stop,
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        console.print(__doc__)
        return
    cmd = args[0]
    if cmd not in _COMMANDS:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        console.print(__doc__)
        sys.exit(1)
    _COMMANDS[cmd](args[1:])


if __name__ == "__main__":
    main()
