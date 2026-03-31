"""TIDE CLI — run batch jobs on the CSU TIDE cluster via JupyterHub.

Usage:
    tide verify                          Test connection and show server status
    tide gpuinfo                         Show nvidia-smi from the TIDE server
    tide run <script.py> [--timeout N]   Upload and execute a Python script
    tide exec "<code>"  [--timeout N]    Execute an inline code string
    tide status                          Show JupyterHub server status
    tide start                           Start the JupyterHub server
    tide stop                            Stop the JupyterHub server
    tide ls [remote/path]                List files on the TIDE server
    tide upload <local> <remote>         Upload a file
    tide download <remote> <local>       Download a file
    tide kernels                         List running kernels
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

_ITERARE_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[3]))
load_dotenv(_ITERARE_ROOT / ".env")

from .client import TIDEClient
from .jobs import run_script, run_code, gpu_info

console = Console()


def _client() -> TIDEClient:
    return TIDEClient()


def cmd_verify(_args: list[str]) -> None:
    try:
        c = _client()
        status = c.verify_connection()
        ready = status["ready"]
        icon = "[green]✓[/green]" if ready else "[yellow]○[/yellow]"
        console.print(f"{icon} Connected as [bold]{c.username}[/bold]")
        console.print(f"  Server ready: {ready}")
        if status.get("started"):
            console.print(f"  Started:      {status['started'][:19]}")
        if status.get("profile"):
            p = status["profile"]
            console.print(f"  GPU:          {p.get('gpu', 'none')}")
            console.print(f"  CPU:          {p.get('cpu', '?')}")
            console.print(f"  RAM:          {p.get('ram', '?')}")
            console.print(f"  Image:        {p.get('image', '?')}")
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        sys.exit(1)


def cmd_gpuinfo(_args: list[str]) -> None:
    try:
        info = gpu_info(_client())
        console.print(info)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def cmd_run(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="tide run")
    p.add_argument("script", help="Local Python script to upload and execute")
    p.add_argument("--timeout", type=int, default=3600, help="Max seconds (default 3600)")
    p.add_argument("--no-stream", action="store_true", help="Suppress live output")
    opts = p.parse_args(args)

    def on_output(text):
        if not opts.no_stream:
            console.print(text, end="")

    console.print(f"[dim]Submitting {opts.script} to TIDE...[/dim]")
    try:
        result = run_script(_client(), opts.script, timeout=opts.timeout, on_output=on_output)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print()
    if result.status == "complete":
        console.print(f"[green]Complete[/green] in {result.elapsed_seconds}s")
    else:
        console.print(f"[red]Failed[/red] after {result.elapsed_seconds}s")
        if result.error:
            console.print(f"[red]{result.error}[/red]")
        sys.exit(1)


def cmd_exec(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="tide exec")
    p.add_argument("code", help="Python code string to execute")
    p.add_argument("--timeout", type=int, default=60)
    opts = p.parse_args(args)

    def on_output(text):
        console.print(text, end="")

    try:
        result = run_code(_client(), opts.code, timeout=opts.timeout, on_output=on_output)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print()
    if result.error:
        console.print(f"[red]{result.error}[/red]")
        sys.exit(1)


def cmd_status(_args: list[str]) -> None:
    cmd_verify(_args)


def cmd_start(_args: list[str]) -> None:
    console.print("[dim]Starting server...[/dim]")
    try:
        status = _client().start_server(wait=True)
        console.print(f"[green]Server ready.[/green] Started: {status.get('started', '?')[:19]}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def cmd_stop(_args: list[str]) -> None:
    try:
        _client().stop_server()
        console.print("[yellow]Server stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def cmd_ls(args: list[str]) -> None:
    path = args[0] if args else ""
    try:
        files = _client().list_files(path)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Type", width=6)
    table.add_column("Name")
    table.add_column("Size", justify="right")
    for f in files:
        size = f"{f['size']:,}" if f.get("size") else "-"
        icon = "[blue]dir[/blue]" if f["type"] == "directory" else "file"
        table.add_row(icon, f["name"], size)
    console.print(table)


def cmd_upload(args: list[str]) -> None:
    if len(args) < 2:
        console.print("[red]Usage: tide upload <local_path> <remote_path>[/red]")
        sys.exit(1)
    try:
        _client().upload_file(args[0], args[1])
        console.print(f"[green]Uploaded:[/green] {args[0]} → {args[1]}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def cmd_download(args: list[str]) -> None:
    if len(args) < 2:
        console.print("[red]Usage: tide download <remote_path> <local_path>[/red]")
        sys.exit(1)
    try:
        _client().download_file(args[0], args[1])
        console.print(f"[green]Downloaded:[/green] {args[1]}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def cmd_kernels(_args: list[str]) -> None:
    try:
        kernels = _client().list_kernels()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    if not kernels:
        console.print("[dim]No running kernels.[/dim]")
        return
    for k in kernels:
        console.print(f"  {k['id']}  {k.get('name', '?')}  last_activity={k.get('last_activity', '?')[:19]}")


_COMMANDS = {
    "verify": cmd_verify,
    "gpuinfo": cmd_gpuinfo,
    "run": cmd_run,
    "exec": cmd_exec,
    "status": cmd_status,
    "start": cmd_start,
    "stop": cmd_stop,
    "ls": cmd_ls,
    "upload": cmd_upload,
    "download": cmd_download,
    "kernels": cmd_kernels,
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
