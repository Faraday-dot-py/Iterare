"""TIDE CLI — submit and manage batch jobs on the CSU/NRP Nautilus cluster.

Usage:
    tide verify                          Test connection to the cluster
    tide submit <script.py> [options]    Submit a Python job
    tide status <job-name>               Get job status
    tide logs <job-name> [--tail N]      Fetch job logs
    tide list [--label key=val]          List jobs in namespace
    tide cancel <job-name>               Delete a job and its pods
    tide wait <job-name> [--timeout N]   Block until job finishes
    tide yaml <manifest.yaml>            Submit a raw Kubernetes Job YAML
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Load .env from iterare root
_ITERARE_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[3]))
load_dotenv(_ITERARE_ROOT / ".env")

from .client import TIDEClient
from .jobs import submit_job, job_status, job_logs, list_jobs, delete_job, wait_for_job
from .manifests import gpu_python_job, cpu_python_job, shell_job, from_yaml

console = Console()


def _client() -> TIDEClient:
    return TIDEClient()


def cmd_verify(_args: list[str]) -> None:
    try:
        info = _client().verify_connection()
        console.print(f"[green]Connected.[/green] Cluster: {info['git_version']} / {info['platform']}")
        console.print(f"Namespace: {os.getenv('TIDE_NAMESPACE', '(not set)')}")
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        sys.exit(1)


def cmd_submit(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="tide submit")
    p.add_argument("script", help="Path to Python script inside the container")
    p.add_argument("--image", default="ghcr.io/csu-tide/hello-csu:main")
    p.add_argument("--gpu", type=int, default=0, help="Number of GPUs (0 = CPU job)")
    p.add_argument("--cpu", type=int, default=4, help="CPU cores")
    p.add_argument("--memory", default="8Gi")
    p.add_argument("--name", default=None)
    p.add_argument("--pvc", action="append", metavar="CLAIM:MOUNT", default=[],
                   help="Mount a PVC: --pvc my-claim:/data")
    p.add_argument("--env", action="append", metavar="KEY=VAL", default=[],
                   help="Set env var: --env SEED=42")
    opts = p.parse_args(args)

    pvc_mounts = {}
    for pvc in opts.pvc:
        if ":" not in pvc:
            console.print(f"[red]Invalid --pvc format '{pvc}'. Use CLAIM:MOUNT_PATH.[/red]")
            sys.exit(1)
        claim, mount = pvc.split(":", 1)
        pvc_mounts[claim] = mount

    env = {}
    for e in opts.env:
        if "=" not in e:
            console.print(f"[red]Invalid --env format '{e}'. Use KEY=VALUE.[/red]")
            sys.exit(1)
        k, v = e.split("=", 1)
        env[k] = v

    if opts.gpu > 0:
        spec = gpu_python_job(
            opts.script, image=opts.image, gpu_count=opts.gpu,
            memory=opts.memory, name=opts.name, pvc_mounts=pvc_mounts, env=env,
        )
    else:
        spec = cpu_python_job(
            opts.script, image=opts.image, cpu_cores=opts.cpu,
            memory=opts.memory, name=opts.name, pvc_mounts=pvc_mounts, env=env,
        )

    try:
        name = submit_job(_client(), spec)
        console.print(f"[green]Submitted:[/green] {name}")
    except Exception as e:
        console.print(f"[red]Submit failed:[/red] {e}")
        sys.exit(1)


def cmd_status(args: list[str]) -> None:
    if not args:
        console.print("[red]Usage: tide status <job-name>[/red]")
        sys.exit(1)
    status = job_status(_client(), args[0])
    _print_status(status)


def cmd_logs(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="tide logs")
    p.add_argument("job_name")
    p.add_argument("--tail", type=int, default=100)
    opts = p.parse_args(args)
    logs = job_logs(_client(), opts.job_name, tail_lines=opts.tail)
    console.print(logs)


def cmd_list(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="tide list")
    p.add_argument("--label", default="", help="Label selector e.g. managed-by=iterare")
    opts = p.parse_args(args)
    jobs = list_jobs(_client(), label_selector=opts.label)
    if not jobs:
        console.print("[dim]No jobs found.[/dim]")
        return
    table = Table(show_header=True)
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Active")
    table.add_column("Succeeded")
    table.add_column("Failed")
    table.add_column("Started")
    style_map = {"complete": "green", "failed": "red", "running": "yellow", "pending": "dim"}
    for j in jobs:
        s = j["status"]
        table.add_row(
            j["name"],
            f"[{style_map.get(s, 'white')}]{s}[/{style_map.get(s, 'white')}]",
            str(j["active"]),
            str(j["succeeded"]),
            str(j["failed"]),
            (j["start_time"] or "")[:19],
        )
    console.print(table)


def cmd_cancel(args: list[str]) -> None:
    if not args:
        console.print("[red]Usage: tide cancel <job-name>[/red]")
        sys.exit(1)
    result = delete_job(_client(), args[0])
    console.print(result)


def cmd_wait(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="tide wait")
    p.add_argument("job_name")
    p.add_argument("--timeout", type=int, default=3600)
    opts = p.parse_args(args)

    def on_poll(status):
        console.print(f"  [dim]{status['name']}: {status['status']}[/dim]")

    try:
        final = wait_for_job(_client(), opts.job_name, timeout=opts.timeout, on_poll=on_poll)
        _print_status(final)
    except TimeoutError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


def cmd_yaml(args: list[str]) -> None:
    if not args:
        console.print("[red]Usage: tide yaml <manifest.yaml>[/red]")
        sys.exit(1)
    c = _client()
    manifest = from_yaml(args[0])
    ns = manifest.get("metadata", {}).get("namespace") or c.namespace
    try:
        result = c.batch.create_namespaced_job(namespace=ns, body=manifest)
        console.print(f"[green]Submitted:[/green] {result.metadata.name}")
    except Exception as e:
        console.print(f"[red]Submit failed:[/red] {e}")
        sys.exit(1)


def _print_status(status: dict) -> None:
    style_map = {"complete": "green", "failed": "red", "running": "yellow", "pending": "dim"}
    s = status["status"]
    console.print(f"[bold]{status['name']}[/bold]  [{style_map.get(s, 'white')}]{s}[/{style_map.get(s, 'white')}]")
    for k in ("active", "succeeded", "failed", "start_time", "completion_time"):
        if status.get(k) is not None:
            console.print(f"  {k}: {status[k]}")


_COMMANDS = {
    "verify": cmd_verify,
    "submit": cmd_submit,
    "status": cmd_status,
    "logs": cmd_logs,
    "list": cmd_list,
    "cancel": cmd_cancel,
    "wait": cmd_wait,
    "yaml": cmd_yaml,
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
