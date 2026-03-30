"""Iterare CLI — task and tool request inspection utility.

This is NOT the agent interface. Agent interaction happens through Claude Code.
This CLI provides visibility into task state, logs, and tool request queues.

Usage:
    iterare tasks               List all tasks
    iterare tasks <id>          Show task state and log
    iterare requests            List pending tool requests
    iterare requests approve <id>
    iterare requests reject <id> [reason]
"""

import sys
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

console = Console()


def cmd_tasks(args: list[str]) -> None:
    from iterare.utils.task import list_tasks, read_task
    from iterare.utils.log import summarize_log

    if args:
        task_id = args[0]
        state = read_task(task_id)
        if not state:
            console.print(f"[red]Task not found: {task_id}[/red]")
            return
        import yaml
        console.print(Syntax(yaml.dump(state, sort_keys=False), "yaml"))
        console.print()
        console.print(summarize_log(task_id))
        return

    tasks = list_tasks()
    if not tasks:
        console.print("[dim]No tasks found.[/dim]")
        return

    table = Table(show_header=True)
    table.add_column("ID", style="dim")
    table.add_column("Status")
    table.add_column("Description")
    table.add_column("Created")

    status_style = {"active": "green", "complete": "blue", "failed": "red"}
    for t in tasks:
        style = status_style.get(t.get("status", ""), "white")
        table.add_row(
            t.get("task_id", "?"),
            f"[{style}]{t.get('status', '?')}[/{style}]",
            (t.get("description") or "")[:60],
            (t.get("created_at") or "")[:19],
        )
    console.print(table)


def cmd_requests(args: list[str]) -> None:
    from iterare.tools.tool_request import list_pending_requests, update_request_status
    import yaml

    if args and args[0] in ("approve", "reject"):
        action = args[0]
        if len(args) < 2:
            console.print(f"[red]Usage: iterare requests {action} <request_id>[/red]")
            return
        request_id = args[1]
        note = " ".join(args[2:]) if len(args) > 2 else ""
        status = "approved" if action == "approve" else "rejected"
        result = update_request_status(request_id, status, note)
        console.print(result)
        return

    pending = list_pending_requests()
    if not pending:
        console.print("[dim]No pending tool requests.[/dim]")
        return

    for req in pending:
        console.print(f"\n[bold]{req['request_id']}[/bold] — {req['name']}")
        console.print(f"  Purpose: {req['purpose']}")
        console.print(f"  Scope:   {req['scope']}")
        console.print(f"  From:    {req['requester']} (task: {req['task_context']})")
        console.print(f"  Gap:     {req['why_existing_insufficient']}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        console.print(__doc__)
        return

    cmd = args[0]
    rest = args[1:]

    if cmd == "tasks":
        cmd_tasks(rest)
    elif cmd == "requests":
        cmd_requests(rest)
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        console.print(__doc__)


if __name__ == "__main__":
    main()
