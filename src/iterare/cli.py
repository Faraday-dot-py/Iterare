"""Iterare CLI — task, run, approval, and tool request inspection.

This is NOT the agent interface. Agent interaction happens through Claude Code.
This CLI provides visibility into task state, runs, approval queues, and tool requests.

Usage:
    iterare tasks                           List all tasks
    iterare tasks <id>                      Show task state and log
    iterare runs <task_id>                  Show run control doc + events
    iterare approvals                       List all pending approvals
    iterare approvals <task_id>             List approvals for a task
    iterare approvals approve <task_id> <approval_id> [note]
    iterare approvals reject  <task_id> <approval_id> [note]
    iterare requests                        List pending tool requests
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


def cmd_runs(args: list[str]) -> None:
    from iterare.utils.run import read_run
    from iterare.utils.events import summarize_events
    import yaml

    if not args:
        console.print("[red]Usage: iterare runs <task_id>[/red]")
        return

    task_id = args[0]
    run = read_run(task_id)
    if not run:
        console.print(f"[dim]No run.yaml found for task {task_id}[/dim]")
        return

    console.print(Syntax(yaml.dump(run, sort_keys=False), "yaml"))
    console.print()
    console.print(summarize_events(task_id))


def cmd_approvals(args: list[str]) -> None:
    from iterare.utils.approvals import list_all_pending, list_approvals, resolve_approval
    import yaml

    if args and args[0] in ("approve", "reject"):
        action = args[0]
        if len(args) < 3:
            console.print(f"[red]Usage: iterare approvals {action} <task_id> <approval_id> [note][/red]")
            return
        task_id = args[1]
        approval_id = args[2]
        note = " ".join(args[3:]) if len(args) > 3 else ""
        decision = "approved" if action == "approve" else "rejected"
        try:
            rec = resolve_approval(task_id, approval_id, decision, note)
            status_color = "green" if decision == "approved" else "red"
            console.print(f"[{status_color}]{decision.upper()}[/{status_color}]: {approval_id}")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
        return

    if args and args[0] not in ("approve", "reject"):
        task_id = args[0]
        pending = list_approvals(task_id, status="pending")
    else:
        pending = list_all_pending()

    if not pending:
        console.print("[dim]No pending approvals.[/dim]")
        return

    for rec in pending:
        tier_color = {"low": "dim", "medium": "yellow", "high": "red"}.get(rec.get("risk_tier", ""), "white")
        console.print(f"\n[bold]{rec['approval_id']}[/bold]  task={rec['task_id']}")
        console.print(f"  Action:    {rec['action']}")
        console.print(f"  Risk tier: [{tier_color}]{rec.get('risk_tier', '?')}[/{tier_color}]")
        console.print(f"  Agent:     {rec.get('agent', '?')}")
        console.print(f"  Created:   {(rec.get('created_at') or '')[:19]}")
        if rec.get("payload"):
            console.print(f"  Payload:   {yaml.dump(rec['payload'], default_flow_style=True).strip()[:120]}")


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
    elif cmd == "runs":
        cmd_runs(rest)
    elif cmd == "approvals":
        cmd_approvals(rest)
    elif cmd == "requests":
        cmd_requests(rest)
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        console.print(__doc__)


if __name__ == "__main__":
    main()
