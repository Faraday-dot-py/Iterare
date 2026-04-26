"""Append-only event log for runs.

Events are written at decision and side-effect boundaries only —
not every model turn. This is the durable audit trail alongside
checkpoints.

Event types:
  plan_created       — orchestrator produced a plan
  delegation_issued  — orchestrator handed off to an agent
  tool_result        — significant tool call result accepted
  approval_requested — paused for human review
  approval_resolved  — approval granted or rejected
  artifact_written   — file or artifact produced
  evidence_updated   — evidence table or README updated
  stop_triggered     — a stop rule fired
  run_complete       — run closed successfully
  run_failed         — run closed with failure

Log location: tasks/<task_id>/events.jsonl
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[4])).resolve()

_VALID_EVENTS = {
    "plan_created",
    "delegation_issued",
    "tool_result",
    "approval_requested",
    "approval_resolved",
    "artifact_written",
    "evidence_updated",
    "stop_triggered",
    "run_complete",
    "run_failed",
}


def log_event(
    task_id: str,
    agent: str,
    event: str,
    data: dict | None = None,
    run_id: str | None = None,
) -> None:
    """Append a structured event to the task's event log."""
    if event not in _VALID_EVENTS:
        raise ValueError(f"Unknown event type: {event!r}. Valid: {sorted(_VALID_EVENTS)}")

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "agent": agent,
        "event": event,
        "data": data or {},
    }
    if run_id:
        entry["run_id"] = run_id

    log_path = _ROOT / "tasks" / task_id / "events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_events(task_id: str) -> list[dict]:
    """Return all events for a task."""
    path = _ROOT / "tasks" / task_id / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_events(task_id: str) -> str:
    entries = read_events(task_id)
    if not entries:
        return f"No events for task {task_id}"
    lines = [f"Task {task_id} — {len(entries)} events\n"]
    for e in entries:
        ts = e["ts"][:19]
        agent = e.get("agent", "?")[:14]
        event = e.get("event", "?")
        data = e.get("data", {})
        detail = ", ".join(f"{k}={v}" for k, v in list(data.items())[:2])
        lines.append(f"  [{ts}] {agent:<14s} {event:<22s} {detail}")
    return "\n".join(lines)
