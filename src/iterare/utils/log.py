"""Structured log writer.

Writes JSON log entries at decision/handoff points to logs/<task_id>.jsonl
Logs are sealed (read-only) when a task completes.
"""

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[4])).resolve()
_LOGS_DIR = _ROOT / "logs"


def _log_path(task_id: str) -> Path:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return _LOGS_DIR / f"{task_id}.jsonl"


def write_log(task_id: str, agent: str, event: str, detail: str) -> None:
    """
    Append a structured log entry.
    event: decision | handoff | failure | result | approval | escalation
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task": task_id,
        "agent": agent,
        "event": event,
        "detail": detail,
    }
    path = _log_path(task_id)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def seal_log(task_id: str) -> str:
    """Mark a task log as complete by making it read-only."""
    path = _log_path(task_id)
    if not path.exists():
        return f"No log found for task {task_id}"
    path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return f"Log sealed: {path.name}"


def read_log(task_id: str) -> list[dict]:
    """Return all log entries for a task as a list of dicts."""
    path = _log_path(task_id)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_log(task_id: str) -> str:
    """Return a human-readable summary of a task log."""
    entries = read_log(task_id)
    if not entries:
        return f"No log entries for task {task_id}"
    lines = [f"Task: {task_id} — {len(entries)} log entries\n"]
    for e in entries:
        lines.append(f"  [{e['ts']}] {e['agent']:12s} {e['event']:12s} {e['detail'][:80]}")
    return "\n".join(lines)
