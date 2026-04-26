"""Run lifecycle management.

A run is the execution context for a task (or significant subtask).
It provides the durable control plane: objective, budgets, stop rules,
checkpoint pointer, write scope, and open approvals.

Files created under tasks/<task_id>/:
  run.yaml          — canonical control document
  resume.md         — human-readable hydration packet for next session
  checkpoints/      — checkpoint snapshots
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[4])).resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_dir(task_id: str) -> Path:
    return _ROOT / "tasks" / task_id


def new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"run-{ts}-{uuid.uuid4().hex[:6]}"


def create_run(
    task_id: str,
    objective: str,
    budgets: dict | None = None,
    stop_rules: list[str] | None = None,
    write_scope: list[str] | None = None,
) -> dict:
    """Initialize a run control document for a task.

    Should be called at the start of any significant execution session.
    """
    run_id = new_run_id()
    run = {
        "run_id": run_id,
        "task_id": task_id,
        "objective": objective,
        "status": "active",
        "created_at": _now(),
        "updated_at": _now(),
        "budgets": budgets or {},
        "stop_rules": stop_rules or [],
        "write_scope": write_scope or [f"tasks/{task_id}/"],
        "active_delegates": [],
        "checkpoint": None,
        "open_approvals": [],
    }
    d = _run_dir(task_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.yaml").write_text(yaml.dump(run, sort_keys=False))
    return run


def read_run(task_id: str) -> dict | None:
    path = _run_dir(task_id) / "run.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def update_run(task_id: str, updates: dict) -> dict:
    run = read_run(task_id) or {}
    run.update(updates)
    run["updated_at"] = _now()
    (_run_dir(task_id) / "run.yaml").write_text(yaml.dump(run, sort_keys=False))
    return run


def checkpoint_run(task_id: str, state: dict, label: str = "") -> str:
    """Write a checkpoint snapshot and update the run's checkpoint pointer.

    Checkpoints happen at decision and side-effect boundaries:
    plan creation, delegation issuance, tool result acceptance,
    approval waits, artifact writes, evidence updates.
    """
    run = read_run(task_id) or {}
    run_id = run.get("run_id", "unknown")

    cp_dir = _run_dir(task_id) / "checkpoints"
    cp_dir.mkdir(exist_ok=True)

    existing = sorted(cp_dir.glob("cp-*.yaml"))
    n = len(existing) + 1
    cp_id = f"cp-{n:03d}"
    cp_path = cp_dir / f"{cp_id}.yaml"

    checkpoint = {
        "checkpoint_id": cp_id,
        "run_id": run_id,
        "task_id": task_id,
        "created_at": _now(),
        "label": label,
        "state": state,
    }
    cp_path.write_text(yaml.dump(checkpoint, sort_keys=False))

    update_run(task_id, {"checkpoint": f"checkpoints/{cp_id}.yaml"})
    return cp_id


def close_run(task_id: str, status: str, summary: str) -> dict:
    """Mark a run complete or failed and write the resume.md hydration packet."""
    run = update_run(task_id, {"status": status})

    resume_lines = [
        f"# Resume: {task_id}",
        f"",
        f"**Run:** {run.get('run_id')}  |  **Status:** {status}",
        f"**Closed:** {_now()}",
        f"",
        f"## Objective",
        f"",
        run.get("objective", ""),
        f"",
        f"## Summary",
        f"",
        summary,
        f"",
        f"## Last Checkpoint",
        f"",
        f"`{run.get('checkpoint') or 'none'}`",
        f"",
        f"## Stop Rules",
        f"",
    ]
    for rule in run.get("stop_rules", []):
        resume_lines.append(f"- {rule}")
    if not run.get("stop_rules"):
        resume_lines.append("*(none defined)*")
    resume_lines += ["", "## Write Scope", ""]
    for p in run.get("write_scope", []):
        resume_lines.append(f"- `{p}`")

    (_run_dir(task_id) / "resume.md").write_text("\n".join(resume_lines) + "\n")
    return run


def write_resume(task_id: str, content: str) -> None:
    """Write a free-form resume.md — for orchestrators that want full control."""
    (_run_dir(task_id) / "resume.md").write_text(content)
