"""Task state management.

Tasks are tracked as YAML files in tasks/<task_id>/state.yaml
This is the lightweight alternative to LangGraph checkpointing —
state lives on disk and is readable by any agent or human.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[4])).resolve()


def new_task_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"task-{ts}-{uuid.uuid4().hex[:6]}"


def task_dir(task_id: str) -> Path:
    return _ROOT / "tasks" / task_id


def create_task(description: str) -> dict:
    """Initialize a new task and write its state file."""
    task_id = new_task_id()
    state = {
        "task_id": task_id,
        "description": description,
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manager_tasks": [],
    }
    d = task_dir(task_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.yaml").write_text(yaml.dump(state, sort_keys=False))
    return state


def read_task(task_id: str) -> dict | None:
    path = task_dir(task_id) / "state.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def update_task(task_id: str, updates: dict) -> dict:
    state = read_task(task_id) or {}
    state.update(updates)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = task_dir(task_id) / "state.yaml"
    path.write_text(yaml.dump(state, sort_keys=False))
    return state


def list_tasks(status: str | None = None) -> list[dict]:
    """List all tasks, optionally filtered by status."""
    tasks_root = _ROOT / "tasks"
    if not tasks_root.exists():
        return []
    tasks = []
    for d in sorted(tasks_root.iterdir()):
        state_file = d / "state.yaml"
        if state_file.exists():
            state = yaml.safe_load(state_file.read_text())
            if status is None or state.get("status") == status:
                tasks.append(state)
    return tasks
