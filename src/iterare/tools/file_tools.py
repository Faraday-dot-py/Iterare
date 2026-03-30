"""File operation utilities for Iterare agents.

Path enforcement:
- read_file: any path within ITERARE_ROOT
- write_task_file: tasks/ directory only (Workers and Managers)
- write_system_record: any path within ITERARE_ROOT (Masters only — enforced by convention)
- list_directory: any path within ITERARE_ROOT
"""

import os
from pathlib import Path

_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[4])).resolve()


def _safe_path(relative_path: str) -> Path:
    """Resolve a relative path and verify it stays within ITERARE_ROOT."""
    resolved = (_ROOT / relative_path).resolve()
    if not str(resolved).startswith(str(_ROOT)):
        raise PermissionError(f"Path '{relative_path}' is outside ITERARE_ROOT.")
    return resolved


def read_file(path: str) -> str:
    """Read a file from within the Iterare workspace."""
    target = _safe_path(path)
    if not target.exists():
        return f"File not found: {path}"
    return target.read_text()


def write_task_file(path: str, content: str) -> str:
    """Write a file to tasks/ — for Workers and Managers."""
    if not path.startswith("tasks/"):
        return "ERROR: write_task_file may only write to the tasks/ directory."
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Written: {path}"


def write_system_record(path: str, content: str) -> str:
    """Write to any path within ITERARE_ROOT. Masters only — enforced by convention."""
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Written: {path}"


def list_directory(path: str = "") -> str:
    """List contents of a directory within the Iterare workspace."""
    target = _safe_path(path) if path else _ROOT
    if not target.is_dir():
        return f"Not a directory: {path}"
    entries = sorted(target.iterdir())
    return "\n".join(
        f"{'[dir] ' if e.is_dir() else '[file]'} {e.name}"
        for e in entries
    )
