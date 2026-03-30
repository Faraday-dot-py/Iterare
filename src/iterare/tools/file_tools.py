"""File operation tools for Iterare agents.

Path enforcement:
- read_file: any path within ITERARE_ROOT
- write_task_file: tasks/ directory only (Workers and Managers)
- list_directory: any path within ITERARE_ROOT
"""

import os
from pathlib import Path
from langchain_core.tools import tool

_ROOT = Path(os.getenv("ITERARE_ROOT", ".")).resolve()


def _safe_path(relative_path: str) -> Path:
    """Resolve a relative path and verify it stays within ITERARE_ROOT."""
    resolved = (_ROOT / relative_path).resolve()
    if not str(resolved).startswith(str(_ROOT)):
        raise PermissionError(f"Path '{relative_path}' is outside ITERARE_ROOT.")
    return resolved


@tool
def read_file(path: str, **kwargs) -> str:
    """
    Read a file from within the Iterare workspace.
    path: relative path from ITERARE_ROOT (e.g. 'docs/structure.md')
    Returns file contents as a string.
    """
    target = _safe_path(path)
    if not target.exists():
        return f"File not found: {path}"
    return target.read_text()


@tool
def write_task_file(path: str, content: str, **kwargs) -> str:
    """
    Write a file within the tasks/ directory.
    path: relative path from ITERARE_ROOT — must start with 'tasks/'
    content: text content to write
    Workers use this to write their README and intermediate outputs.
    """
    if not path.startswith("tasks/"):
        return "ERROR: write_task_file may only write to tasks/ directory."
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Written: {path}"


@tool
def list_directory(path: str = "", **kwargs) -> str:
    """
    List contents of a directory within the Iterare workspace.
    path: relative path from ITERARE_ROOT (empty string = root)
    Returns newline-separated list of entries.
    """
    target = _safe_path(path) if path else _ROOT
    if not target.is_dir():
        return f"Not a directory: {path}"
    entries = sorted(target.iterdir())
    return "\n".join(
        f"{'[dir] ' if e.is_dir() else '[file]'} {e.name}"
        for e in entries
    )
