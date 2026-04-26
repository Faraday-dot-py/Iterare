"""Task archival lifecycle.

A completed task is not done until it has been archived:
  - final manifest (metrics, artifacts, status)
  - sealed event log
  - provenance bundle
  - last checkpoint

Archive location: archive/<task_id>/
"""

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[4])).resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(
    task_id: str,
    final_status: str,
    summary: str,
    metrics: dict | None = None,
    artifacts: list[dict] | None = None,
) -> Path:
    """Write the archive manifest for a completed task.

    Call this before archive_task() — it sets the permanent record.
    artifacts: list of {"path": ..., "description": ...}
    """
    manifest = {
        "task_id": task_id,
        "archived_at": _now(),
        "final_status": final_status,
        "summary": summary,
        "metrics": metrics or {},
        "artifacts": artifacts or [],
    }
    archive_dir = _ROOT / "archive" / task_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_dir / "manifest.yaml"
    manifest_path.write_text(yaml.dump(manifest, sort_keys=False))
    return manifest_path


def archive_task(task_id: str, keep_task_tree: bool = True) -> Path:
    """Copy provenance, events, checkpoints, and run.yaml to the archive.

    Does not delete the task tree by default (keep_task_tree=True).
    """
    task_dir = _ROOT / "tasks" / task_id
    archive_dir = _ROOT / "archive" / task_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    to_copy = ["run.yaml", "events.jsonl", "provenance.jsonl", "resume.md"]
    for fname in to_copy:
        src = task_dir / fname
        if src.exists():
            shutil.copy2(src, archive_dir / fname)

    cp_src = task_dir / "checkpoints"
    if cp_src.exists():
        cp_dst = archive_dir / "checkpoints"
        if cp_dst.exists():
            shutil.rmtree(cp_dst)
        shutil.copytree(cp_src, cp_dst)

    return archive_dir


def read_manifest(task_id: str) -> dict | None:
    path = _ROOT / "archive" / task_id / "manifest.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def list_archives() -> list[dict]:
    archive_root = _ROOT / "archive"
    if not archive_root.exists():
        return []
    results = []
    for d in sorted(archive_root.iterdir()):
        if d.is_dir():
            manifest_path = d / "manifest.yaml"
            if manifest_path.exists():
                results.append(yaml.safe_load(manifest_path.read_text()))
            else:
                results.append({"task_id": d.name, "final_status": "unknown"})
    return results
