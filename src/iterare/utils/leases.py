"""Write lease management.

A write lease is an explicit, file-backed record that authorizes an
agent to write to specific paths. Workers can only write inside their
leased sandbox; anything outside that scope must be proposed as a patch
or promotion request.

Lease files: tasks/<task_id>/leases/<agent_id>.yaml

Enforcement is currently by convention and audit, not hard lock.
The machinery exists so violations are detectable.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[4])).resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _leases_dir(task_id: str) -> Path:
    d = _ROOT / "tasks" / task_id / "leases"
    d.mkdir(parents=True, exist_ok=True)
    return d


def acquire_lease(
    task_id: str,
    agent_id: str,
    allowed_paths: list[str],
    run_id: str | None = None,
) -> dict:
    """Record that agent_id is authorized to write to allowed_paths."""
    lease_id = f"lease-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    lease = {
        "lease_id": lease_id,
        "task_id": task_id,
        "run_id": run_id,
        "agent_id": agent_id,
        "allowed_paths": allowed_paths,
        "acquired_at": _now(),
        "released_at": None,
    }
    path = _leases_dir(task_id) / f"{agent_id}.yaml"
    path.write_text(yaml.dump(lease, sort_keys=False))
    return lease


def release_lease(task_id: str, agent_id: str) -> None:
    path = _leases_dir(task_id) / f"{agent_id}.yaml"
    if not path.exists():
        return
    lease = yaml.safe_load(path.read_text())
    lease["released_at"] = _now()
    path.write_text(yaml.dump(lease, sort_keys=False))


def check_lease(task_id: str, agent_id: str, target_path: str) -> bool:
    """Return True if agent_id has an active lease covering target_path."""
    path = _leases_dir(task_id) / f"{agent_id}.yaml"
    if not path.exists():
        return False
    lease = yaml.safe_load(path.read_text())
    if lease.get("released_at") is not None:
        return False
    # normalize to relative path for comparison
    target = str(target_path).lstrip("/")
    return any(target.startswith(p.lstrip("/")) for p in lease.get("allowed_paths", []))


def read_lease(task_id: str, agent_id: str) -> dict | None:
    path = _leases_dir(task_id) / f"{agent_id}.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def list_leases(task_id: str, active_only: bool = True) -> list[dict]:
    d = _leases_dir(task_id)
    results = []
    for f in sorted(d.glob("*.yaml")):
        lease = yaml.safe_load(f.read_text())
        if active_only and lease.get("released_at") is not None:
            continue
        results.append(lease)
    return results
