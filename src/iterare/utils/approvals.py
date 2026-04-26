"""Approval queue for risk-tiered human review.

Risk tiers:
  low     — auto-proceed inside write scope (no pause needed)
  medium  — pause for human review; default for system-of-record writes
             and tool promotions
  high    — pause + explicit confirmation; for destructive, external-impact,
             financial, or security-sensitive actions

Approval files live at: tasks/<task_id>/approvals/<approval_id>.yaml

The approval queue is checked via `iterare approvals` CLI.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[4])).resolve()
_VALID_TIERS = {"low", "medium", "high"}
_VALID_STATUSES = {"pending", "approved", "rejected"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _approvals_dir(task_id: str) -> Path:
    d = _ROOT / "tasks" / task_id / "approvals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def submit_approval(
    task_id: str,
    action: str,
    risk_tier: str,
    agent: str,
    payload: dict | None = None,
    run_id: str | None = None,
) -> dict:
    """Submit an action for human review.

    Low-tier actions inside write scope don't need approval — call this
    only when the action warrants a pause.
    """
    if risk_tier not in _VALID_TIERS:
        raise ValueError(f"risk_tier must be one of {_VALID_TIERS}")

    approval_id = f"apr-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    record = {
        "approval_id": approval_id,
        "task_id": task_id,
        "run_id": run_id,
        "action": action,
        "risk_tier": risk_tier,
        "agent": agent,
        "status": "pending",
        "created_at": _now(),
        "resolved_at": None,
        "resolution_note": None,
        "payload": payload or {},
    }
    path = _approvals_dir(task_id) / f"{approval_id}.yaml"
    path.write_text(yaml.dump(record, sort_keys=False))
    return record


def list_approvals(task_id: str, status: str | None = None) -> list[dict]:
    d = _approvals_dir(task_id)
    results = []
    for f in sorted(d.glob("apr-*.yaml")):
        rec = yaml.safe_load(f.read_text())
        if status is None or rec.get("status") == status:
            results.append(rec)
    return results


def list_all_pending() -> list[dict]:
    """Return all pending approvals across all tasks."""
    tasks_root = _ROOT / "tasks"
    if not tasks_root.exists():
        return []
    results = []
    for task_dir in sorted(tasks_root.iterdir()):
        approvals_dir = task_dir / "approvals"
        if not approvals_dir.exists():
            continue
        for f in sorted(approvals_dir.glob("apr-*.yaml")):
            rec = yaml.safe_load(f.read_text())
            if rec.get("status") == "pending":
                results.append(rec)
    return results


def resolve_approval(
    task_id: str,
    approval_id: str,
    decision: str,
    note: str = "",
) -> dict:
    """Approve or reject a pending approval request."""
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be 'approved' or 'rejected'")

    path = _approvals_dir(task_id) / f"{approval_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Approval not found: {approval_id}")

    rec = yaml.safe_load(path.read_text())
    rec["status"] = decision
    rec["resolved_at"] = _now()
    rec["resolution_note"] = note
    path.write_text(yaml.dump(rec, sort_keys=False))
    return rec


def get_approval(task_id: str, approval_id: str) -> dict | None:
    path = _approvals_dir(task_id) / f"{approval_id}.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())
