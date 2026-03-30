"""Tool request queue.

Agents submit YAML request files to tools/requests/.
The Master (Interface Agent acting as Master) reviews the queue and approves,
rejects, or assigns builds.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[4])).resolve()
_REQUESTS_DIR = _ROOT / "tools" / "requests"


def submit_tool_request(
    name: str,
    purpose: str,
    why_existing_insufficient: str,
    inputs: str,
    outputs: str,
    scope: str,
    requester: str = "unknown",
    task_context: str = "unknown",
) -> str:
    """
    Submit a request to build or approve a new tool.
    scope: narrow | moderate | broad
    Returns the request ID.
    """
    if scope not in ("narrow", "moderate", "broad"):
        return "ERROR: scope must be narrow | moderate | broad"

    request_id = f"req-{uuid.uuid4().hex[:8]}"
    payload = {
        "request_id": request_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "name": name,
        "purpose": purpose,
        "why_existing_insufficient": why_existing_insufficient,
        "inputs": inputs,
        "outputs": outputs,
        "scope": scope,
        "requester": requester,
        "task_context": task_context,
    }

    _REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REQUESTS_DIR / f"{request_id}.yaml"
    out_path.write_text(yaml.dump(payload, sort_keys=False))
    return f"Tool request submitted: {request_id}"


def list_pending_requests() -> list[dict]:
    """Return all pending tool requests from the queue."""
    if not _REQUESTS_DIR.exists():
        return []
    requests = []
    for f in sorted(_REQUESTS_DIR.glob("*.yaml")):
        data = yaml.safe_load(f.read_text())
        if data.get("status") == "pending":
            requests.append(data)
    return requests


def update_request_status(request_id: str, status: str, note: str = "") -> str:
    """Update a tool request status (approved | rejected | building | complete)."""
    path = _REQUESTS_DIR / f"{request_id}.yaml"
    if not path.exists():
        return f"Request not found: {request_id}"
    data = yaml.safe_load(path.read_text())
    data["status"] = status
    if note:
        data["note"] = note
    path.write_text(yaml.dump(data, sort_keys=False))
    return f"Updated {request_id} → {status}"
