"""Tool request queue mechanism.

Agents write YAML request files to tools/requests/.
The Master polls this directory and approves, rejects, or assigns builds.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from langchain_core.tools import tool

_ROOT = Path(os.getenv("ITERARE_ROOT", ".")).resolve()
_REQUESTS_DIR = _ROOT / "tools" / "requests"


@tool
def submit_tool_request(
    name: str,
    purpose: str,
    why_existing_insufficient: str,
    inputs: str,
    outputs: str,
    scope: str,
    **kwargs,
) -> str:
    """
    Submit a request to build or approve a new tool.
    The Master will review all pending requests in tools/requests/.

    name: proposed tool name
    purpose: what it does (one sentence)
    why_existing_insufficient: specific gap not covered by existing tools
    inputs: what data the tool takes
    outputs: what the tool returns
    scope: narrow | moderate | broad
    """
    if scope not in ("narrow", "moderate", "broad"):
        return "ERROR: scope must be narrow | moderate | broad"

    requester = kwargs.get("_worker_id") or kwargs.get("_agent_id") or "unknown"
    task_context = kwargs.get("_task_id") or "unknown"
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

    return f"Tool request submitted: {request_id} ({out_path.name})"
