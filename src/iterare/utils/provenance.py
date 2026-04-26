"""PROV-lite provenance records.

Ties claims and artifacts back to the agents, activities, and sources
that produced them. This is the lightweight version of W3C PROV —
enough to trace "who said what, from what source, with what confidence."

Schema (one record per line in provenance.jsonl):
  ts             — ISO timestamp
  task_id        — task this belongs to
  entity         — what was produced (e.g. "tasks/foo/README.md#sota-claim")
  activity       — what process produced it (e.g. "exp53_hotflip_analysis")
  agent          — which agent (e.g. "worker-1", "manager-44", "interface")
  source         — input artifact or URL (e.g. path to results JSON)
  evidence_grade — H | M | L  (matches README evidence levels)
  note           — optional free-text

Log location: tasks/<task_id>/provenance.jsonl
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[4])).resolve()
_VALID_GRADES = {"H", "M", "L"}


def record_provenance(
    task_id: str,
    entity: str,
    activity: str,
    agent: str,
    source: str,
    evidence_grade: str,
    note: str = "",
) -> None:
    """Append a provenance record.

    entity: a claim, artifact, or README section being attributed
    activity: the process that produced it (experiment name, analysis step, etc.)
    source: the input artifact — file path, URL, or "observation"
    evidence_grade: H, M, or L
    """
    if evidence_grade not in _VALID_GRADES:
        raise ValueError(f"evidence_grade must be H, M, or L — got {evidence_grade!r}")

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "entity": entity,
        "activity": activity,
        "agent": agent,
        "source": source,
        "evidence_grade": evidence_grade,
    }
    if note:
        entry["note"] = note

    path = _ROOT / "tasks" / task_id / "provenance.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_provenance(task_id: str) -> list[dict]:
    path = _ROOT / "tasks" / task_id / "provenance.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trace_entity(task_id: str, entity_pattern: str) -> list[dict]:
    """Return all provenance records whose entity contains the given pattern."""
    return [
        r for r in read_provenance(task_id)
        if entity_pattern in r.get("entity", "")
    ]
