"""UserPromptSubmit hook — injects active run context into each prompt.

Outputs JSON with additionalContext when there are active runs or pending
approvals. Outputs nothing when the project is idle, so there's no noise.

Called by Claude Code on every user prompt via settings.local.json hook.
"""

import json
import os
import sys
from pathlib import Path

_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[4])).resolve()


def _load_active_runs() -> list[dict]:
    tasks_root = _ROOT / "tasks"
    if not tasks_root.exists():
        return []
    try:
        import yaml
    except ImportError:
        return []
    active = []
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir():
            continue
        run_yaml = task_dir / "run.yaml"
        if not run_yaml.exists():
            continue
        try:
            run = yaml.safe_load(run_yaml.read_text())
            if run.get("status") == "active":
                resume_path = task_dir / "resume.md"
                run["_resume"] = resume_path.read_text() if resume_path.exists() else None
                active.append(run)
        except Exception:
            pass
    return active


def _load_pending_approvals() -> list[dict]:
    tasks_root = _ROOT / "tasks"
    if not tasks_root.exists():
        return []
    try:
        import yaml
    except ImportError:
        return []
    pending = []
    for task_dir in sorted(tasks_root.iterdir()):
        approvals_dir = task_dir / "approvals"
        if not approvals_dir.exists():
            continue
        for f in sorted(approvals_dir.glob("apr-*.yaml")):
            try:
                rec = yaml.safe_load(f.read_text())
                if rec.get("status") == "pending":
                    pending.append(rec)
            except Exception:
                pass
    return pending


def main() -> None:
    active_runs = _load_active_runs()
    pending_approvals = _load_pending_approvals()

    if not active_runs and not pending_approvals:
        return

    parts = ["## Iterare Session Context\n"]

    for run in active_runs:
        parts.append(f"### Active Run: {run['task_id']}")
        parts.append(f"Run ID: `{run['run_id']}` | Status: {run['status']}")
        parts.append(f"Objective: {run.get('objective', '(none)')}")
        if run.get("stop_rules"):
            parts.append("Stop rules: " + "; ".join(run["stop_rules"]))
        if run.get("checkpoint"):
            parts.append(f"Last checkpoint: `{run['checkpoint']}`")
        parts.append("")
        if run.get("_resume"):
            parts.append(run["_resume"].strip())
            parts.append("")

    if pending_approvals:
        tier_order = {"high": 0, "medium": 1, "low": 2}
        pending_approvals.sort(key=lambda r: tier_order.get(r.get("risk_tier", "low"), 9))
        parts.append(f"### {len(pending_approvals)} Pending Approval(s)")
        for apr in pending_approvals:
            tier = apr.get("risk_tier", "?").upper()
            parts.append(f"- [{tier}] `{apr['approval_id']}`: {apr['action']} (task={apr['task_id']})")
        parts.append("\nResolve with: `iterare approvals approve <task_id> <approval_id>`")
        parts.append("")

    context = "\n".join(parts)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
