"""Manager Agent node.

Receives a manager task from the Master, breaks it into Worker tasks,
monitors Workers, handles failures, and writes its summary README on close.
"""

import json
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from iterare.state import IterareState, ManagerTask, WorkerTask
from iterare.agents.base import load_template, get_llm, build_system_message, make_log_entry


@tool
def create_worker_tasks(tasks: list[dict]) -> str:
    """
    Define the worker-level steps for the current manager task.
    Each task must have: description (str).
    Call this once with the full ordered list of worker steps.
    tasks: list of dicts with key: description
    """
    return json.dumps(tasks)


@tool
def request_system_write(path: str, content: str) -> str:
    """
    Request that the Master write to a system-of-record file.
    The Master will decide whether to approve and execute the write.
    path: relative path from ITERARE_ROOT
    content: markdown content
    """
    # Logged as a request; actual write happens in master_node
    return f"WRITE_REQUEST:{path}:{content}"


_TOOLS = [create_worker_tasks, request_system_write]
_TOOL_MAP = {t.name: t for t in _TOOLS}


def manager_node(state: IterareState) -> dict:
    """Manager Agent: breaks down its task and coordinates workers."""
    template = load_template("manager")
    llm = get_llm().bind_tools(_TOOLS)
    system = build_system_message(template)

    idx = state["current_manager_idx"]
    current_task = state["manager_tasks"][idx]

    # If we already have worker tasks, check completion
    if state.get("worker_tasks"):
        pending = [w for w in state["worker_tasks"] if w["status"] not in ("complete", "escalated")]
        failed = [w for w in state["worker_tasks"] if w["status"] == "failed"]

        if not pending:
            # All workers done — write README and mark manager complete
            summary = _build_manager_readme(state, current_task)
            readme_path = (
                Path(os.getenv("ITERARE_ROOT", "."))
                / "tasks"
                / state["task_id"]
                / f"manager-{idx:03d}"
                / "README.md"
            )
            readme_path.parent.mkdir(parents=True, exist_ok=True)
            readme_path.write_text(summary)

            # Update manager task status in the list
            updated_managers = list(state["manager_tasks"])
            updated_managers[idx] = {**current_task, "status": "complete", "summary": summary}

            return {
                "manager_tasks": updated_managers,
                "current_node": "master",
                "log": [make_log_entry("manager", "result", f"Manager {idx} complete. README written.")],
            }

        # Advance to next pending worker
        next_worker_idx = next(
            i for i, w in enumerate(state["worker_tasks"]) if w["status"] not in ("complete", "escalated")
        )
        return {
            "current_node": "worker",
            "current_worker_idx": next_worker_idx,
            "log": [make_log_entry("manager", "handoff", f"Delegating worker step {next_worker_idx}: {state['worker_tasks'][next_worker_idx]['description']}")],
        }

    # First call for this manager — plan worker tasks
    task_msg = HumanMessage(content=(
        f"Manager Task: {current_task['description']}\n"
        f"Task ID: {state['task_id']}\n\n"
        f"Break this into atomic, single-step Worker tasks by calling create_worker_tasks."
    ))

    messages = [system, task_msg]
    response = llm.invoke(messages)
    updates: dict = {
        "messages": [response],
        "log": [make_log_entry("manager", "decision", f"Planning worker tasks for: {current_task['description']}")],
    }

    if response.tool_calls:
        call = response.tool_calls[0]
        if call["name"] == "create_worker_tasks":
            raw = json.loads(call["args"]["tasks"]) if isinstance(call["args"]["tasks"], str) else call["args"]["tasks"]
            worker_tasks: list[WorkerTask] = [
                WorkerTask(
                    id=f"wrk-{i:03d}",
                    description=t["description"],
                    status="pending",
                    attempts=0,
                    result=None,
                    error=None,
                )
                for i, t in enumerate(raw)
            ]
            updates["worker_tasks"] = worker_tasks
            updates["current_worker_idx"] = 0
            updates["current_node"] = "worker"
            updates["log"].append(make_log_entry("manager", "handoff", f"Created {len(worker_tasks)} worker tasks"))
            updates["messages"].append(ToolMessage(content=json.dumps(raw), tool_call_id=call["id"]))
    else:
        updates["current_node"] = "master"  # Nothing to delegate; return up

    return updates


def manager_triage_node(state: IterareState) -> dict:
    """
    Manager decides what to do with a worker that has exhausted retries.
    Options: mark escalated and skip, or retry with different framing.
    """
    idx = state["current_worker_idx"]
    worker = state["worker_tasks"][idx]

    # For now: mark as escalated, log it, and move on
    updated_workers = list(state["worker_tasks"])
    updated_workers[idx] = {**worker, "status": "escalated", "error": f"Escalated after {worker['attempts']} attempts"}

    return {
        "worker_tasks": updated_workers,
        "current_node": "manager",
        "log": [make_log_entry("manager", "decision", f"Worker {worker['id']} escalated after {worker['attempts']} failed attempts. Moving on.")],
    }


def _build_manager_readme(state: IterareState, task: ManagerTask) -> str:
    lines = [
        f"# Manager Task: {task['description']}",
        f"**Task ID**: {state['task_id']} | **Manager ID**: {task['id']}",
        "",
        "## Worker Results",
        "",
    ]
    for w in state.get("worker_tasks", []):
        status_badge = {"complete": "✓", "escalated": "⚠", "failed": "✗"}.get(w["status"], "?")
        lines.append(f"### [{status_badge}] {w['id']}: {w['description']}")
        if w.get("result"):
            lines.append(f"**Result**: {w['result']}")
        if w.get("error"):
            lines.append(f"**Error**: {w['error']}")
        lines.append("")
    return "\n".join(lines)
