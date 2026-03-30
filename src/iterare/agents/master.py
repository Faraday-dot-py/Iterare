"""Master Agent node.

Plans high-level tasks, spawns Managers via task specs, monitors progress,
and is the sole writer to systems of record.
"""

import uuid
import json
from pathlib import Path
import os

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from iterare.state import IterareState, ManagerTask, WorkerTask, LogEntry
from iterare.agents.base import load_template, get_llm, build_system_message, make_log_entry


@tool
def create_manager_tasks(tasks: list[dict]) -> str:
    """
    Define the manager-level task breakdown for this research task.
    Each task must have: description (str), and optionally memory_policy (str).
    Call this once with the full list of manager tasks.
    tasks: list of dicts with keys: description, memory_policy (optional)
    """
    return json.dumps(tasks)


@tool
def write_system_record(path: str, content: str) -> str:
    """
    Write to a system-of-record markdown file. Masters are the only agents
    that may call this tool. Path must be relative to ITERARE_ROOT.
    path: relative path (e.g. 'tasks/task-001/README.md')
    content: markdown content to write
    """
    root = Path(os.getenv("ITERARE_ROOT", "."))
    target = (root / path).resolve()
    # Safety: must stay within ITERARE_ROOT
    if not str(target).startswith(str(root.resolve())):
        return f"ERROR: path {path} is outside ITERARE_ROOT"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Written: {path}"


_TOOLS = [create_manager_tasks, write_system_record]
_TOOL_MAP = {t.name: t for t in _TOOLS}


def master_node(state: IterareState) -> dict:
    """Master Agent: plans task and creates manager task list."""
    template = load_template("master")
    llm = get_llm().bind_tools(_TOOLS)
    system = build_system_message(template)

    # Build context message with task description
    task_msg = HumanMessage(content=(
        f"Task ID: {state['task_id']}\n"
        f"Task: {state['task_description']}\n\n"
        f"Break this into manager-level tasks by calling create_manager_tasks. "
        f"If all managers are complete, write summary to system of record and finish."
    ))

    # If we already have manager tasks, check completion status
    if state.get("manager_tasks"):
        completed = [m for m in state["manager_tasks"] if m["status"] == "complete"]
        pending = [m for m in state["manager_tasks"] if m["status"] != "complete"]

        if not pending:
            # All managers done — write summary and return to interface
            summary = _build_summary(state)
            path = f"tasks/{state['task_id']}/README.md"
            write_system_record.invoke({"path": path, "content": summary})
            return {
                "current_node": "end",
                "log": [make_log_entry("master", "result", f"Task {state['task_id']} complete. Summary written.")],
                "messages": [HumanMessage(content=f"[Master] Task complete. Summary at {path}")],
            }

        # Advance to the next pending manager
        next_idx = next(i for i, m in enumerate(state["manager_tasks"]) if m["status"] != "complete")
        return {
            "current_node": "manager",
            "current_manager_idx": next_idx,
            "worker_tasks": [],
            "current_worker_idx": 0,
            "log": [make_log_entry("master", "handoff", f"Delegating to manager {next_idx}: {state['manager_tasks'][next_idx]['description']}")],
        }

    # First call — plan the task
    messages = [system, task_msg]
    response = llm.invoke(messages)
    updates: dict = {
        "messages": [response],
        "log": [make_log_entry("master", "decision", "Planning task breakdown")],
    }

    if response.tool_calls:
        call = response.tool_calls[0]
        if call["name"] == "create_manager_tasks":
            raw_tasks = json.loads(call["args"]["tasks"]) if isinstance(call["args"]["tasks"], str) else call["args"]["tasks"]
            manager_tasks: list[ManagerTask] = [
                ManagerTask(
                    id=f"mgr-{i:03d}",
                    description=t["description"],
                    status="pending",
                    worker_tasks=[],
                    summary=None,
                )
                for i, t in enumerate(raw_tasks)
            ]
            updates["manager_tasks"] = manager_tasks
            updates["current_manager_idx"] = 0
            updates["worker_tasks"] = []
            updates["current_worker_idx"] = 0
            updates["current_node"] = "manager"
            updates["log"].append(make_log_entry("master", "handoff", f"Created {len(manager_tasks)} manager tasks"))
            updates["messages"].append(
                ToolMessage(content=json.dumps(raw_tasks), tool_call_id=call["id"])
            )
        elif call["name"] == "write_system_record":
            result = write_system_record.invoke(call["args"])
            updates["current_node"] = "end"
            updates["messages"].append(ToolMessage(content=result, tool_call_id=call["id"]))
    else:
        updates["current_node"] = "end"

    return updates


def _build_summary(state: IterareState) -> str:
    """Build the master-level README summary from completed manager tasks."""
    lines = [
        f"# Task: {state['task_description']}",
        f"**Task ID**: {state['task_id']}",
        "",
        "## Manager Summaries",
        "",
    ]
    for m in state.get("manager_tasks", []):
        lines.append(f"### {m['id']}: {m['description']}")
        lines.append(m.get("summary") or "_No summary provided._")
        lines.append("")
    return "\n".join(lines)
