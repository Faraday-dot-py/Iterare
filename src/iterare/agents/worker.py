"""Worker Agent node.

Executes a single concrete step. On failure: logs, notifies manager, attempts
self-fix. Escalates if unable to resolve. Writes its own task README.
"""

import os
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

from iterare.state import IterareState, WorkerTask
from iterare.agents.base import load_template, get_llm, build_system_message, make_log_entry
from iterare.tools.file_tools import read_file, write_task_file, list_directory
from iterare.tools.tool_request import submit_tool_request

_TOOLS = [read_file, write_task_file, list_directory, submit_tool_request]
_TOOL_MAP = {t.name: t for t in _TOOLS}


def worker_node(state: IterareState) -> dict:
    """Worker Agent: executes one step and reports result or failure."""
    template = load_template("worker")
    llm = get_llm().bind_tools(_TOOLS)
    system = build_system_message(template)

    idx = state["current_worker_idx"]
    worker = state["worker_tasks"][idx]
    max_attempts = state.get("max_attempts", 3)

    # Check if we've already hit max attempts
    if worker["attempts"] >= max_attempts:
        return {
            "current_node": "manager_triage",
            "log": [make_log_entry("worker", "failure", f"Worker {worker['id']} hit max attempts ({max_attempts}). Escalating.")],
        }

    task_msg = HumanMessage(content=(
        f"Worker Task: {worker['description']}\n"
        f"Task ID: {state['task_id']} | Worker ID: {worker['id']}\n"
        f"Attempt: {worker['attempts'] + 1} of {max_attempts}\n\n"
        f"Execute this step. Use available tools as needed. "
        f"When complete, call write_task_file to save your README with your reasoning and results."
    ))

    if worker.get("error"):
        task_msg = HumanMessage(content=(
            task_msg.content +
            f"\n\nPrevious attempt failed with: {worker['error']}\nTry a different approach."
        ))

    messages = [system, task_msg]
    response = llm.invoke(messages)

    updates: dict = {
        "messages": [response],
        "log": [make_log_entry("worker", "decision", f"Worker {worker['id']} attempt {worker['attempts'] + 1}")],
    }

    # Execute any tool calls
    result_text = None
    error_text = None
    tool_messages = []

    if response.tool_calls:
        for call in response.tool_calls:
            tool_fn = _TOOL_MAP.get(call["name"])
            if tool_fn:
                try:
                    tool_result = tool_fn.invoke({**call["args"], "_task_id": state["task_id"], "_worker_id": worker["id"]})
                    tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))
                    result_text = str(tool_result)
                except Exception as e:
                    error_text = str(e)
                    tool_messages.append(ToolMessage(content=f"ERROR: {e}", tool_call_id=call["id"]))
            else:
                tool_messages.append(ToolMessage(content=f"Unknown tool: {call['name']}", tool_call_id=call["id"]))

    updates["messages"].extend(tool_messages)

    # Update worker state
    updated_workers = list(state["worker_tasks"])
    current_worker = dict(worker)
    current_worker["attempts"] = worker["attempts"] + 1

    if error_text:
        current_worker["status"] = "failed"
        current_worker["error"] = error_text
        updates["log"].append(make_log_entry("worker", "failure", f"Worker {worker['id']} failed: {error_text}"))
        # Route back to self for retry (manager_node will check attempts)
        updates["current_node"] = "worker"
    else:
        current_worker["status"] = "complete"
        current_worker["result"] = result_text or str(response.content)
        current_worker["error"] = None
        updates["log"].append(make_log_entry("worker", "result", f"Worker {worker['id']} complete"))
        updates["current_node"] = "manager"

    updated_workers[idx] = current_worker
    updates["worker_tasks"] = updated_workers

    return updates
