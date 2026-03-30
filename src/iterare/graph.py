"""Iterare LangGraph graph definition.

Hierarchy: Interface → (approval) → Master → Manager → Worker
                                                 ↑           |
                                         manager_triage ←────┘ (on max failures)
"""

import os
import uuid
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command

from iterare.state import IterareState
from iterare.agents.interface import interface_node
from iterare.agents.master import master_node
from iterare.agents.manager import manager_node, manager_triage_node
from iterare.agents.worker import worker_node
from iterare.agents.base import make_log_entry


# ── Approval gate ────────────────────────────────────────────────────────────

def approval_node(state: IterareState) -> dict:
    """
    Pause execution and present the Interface Agent's spawn proposal to the
    lead developer. Resume with {"approved": True/False}.
    """
    response = interrupt({
        "type": "spawn_approval",
        "proposal": state["proposal"],
    })
    approved = bool(response.get("approved", False))
    updates = {
        "approval_granted": approved,
        "log": [make_log_entry("approval", "decision", f"Lead developer {'approved' if approved else 'rejected'} spawn proposal")],
    }
    if approved:
        task_id = f"task-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        updates["task_id"] = task_id
        updates["task_description"] = _extract_task(state)
        updates["manager_tasks"] = []
        updates["worker_tasks"] = []
        updates["current_manager_idx"] = 0
        updates["current_worker_idx"] = 0
        updates["current_node"] = "master"
    else:
        updates["current_node"] = "interface"
    return updates


def _extract_task(state: IterareState) -> str:
    """Pull the task description from the most recent human message."""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        if hasattr(msg, "__class__") and msg.__class__.__name__ == "HumanMessage":
            return msg.content
    return state.get("proposal", "Unspecified task")


# ── Routing functions ─────────────────────────────────────────────────────────

def route(state: IterareState) -> str:
    """Central router — reads current_node from state."""
    node = state.get("current_node", "interface")
    if node == "end":
        return END
    return node


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """Assemble and compile the Iterare LangGraph."""
    g = StateGraph(IterareState)

    g.add_node("interface", interface_node)
    g.add_node("approval", approval_node)
    g.add_node("master", master_node)
    g.add_node("manager", manager_node)
    g.add_node("worker", worker_node)
    g.add_node("manager_triage", manager_triage_node)

    g.set_entry_point("interface")

    # All nodes route through the central router
    for node in ("interface", "approval", "master", "manager", "worker", "manager_triage"):
        g.add_conditional_edges(node, route)

    kwargs = {}
    if checkpointer:
        kwargs["checkpointer"] = checkpointer

    return g.compile(**kwargs)


# ── Default state initializer ─────────────────────────────────────────────────

def initial_state(user_message: str) -> IterareState:
    """Create a clean initial state for a new conversation turn."""
    from langchain_core.messages import HumanMessage
    return IterareState(
        messages=[HumanMessage(content=user_message)],
        current_node="interface",
        task_id=None,
        task_description=None,
        manager_tasks=[],
        current_manager_idx=0,
        worker_tasks=[],
        current_worker_idx=0,
        lateral_messages=[],
        proposal=None,
        approval_granted=None,
        max_attempts=int(os.getenv("ITERARE_MAX_WORKER_ATTEMPTS", "3")),
        log=[],
        log_file=None,
    )
