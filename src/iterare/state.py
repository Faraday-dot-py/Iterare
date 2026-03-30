"""Iterare graph state definitions."""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class WorkerTask(TypedDict):
    id: str
    description: str
    status: str          # pending | active | complete | failed | escalated
    attempts: int
    result: Optional[str]
    error: Optional[str]


class ManagerTask(TypedDict):
    id: str
    description: str
    status: str          # pending | active | complete | failed
    worker_tasks: list[WorkerTask]
    summary: Optional[str]


class LogEntry(TypedDict):
    timestamp: str
    agent: str
    event: str           # decision | handoff | failure | result | approval
    detail: str


class IterareState(TypedDict):
    # Conversation with the lead developer
    messages: Annotated[list, add_messages]

    # Current routing target
    current_node: str    # interface | approval | master | manager | worker | manager_triage | end

    # Top-level task (set when Interface spawns a Master)
    task_id: Optional[str]
    task_description: Optional[str]

    # Master's plan: list of manager tasks
    manager_tasks: list[ManagerTask]
    current_manager_idx: int

    # Current Manager's plan: list of worker tasks
    worker_tasks: list[WorkerTask]
    current_worker_idx: int

    # Lateral worker communication (workers write here; manager reads)
    lateral_messages: Annotated[list, operator.add]

    # Approval gate
    proposal: Optional[str]          # 1-3 line proposal from Interface
    approval_granted: Optional[bool]

    # Worker loop detection
    max_attempts: int

    # Structured log (append-only; written at decision/handoff points)
    log: Annotated[list[LogEntry], operator.add]

    # Path to the hot log file for this task
    log_file: Optional[str]
