"""Iterare CLI — Interface Agent entry point.

Usage:
    iterare              Start a new session
    iterare --thread ID  Resume a saved session by thread ID
"""

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.rule import Rule

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

# Load .env from repo root
_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_ROOT / ".env")
os.environ.setdefault("ITERARE_ROOT", str(_ROOT))

from iterare.graph import build_graph, initial_state
from iterare.checkpointer import get_checkpointer

console = Console()

_BANNER = """
[bold]Iterare[/bold] — agent-native research platform
Type your research idea or question. Type [bold]exit[/bold] to quit.
"""

_SESSION_FILE = _ROOT / ".iterare_session"


def _load_session() -> str | None:
    if _SESSION_FILE.exists():
        return _SESSION_FILE.read_text().strip() or None
    return None


def _save_session(thread_id: str) -> None:
    _SESSION_FILE.write_text(thread_id)


def _clear_session() -> None:
    if _SESSION_FILE.exists():
        _SESSION_FILE.unlink()


def _handle_interrupt(interrupt_data: dict, graph, config: dict) -> dict:
    """Present an interrupt to the user and return the resume payload."""
    if interrupt_data.get("type") == "spawn_approval":
        console.print()
        console.print(Rule("Agent Spawn Proposal", style="yellow"))
        console.print(Markdown(interrupt_data["proposal"]))
        console.print()
        approved = Confirm.ask("[yellow]Proceed?[/yellow]", default=True)
        return {"approved": approved}

    # Generic interrupt fallback
    console.print()
    console.print(Rule("Input Required", style="yellow"))
    console.print(interrupt_data)
    response = Prompt.ask("> ")
    return {"response": response}


def _run_turn(graph, user_input: str, config: dict) -> None:
    """Run one conversation turn, handling interrupts and streaming output."""
    state = initial_state(user_input)

    try:
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            _print_updates(chunk)

    except GraphInterrupt as exc:
        interrupt_data = exc.args[0] if exc.args else {}
        resume_payload = _handle_interrupt(interrupt_data, graph, config)

        # Resume the graph
        try:
            for chunk in graph.stream(Command(resume=resume_payload), config=config, stream_mode="updates"):
                _print_updates(chunk)
        except GraphInterrupt as inner_exc:
            # Nested interrupt (e.g. human approval gates during task execution)
            inner_data = inner_exc.args[0] if inner_exc.args else {}
            inner_resume = _handle_interrupt(inner_data, graph, config)
            for chunk in graph.stream(Command(resume=inner_resume), config=config, stream_mode="updates"):
                _print_updates(chunk)


def _print_updates(chunk: dict) -> None:
    """Print meaningful agent updates; suppress noise."""
    for node_name, updates in chunk.items():
        if node_name in ("__interrupt__",):
            continue

        msgs = updates.get("messages", [])
        for msg in msgs:
            if hasattr(msg, "type") and msg.type == "ai":
                content = msg.content
                if content and not content.startswith("["):
                    console.print()
                    console.print(f"[dim]{node_name}[/dim]")
                    console.print(Markdown(str(content)))

        # Show log events at decision/result level only
        for entry in updates.get("log", []):
            if entry["event"] in ("result", "failure"):
                icon = "✓" if entry["event"] == "result" else "✗"
                console.print(f"  [dim]{icon} {entry['agent']}: {entry['detail']}[/dim]")


def _resume_turn(graph, config: dict) -> None:
    """Resume an interrupted graph (e.g. after process restart mid-task)."""
    console.print("[dim]Resuming previous session...[/dim]")
    state = graph.get_state(config)
    if state and state.next:
        try:
            for chunk in graph.stream(None, config=config, stream_mode="updates"):
                _print_updates(chunk)
        except GraphInterrupt as exc:
            interrupt_data = exc.args[0] if exc.args else {}
            resume_payload = _handle_interrupt(interrupt_data, graph, config)
            for chunk in graph.stream(Command(resume=resume_payload), config=config, stream_mode="updates"):
                _print_updates(chunk)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Iterare Interface Agent")
    parser.add_argument("--thread", help="Resume a specific thread ID")
    parser.add_argument("--clear", action="store_true", help="Clear saved session and start fresh")
    parser.add_argument("--list-threads", action="store_true", help="List saved thread IDs")
    args = parser.parse_args()

    if args.clear:
        _clear_session()
        console.print("[dim]Session cleared.[/dim]")

    console.print(_BANNER)

    # Determine thread ID
    if args.thread:
        thread_id = args.thread
        console.print(f"[dim]Thread: {thread_id}[/dim]")
    else:
        saved = _load_session()
        if saved:
            use_saved = Confirm.ask(f"[dim]Resume previous session ({saved[:12]}...)?[/dim]", default=True)
            thread_id = saved if use_saved else str(uuid.uuid4())
        else:
            thread_id = str(uuid.uuid4())

    _save_session(thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        # If resuming a thread with pending state, try to continue
        try:
            state = graph.get_state(config)
            if state and state.next:
                _resume_turn(graph, config)
        except Exception:
            pass  # New thread or no pending state

        # Main interaction loop
        while True:
            try:
                console.print()
                user_input = Prompt.ask("[bold green]>[/bold green]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session saved. Run 'iterare' to resume.[/dim]")
                break

            if not user_input.strip():
                continue
            if user_input.strip().lower() in ("exit", "quit", "q"):
                console.print("[dim]Session saved. Run 'iterare' to resume.[/dim]")
                break

            _run_turn(graph, user_input, config)


if __name__ == "__main__":
    main()
