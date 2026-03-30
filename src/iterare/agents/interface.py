"""Interface Agent node.

The sole direct interface for the lead developer. Has full Manager capabilities
but will not spin up agents without asking first. Holds highest authority.
"""

import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from iterare.state import IterareState
from iterare.agents.base import load_template, get_llm, build_system_message, make_log_entry


@tool
def propose_spawn(proposal: str) -> str:
    """
    Propose spawning a Master Agent to the lead developer.
    Call this when the task warrants sustained multi-agent execution.
    proposal: 1-3 line description of what the Master would do and why.
    """
    return proposal


@tool
def respond_directly(response: str) -> str:
    """
    Respond directly to the lead developer without spawning any agents.
    Use for simple questions, small tasks, or clarifications.
    """
    return response


_TOOLS = [propose_spawn, respond_directly]
_TOOL_MAP = {t.name: t for t in _TOOLS}


def interface_node(state: IterareState) -> dict:
    """Interface Agent: receives user input and decides to handle or delegate."""
    template = load_template("interface")
    llm = get_llm().bind_tools(_TOOLS)
    system = build_system_message(template)

    messages = [system] + state["messages"]
    response = llm.invoke(messages)

    updates: dict = {
        "messages": [response],
        "log": [make_log_entry("interface", "decision", str(response.content))],
    }

    # Parse tool calls
    if response.tool_calls:
        call = response.tool_calls[0]
        name = call["name"]
        args = call["args"]

        if name == "propose_spawn":
            proposal = args["proposal"]
            updates["proposal"] = proposal
            updates["current_node"] = "approval"
            updates["log"].append(make_log_entry("interface", "handoff", f"Proposing spawn: {proposal}"))

        elif name == "respond_directly":
            # Deliver the response as a plain AI message and end
            updates["messages"].append(AIMessage(content=args["response"]))
            updates["current_node"] = "end"

        # Append a ToolMessage so the conversation history is valid
        updates["messages"].append(
            ToolMessage(content=args.get("proposal") or args.get("response", ""), tool_call_id=call["id"])
        )
    else:
        # No tool call — treat as direct response
        updates["current_node"] = "end"

    return updates
