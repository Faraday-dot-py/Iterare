"""Base agent: template loading and LLM setup."""

import os
from pathlib import Path
from datetime import datetime, timezone

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

from iterare.state import LogEntry

# Templates directory relative to the repo root
_TEMPLATES_DIR = Path(os.getenv("ITERARE_ROOT", ".")) / "templates"

# Default model; override via ITERARE_MODEL env var
_DEFAULT_MODEL = "claude-sonnet-4-6"


def load_template(role: str) -> dict:
    """Load and return a parsed agent guideline template by role name."""
    path = _TEMPLATES_DIR / f"{role}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_llm(temperature: float = 0.0) -> ChatAnthropic:
    """Return a configured ChatAnthropic instance."""
    return ChatAnthropic(
        model=os.getenv("ITERARE_MODEL", _DEFAULT_MODEL),
        temperature=temperature,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


def build_system_message(template: dict) -> SystemMessage:
    """Build a SystemMessage from a loaded template."""
    return SystemMessage(content=template["system_prompt"])


def make_log_entry(agent: str, event: str, detail: str) -> LogEntry:
    """Create a structured log entry."""
    return LogEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent=agent,
        event=event,
        detail=detail,
    )
