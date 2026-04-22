"""iterare agent interface to the Codex CLI tool.

Thin re-export so iterare tasks can import from a consistent location:

    from iterare.tools.codex_runner import CodexRunner, codex_exec

See `code/tools/codex/runner.py` for full documentation.
"""

import sys
from pathlib import Path

_CODEX_TOOL = Path(__file__).resolve().parents[3] / "code" / "tools"
if str(_CODEX_TOOL) not in sys.path:
    sys.path.insert(0, str(_CODEX_TOOL))

from codex.runner import CodexRunner, codex_exec, codex_review_file  # noqa: F401

__all__ = ["CodexRunner", "codex_exec", "codex_review_file"]
