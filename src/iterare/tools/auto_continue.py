"""
auto_continue — run an experiment submit script, then resume Claude when done.

When experiments take hours, the user shouldn't have to babysit the terminal
and manually say "your experiments finished." This wrapper runs the submit
script, waits for completion, then invokes `claude -c -p` to continue the
research conversation automatically.

CLI usage (run in tmux/screen so it survives terminal close):
    python3 -m iterare.tools.auto_continue tasks/steer-001/submit_batch_exp42_44.py

    # Custom message:
    python3 -m iterare.tools.auto_continue --message "..." submit.py

    # Dry-run (skip claude invocation):
    python3 -m iterare.tools.auto_continue --dry-run submit.py

Programmatic usage (from a Claude Bash call):
    from iterare.tools.auto_continue import submit_and_continue
    submit_and_continue("tasks/steer-001/submit_batch_exp42_44.py")
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Default message sent to Claude when experiments finish.
DEFAULT_MESSAGE = (
    "Your experiments have finished, please continue using Iterare guidelines"
)

# Project root — two levels up from src/iterare/tools/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _find_claude() -> str | None:
    """Return path to claude CLI, or None if not found."""
    return shutil.which("claude")


def submit_and_continue(
    script: str | Path,
    message: str = DEFAULT_MESSAGE,
    dry_run: bool = False,
    cwd: str | Path | None = None,
) -> int:
    """
    Run *script* as a subprocess, then invoke `claude -c -p <message>`.

    Returns the exit code of the submit script (0 = success).
    If claude is not found or dry_run is True, skips the continuation step.
    """
    script = Path(script)
    cwd = Path(cwd) if cwd else _PROJECT_ROOT

    claude = _find_claude()
    if not claude and not dry_run:
        print(
            "WARNING: 'claude' not found in PATH — continuation step will be skipped.",
            flush=True,
        )

    # ── Run the submit script ─────────────────────────────────────────────────
    print(f"[auto_continue] Running: {script.name}", flush=True)
    print(f"[auto_continue] Working dir: {cwd}", flush=True)
    t0 = time.time()

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(cwd),
    )

    elapsed = time.time() - t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(
        f"[auto_continue] {script.name} exited {proc.returncode} "
        f"after {h:02d}:{m:02d}:{s:02d}",
        flush=True,
    )

    # ── Resume Claude ─────────────────────────────────────────────────────────
    if dry_run:
        print(f"[auto_continue] DRY RUN — would run: claude -c -p {message!r}", flush=True)
        return proc.returncode

    if not claude:
        print("[auto_continue] Skipping continuation (claude not in PATH).", flush=True)
        return proc.returncode

    print(f"[auto_continue] Resuming Claude: {message!r}", flush=True)
    subprocess.run(
        [claude, "-c", "-p", message],
        cwd=str(cwd),
    )

    return proc.returncode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="iterare-continue",
        description="Run an experiment submit script, then resume Claude when done.",
    )
    parser.add_argument("script", help="Path to submit script")
    parser.add_argument(
        "--message", "-m",
        default=DEFAULT_MESSAGE,
        help="Message to send Claude on completion (default: standard Iterare prompt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the submit script but skip the Claude invocation",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory (default: project root)",
    )
    args = parser.parse_args(argv)

    rc = submit_and_continue(
        script=args.script,
        message=args.message,
        dry_run=args.dry_run,
        cwd=args.cwd,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
