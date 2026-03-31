"""TIDE compute runner — callable by any Iterare agent.

Agents use this module when a task requires significant compute (ML training,
large-scale data processing, heavy inference). The host machine is not adequate
for these workloads; all such jobs run on TIDE.

Usage (from agent context):
    from iterare.tools.tide_runner import run_on_tide, tide_available

    if tide_available():
        result = run_on_tide(code="import torch; print(torch.cuda.device_count())")
        print(result.output)
"""

import os
import sys
import tempfile
from pathlib import Path

_TIDE_TOOL = Path(__file__).resolve().parents[3] / "code" / "tools"


def tide_available() -> bool:
    """Return True if the TIDE tool and required env vars are configured."""
    return bool(
        os.getenv("TIDE_API_KEY") and os.getenv("TIDE_USERNAME")
    )


def _get_client():
    if str(_TIDE_TOOL) not in sys.path:
        sys.path.insert(0, str(_TIDE_TOOL))
    from tide.client import TIDEClient
    return TIDEClient()


def run_on_tide(
    script_path: str | None = None,
    code: str | None = None,
    timeout: int = 86400,
    on_output=None,
):
    """
    Run a compute job on TIDE. Provide either a script file path or inline code.

    script_path: path to a local .py file to upload and execute
    code: inline Python code string to execute
    timeout: max seconds (default 86400 = 24h)
    on_output: optional callable(text) for live stdout streaming

    Returns a JobResult with .output, .error, .status, .elapsed_seconds
    Raises RuntimeError if TIDE is not configured.
    """
    if not tide_available():
        raise RuntimeError(
            "TIDE is not configured. Set TIDE_API_KEY and TIDE_USERNAME in .env"
        )
    if not script_path and not code:
        raise ValueError("Provide either script_path or code.")

    if str(_TIDE_TOOL) not in sys.path:
        sys.path.insert(0, str(_TIDE_TOOL))
    from tide.jobs import run_script, run_code

    client = _get_client()

    if script_path:
        return run_script(client, script_path, timeout=timeout, on_output=on_output)

    # Write inline code to a temp file and submit as script
    # (exec() of multi-line code loses tracebacks; script file preserves them)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        return run_script(client, tmp_path, timeout=timeout, on_output=on_output)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def gpu_info_tide() -> str:
    """Return nvidia-smi output from the TIDE server."""
    if str(_TIDE_TOOL) not in sys.path:
        sys.path.insert(0, str(_TIDE_TOOL))
    from tide.jobs import gpu_info
    return gpu_info(_get_client())
