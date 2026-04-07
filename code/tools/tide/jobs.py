"""High-level batch job interface for TIDE.

A "job" here means: upload a script → create kernel → execute → return output.
This maps to the interactive GPU server rather than a Kubernetes Job,
which is appropriate for the JupyterHub-based TIDE access model.

For true Kubernetes batch jobs (long-running, no interactive server needed),
see k8s_jobs.py (requires separate kubeconfig setup).
"""

import os
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .client import TIDEClient
from .execute import execute, ExecutionResult


@dataclass
class JobResult:
    job_id: str
    status: str                  # complete | failed
    output: str
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    remote_script_path: Optional[str] = None


def run_script(
    client: TIDEClient,
    local_script: str,
    timeout: int = 3600,
    remote_dir: str = "iterare_jobs",
    on_output=None,
    cleanup: bool = True,
    env_inject: dict | None = None,
) -> JobResult:
    """
    Upload a local Python script to TIDE and execute it.
    Returns the job result with all output captured.

    local_script: path to a .py file on this machine
    timeout: max seconds to wait for completion
    remote_dir: directory on the TIDE server to upload to
    on_output: callable(text) for live stdout streaming
    cleanup: delete the remote script after execution
    env_inject: dict of env vars to set in the remote kernel before running
                the script. Use this to pass secrets (e.g. PUSHBULLET_API_KEY)
                without embedding them in script files.
    """
    script_path = Path(local_script)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {local_script}")

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    remote_path = f"{remote_dir}/{job_id}_{script_path.name}"

    # Ensure server is running
    client.start_server(wait=True)

    # Upload script — prepend env var injection block if requested
    if env_inject:
        import json as _json
        env_block = (
            "import os as _os\n"
            f"_os.environ.update({_json.dumps(env_inject)})\n"
            "del _os\n\n"
        )
        original = script_path.read_text()
        import tempfile, pathlib
        tmp = pathlib.Path(tempfile.mktemp(suffix=".py"))
        tmp.write_text(env_block + original)
        client.upload_file(str(tmp), remote_path)
        tmp.unlink()
    else:
        client.upload_file(str(script_path), remote_path)

    # Execute via kernel — runpy.run_path is safer than exec(open().read())
    code = f"import runpy; runpy.run_path('{remote_path}', run_name='__main__')"
    kernel_id = client.create_kernel()
    try:
        exec_result = execute(
            client.server_api,
            client.server_ws,
            kernel_id,
            client.token,
            code,
            timeout=timeout,
            on_output=on_output,
        )
    finally:
        client.delete_kernel(kernel_id)
        if cleanup:
            try:
                client.delete_file(remote_path)
            except Exception:
                pass

    return JobResult(
        job_id=job_id,
        status="failed" if exec_result.error else "complete",
        output=exec_result.output,
        error=exec_result.error,
        elapsed_seconds=exec_result.elapsed_seconds,
        remote_script_path=None if cleanup else remote_path,
    )


def run_code(
    client: TIDEClient,
    code: str,
    timeout: int = 3600,
    on_output=None,
) -> JobResult:
    """
    Execute an inline Python code string on TIDE.
    Useful for quick jobs without creating a local script file.
    """
    job_id = f"job-{uuid.uuid4().hex[:8]}"

    client.start_server(wait=True)
    kernel_id = client.create_kernel()
    try:
        exec_result = execute(
            client.server_api,
            client.server_ws,
            kernel_id,
            client.token,
            code,
            timeout=timeout,
            on_output=on_output,
        )
    finally:
        client.delete_kernel(kernel_id)

    return JobResult(
        job_id=job_id,
        status="failed" if exec_result.error else "complete",
        output=exec_result.output,
        error=exec_result.error,
        elapsed_seconds=exec_result.elapsed_seconds,
    )


def gpu_info(client: TIDEClient) -> str:
    """Return nvidia-smi output from the TIDE server."""
    result = run_code(
        client,
        "import subprocess; print(subprocess.check_output(['nvidia-smi'], text=True))",
        timeout=30,
    )
    return result.output or result.error or "No GPU info available."
