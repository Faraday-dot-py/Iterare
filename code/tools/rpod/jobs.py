"""High-level job interface for RunPod.

A job: create pod → wait → upload script → execute → download outputs → terminate.
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from .client import RunPodClient, DEFAULT_GPU, DEFAULT_IMAGE, DEFAULT_DISK_GB
from .execute import run_ssh, upload_file, download_file, ExecutionResult


@dataclass
class JobResult:
    job_id: str
    pod_id: str
    status: str               # complete | failed
    output: str
    stderr: str = ""
    exit_code: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


def run_script(
    client: RunPodClient,
    local_script: str,
    *,
    gpu_type_id: str = DEFAULT_GPU,
    gpu_count: int = 1,
    image_name: str = DEFAULT_IMAGE,
    container_disk_in_gb: int = DEFAULT_DISK_GB,
    timeout: int = 3600,
    download_paths: Optional[list[tuple[str, str]]] = None,
    on_output: Optional[Callable[[str], None]] = None,
    terminate_on_complete: bool = True,
    env: Optional[dict] = None,
) -> JobResult:
    """
    Create a pod, upload a Python script, execute it, optionally download outputs.

    download_paths: list of (remote_path, local_path) to pull after execution
    """
    script_path = Path(local_script)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {local_script}")

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    pod_name = f"iterare-{job_id}"
    remote_script = f"/root/{script_path.name}"

    key_path, _ = client.ensure_ssh_key()
    pod = client.create_pod(
        name=pod_name,
        gpu_type_id=gpu_type_id,
        gpu_count=gpu_count,
        image_name=image_name,
        container_disk_in_gb=container_disk_in_gb,
        env=env,
    )
    pod_id = pod["id"]

    try:
        pod = client.wait_for_pod(pod_id, timeout=300)
        host, port = client.ssh_info(pod)

        upload_file(host, port, key_path, str(script_path), remote_script)

        exec_result = run_ssh(
            host, port, key_path,
            f"cd /root && python3 {remote_script}",
            timeout=timeout,
            on_output=on_output,
        )

        if download_paths:
            for remote_path, local_path in download_paths:
                try:
                    download_file(host, port, key_path, remote_path, local_path)
                except Exception as e:
                    if on_output:
                        on_output(f"[warn] download {remote_path} failed: {e}\n")

    finally:
        if terminate_on_complete:
            try:
                client.terminate_pod(pod_id)
            except Exception:
                pass

    return JobResult(
        job_id=job_id,
        pod_id=pod_id,
        status="complete" if exec_result.success else "failed",
        output=exec_result.stdout,
        stderr=exec_result.stderr,
        exit_code=exec_result.exit_code,
        elapsed_seconds=exec_result.elapsed_seconds,
        error=exec_result.error,
    )


def run_code(
    client: RunPodClient,
    code: str,
    *,
    gpu_type_id: str = DEFAULT_GPU,
    gpu_count: int = 1,
    image_name: str = DEFAULT_IMAGE,
    container_disk_in_gb: int = DEFAULT_DISK_GB,
    timeout: int = 3600,
    on_output: Optional[Callable[[str], None]] = None,
    terminate_on_complete: bool = True,
    env: Optional[dict] = None,
) -> JobResult:
    """Execute an inline Python code string on a RunPod instance."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        return run_script(
            client, tmp_path,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
            image_name=image_name,
            container_disk_in_gb=container_disk_in_gb,
            timeout=timeout,
            on_output=on_output,
            terminate_on_complete=terminate_on_complete,
            env=env,
        )
    finally:
        os.unlink(tmp_path)


def gpu_info(client: RunPodClient, gpu_type_id: str = DEFAULT_GPU) -> str:
    """Return nvidia-smi output from a freshly created pod."""
    result = run_code(
        client,
        "import subprocess; print(subprocess.check_output(['nvidia-smi'], text=True))",
        gpu_type_id=gpu_type_id,
        gpu_count=1,
        timeout=120,
    )
    return result.output or result.error or "No GPU info available."
