"""Iterare integration for RunPod compute jobs.

Usage in worker agents:
    from iterare.tools.rpod_runner import run_on_rpod, RunPodJobResult

    result = run_on_rpod(
        script_path="tasks/<id>/manager-1/worker-1/train.py",
        gpu_type_id="NVIDIA H100 80GB HBM3",
        gpu_count=8,
        download_paths=[("/root/artifact.gz", "tasks/<id>/.../artifact.gz")],
        timeout=3600,
        on_output=print,
    )
    if result.success:
        print(result.output)
    else:
        print("Error:", result.error)
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

_ITERARE_ROOT = Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[3]))
_TOOLS_PATH = str(_ITERARE_ROOT / "code" / "tools")
if _TOOLS_PATH not in sys.path:
    sys.path.insert(0, _TOOLS_PATH)

from rpod.client import RunPodClient, DEFAULT_GPU, DEFAULT_IMAGE
from rpod.jobs import run_script, run_code, JobResult


def run_on_rpod(
    script_path: str,
    *,
    gpu_type_id: str = DEFAULT_GPU,
    gpu_count: int = 1,
    image_name: str = DEFAULT_IMAGE,
    container_disk_in_gb: int = 50,
    timeout: int = 3600,
    download_paths: Optional[list[tuple[str, str]]] = None,
    on_output: Optional[Callable[[str], None]] = None,
    terminate_on_complete: bool = True,
    env: Optional[dict] = None,
) -> JobResult:
    """
    Run a Python script on a RunPod GPU instance and return the result.

    script_path: local path to the .py script (absolute or relative to cwd)
    gpu_type_id: e.g. "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"
    gpu_count:   number of GPUs (1–8 depending on availability)
    download_paths: [(remote_path, local_path), ...] to pull after job completes
    """
    client = RunPodClient()
    return run_script(
        client,
        script_path,
        gpu_type_id=gpu_type_id,
        gpu_count=gpu_count,
        image_name=image_name,
        container_disk_in_gb=container_disk_in_gb,
        timeout=timeout,
        download_paths=download_paths,
        on_output=on_output,
        terminate_on_complete=terminate_on_complete,
        env=env,
    )


def gpu_info_rpod(gpu_type_id: str = DEFAULT_GPU) -> str:
    """Return nvidia-smi output from a fresh RunPod instance."""
    from rpod.jobs import gpu_info
    client = RunPodClient()
    return gpu_info(client, gpu_type_id=gpu_type_id)
