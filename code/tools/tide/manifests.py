"""Pre-built job templates for common TIDE workload patterns.

All templates return a JobSpec that can be submitted as-is or further
customised before calling submit_job().
"""

import uuid
from .jobs import JobSpec


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


def gpu_python_job(
    script_path: str,
    image: str = "ghcr.io/csu-tide/hello-csu:main",
    gpu_count: int = 1,
    memory: str = "16Gi",
    name: str | None = None,
    pvc_mounts: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
) -> JobSpec:
    """
    Run a Python script on a TIDE GPU node.

    script_path: path to the .py script inside the container
    image: container image (must have Python + your deps installed)
    gpu_count: number of NVIDIA GPUs to request (default 1, max 4 per node)
    memory: memory limit (default 16Gi)
    pvc_mounts: {claim_name: mount_path} persistent volumes to attach
    """
    return JobSpec(
        name=name or _uid("gpu-py"),
        image=image,
        command=["python", script_path],
        cpu_request="2",
        cpu_limit="8",
        memory_request=memory,
        memory_limit=memory,
        gpu_count=gpu_count,
        pvc_mounts=pvc_mounts or {},
        env=env or {},
        labels={"workload-type": "gpu-python"},
    )


def cpu_python_job(
    script_path: str,
    image: str = "ghcr.io/csu-tide/hello-csu:main",
    cpu_cores: int = 4,
    memory: str = "8Gi",
    name: str | None = None,
    pvc_mounts: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
) -> JobSpec:
    """
    Run a Python script on a TIDE CPU node.
    TIDE has 6 CPU nodes with 64 cores each.
    """
    return JobSpec(
        name=name or _uid("cpu-py"),
        image=image,
        command=["python", script_path],
        cpu_request=str(cpu_cores),
        cpu_limit=str(cpu_cores),
        memory_request=memory,
        memory_limit=memory,
        gpu_count=0,
        pvc_mounts=pvc_mounts or {},
        env=env or {},
        labels={"workload-type": "cpu-python"},
    )


def shell_job(
    script: str,
    image: str = "ubuntu:22.04",
    cpu_cores: int = 2,
    memory: str = "4Gi",
    name: str | None = None,
    pvc_mounts: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
) -> JobSpec:
    """
    Run an inline shell script.
    script: full shell script as a string (run via bash -c)
    """
    return JobSpec(
        name=name or _uid("shell"),
        image=image,
        command=["bash", "-c", script],
        cpu_request=str(cpu_cores),
        cpu_limit=str(cpu_cores),
        memory_request=memory,
        memory_limit=memory,
        gpu_count=0,
        pvc_mounts=pvc_mounts or {},
        env=env or {},
        labels={"workload-type": "shell"},
    )


def from_yaml(path: str) -> dict:
    """
    Load a raw Kubernetes Job manifest from a YAML file.
    Returns the raw dict (not a JobSpec) — pass directly to:
      client.batch.create_namespaced_job(namespace=..., body=manifest)
    """
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
