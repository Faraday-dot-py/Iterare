"""Job submission, monitoring, log retrieval, and deletion for TIDE/NRP."""

import time
from dataclasses import dataclass, field
from typing import Optional

from kubernetes import client as k8s_client
from kubernetes.client.exceptions import ApiException

from .client import TIDEClient


@dataclass
class JobSpec:
    """Describes a batch job to submit to TIDE.

    Minimal required fields: name, image, command.
    All resource values follow Kubernetes notation (e.g. "4", "500m" for CPU;
    "8Gi", "500Mi" for memory; "1" for 1 GPU).
    """
    name: str
    image: str
    command: list[str]

    # Resources
    cpu_request: str = "1"
    cpu_limit: str = "4"
    memory_request: str = "4Gi"
    memory_limit: str = "16Gi"
    gpu_count: int = 0                  # 0 = CPU-only job
    gpu_type: str = "nvidia.com/gpu"    # override for specific GPU models

    # Job behaviour
    restart_policy: str = "Never"       # Never | OnFailure
    backoff_limit: int = 2              # retries before marking failed
    active_deadline_seconds: Optional[int] = None   # wall-clock timeout

    # Storage — PVCs to mount: {claim_name: mount_path}
    pvc_mounts: dict[str, str] = field(default_factory=dict)

    # Extra environment variables: {KEY: value}
    env: dict[str, str] = field(default_factory=dict)

    # Raw extra container args (appended after command)
    args: list[str] = field(default_factory=list)

    # Labels added to the job and pods
    labels: dict[str, str] = field(default_factory=dict)


def submit_job(client: TIDEClient, spec: JobSpec) -> str:
    """Submit a batch job to TIDE. Returns the created job name."""
    manifest = _build_manifest(spec, client.namespace)
    try:
        result = client.batch.create_namespaced_job(
            namespace=client.namespace,
            body=manifest,
        )
        return result.metadata.name
    except ApiException as e:
        raise RuntimeError(f"Job submission failed: {e.status} {e.reason}\n{e.body}") from e


def job_status(client: TIDEClient, job_name: str) -> dict:
    """Return a summary of job status."""
    try:
        job = client.batch.read_namespaced_job(name=job_name, namespace=client.namespace)
    except ApiException as e:
        if e.status == 404:
            return {"status": "not_found", "name": job_name}
        raise

    status = job.status
    conditions = {c.type: c.status for c in (status.conditions or [])}

    if conditions.get("Complete") == "True":
        phase = "complete"
    elif conditions.get("Failed") == "True":
        phase = "failed"
    elif status.active:
        phase = "running"
    else:
        phase = "pending"

    return {
        "name": job_name,
        "status": phase,
        "active": status.active or 0,
        "succeeded": status.succeeded or 0,
        "failed": status.failed or 0,
        "start_time": str(status.start_time) if status.start_time else None,
        "completion_time": str(status.completion_time) if status.completion_time else None,
        "conditions": conditions,
    }


def job_logs(client: TIDEClient, job_name: str, tail_lines: int = 100) -> str:
    """Fetch logs from the pod(s) belonging to a job."""
    pods = client.core.list_namespaced_pod(
        namespace=client.namespace,
        label_selector=f"job-name={job_name}",
    )
    if not pods.items:
        return f"No pods found for job '{job_name}'."

    output = []
    for pod in pods.items:
        pod_name = pod.metadata.name
        pod_phase = pod.status.phase
        output.append(f"=== Pod: {pod_name} ({pod_phase}) ===")
        try:
            logs = client.core.read_namespaced_pod_log(
                name=pod_name,
                namespace=client.namespace,
                tail_lines=tail_lines,
            )
            output.append(logs or "(no output)")
        except ApiException as e:
            output.append(f"(log unavailable: {e.reason})")

    return "\n".join(output)


def list_jobs(client: TIDEClient, label_selector: str = "") -> list[dict]:
    """List jobs in the namespace, optionally filtered by label selector."""
    jobs = client.batch.list_namespaced_job(
        namespace=client.namespace,
        label_selector=label_selector or None,
    )
    return [job_status(client, j.metadata.name) for j in jobs.items]


def delete_job(client: TIDEClient, job_name: str, delete_pods: bool = True) -> str:
    """Delete a job and optionally its pods."""
    propagation = "Foreground" if delete_pods else "Orphan"
    try:
        client.batch.delete_namespaced_job(
            name=job_name,
            namespace=client.namespace,
            body=k8s_client.V1DeleteOptions(propagation_policy=propagation),
        )
        return f"Deleted job '{job_name}'."
    except ApiException as e:
        if e.status == 404:
            return f"Job '{job_name}' not found."
        raise


def wait_for_job(
    client: TIDEClient,
    job_name: str,
    poll_interval: int = 10,
    timeout: int = 3600,
    on_poll=None,
) -> dict:
    """
    Block until a job reaches a terminal state (complete or failed).
    on_poll: optional callable(status_dict) called after each poll.
    Returns the final status dict.
    """
    elapsed = 0
    while elapsed < timeout:
        status = job_status(client, job_name)
        if on_poll:
            on_poll(status)
        if status["status"] in ("complete", "failed", "not_found"):
            return status
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"Job '{job_name}' did not complete within {timeout}s.")


# ── Manifest builder ──────────────────────────────────────────────────────────

def _build_manifest(spec: JobSpec, namespace: str) -> k8s_client.V1Job:
    """Construct a V1Job object from a JobSpec."""
    labels = {"app": spec.name, "managed-by": "iterare", **spec.labels}

    # Resource requirements
    requests = {"cpu": spec.cpu_request, "memory": spec.memory_request}
    limits = {"cpu": spec.cpu_limit, "memory": spec.memory_limit}
    if spec.gpu_count > 0:
        limits[spec.gpu_type] = str(spec.gpu_count)
        requests[spec.gpu_type] = str(spec.gpu_count)

    # Volume mounts
    volume_mounts = [
        k8s_client.V1VolumeMount(name=_pvc_vol_name(claim), mount_path=path)
        for claim, path in spec.pvc_mounts.items()
    ]
    volumes = [
        k8s_client.V1Volume(
            name=_pvc_vol_name(claim),
            persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                claim_name=claim
            ),
        )
        for claim in spec.pvc_mounts
    ]

    # Environment variables
    env_vars = [
        k8s_client.V1EnvVar(name=k, value=v)
        for k, v in spec.env.items()
    ]

    container = k8s_client.V1Container(
        name=spec.name,
        image=spec.image,
        command=spec.command,
        args=spec.args or None,
        resources=k8s_client.V1ResourceRequirements(requests=requests, limits=limits),
        volume_mounts=volume_mounts or None,
        env=env_vars or None,
    )

    pod_spec = k8s_client.V1PodSpec(
        containers=[container],
        restart_policy=spec.restart_policy,
        volumes=volumes or None,
    )

    pod_template = k8s_client.V1PodTemplateSpec(
        metadata=k8s_client.V1ObjectMeta(labels=labels),
        spec=pod_spec,
    )

    job_spec = k8s_client.V1JobSpec(
        template=pod_template,
        backoff_limit=spec.backoff_limit,
        active_deadline_seconds=spec.active_deadline_seconds,
    )

    return k8s_client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=k8s_client.V1ObjectMeta(name=spec.name, namespace=namespace, labels=labels),
        spec=job_spec,
    )


def _pvc_vol_name(claim_name: str) -> str:
    return f"vol-{claim_name.lower().replace('_', '-')}"
