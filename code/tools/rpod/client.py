"""RunPodClient — pod lifecycle management via the RunPod SDK.

Handles pod creation, status polling, SSH key registration, and teardown.
SSH key is auto-generated at ~/.runpod/ssh/iterare (if absent) and registered
with RunPod on first use.
"""

import os
import time
from pathlib import Path
from typing import Optional

import runpod
from dotenv import load_dotenv

load_dotenv(Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[3])) / ".env")

SSH_KEY_DIR = Path.home() / ".runpod" / "ssh"
SSH_KEY_PATH = SSH_KEY_DIR / "iterare"

# Default pod config — override per-call as needed
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPU = "NVIDIA H100 80GB HBM3"
DEFAULT_DISK_GB = 50


class RunPodClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY") or _required("RUNPOD_API_KEY")
        runpod.api_key = self.api_key

    # ── SSH key management ────────────────────────────────────────────────────

    def ensure_ssh_key(self) -> tuple[Path, str]:
        """Return (private_key_path, public_key_str), generating + registering if needed."""
        SSH_KEY_DIR.mkdir(parents=True, exist_ok=True)

        if not SSH_KEY_PATH.exists():
            _generate_ssh_key(SSH_KEY_PATH)

        pub_path = SSH_KEY_PATH.with_suffix(".pub")
        pubkey = pub_path.read_text().strip()

        # Register with RunPod if not already set
        user = runpod.get_user()
        if user.get("pubKey") != pubkey:
            runpod.update_user_settings(pubkey=pubkey)

        return SSH_KEY_PATH, pubkey

    # ── Pod lifecycle ─────────────────────────────────────────────────────────

    def create_pod(
        self,
        name: str,
        gpu_type_id: str = DEFAULT_GPU,
        gpu_count: int = 1,
        image_name: str = DEFAULT_IMAGE,
        container_disk_in_gb: int = DEFAULT_DISK_GB,
        env: Optional[dict] = None,
        cloud_type: str = "ALL",
        country_code: Optional[str] = None,
    ) -> dict:
        """Create a pod with SSH enabled. Returns the pod dict."""
        self.ensure_ssh_key()
        pod = runpod.create_pod(
            name=name,
            image_name=image_name,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
            container_disk_in_gb=container_disk_in_gb,
            cloud_type=cloud_type,
            country_code=country_code,
            start_ssh=True,
            support_public_ip=True,
            ports="22/tcp",
            env=env or {},
        )
        return pod

    def get_pod(self, pod_id: str) -> dict:
        return runpod.get_pod(pod_id)

    def wait_for_pod(self, pod_id: str, timeout: int = 300) -> dict:
        """Poll until the pod is running and has SSH port info. Returns pod dict."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            pod = self.get_pod(pod_id)
            if pod.get("desiredStatus") == "RUNNING" and _ssh_port(pod):
                return pod
            time.sleep(5)
        raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")

    def terminate_pod(self, pod_id: str) -> None:
        runpod.terminate_pod(pod_id)

    def stop_pod(self, pod_id: str) -> None:
        runpod.stop_pod(pod_id)

    def list_pods(self) -> list[dict]:
        return runpod.get_pods()

    def ssh_info(self, pod: dict) -> tuple[str, int]:
        """Return (host, port) for SSH connection from a pod dict."""
        port = _ssh_port(pod)
        if not port:
            raise RuntimeError(f"Pod {pod['id']} has no public SSH port yet")
        return port["ip"], port["publicPort"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ssh_port(pod: dict) -> Optional[dict]:
    """Find the public SSH port entry from a pod's runtime info."""
    runtime = pod.get("runtime") or {}
    for p in runtime.get("ports", []):
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            return p
    return None


def _generate_ssh_key(key_path: Path) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
        backend=default_backend(),
    )
    key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)

    pub_key = private_key.public_key()
    pub_str = pub_key.public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode() + " iterare"
    key_path.with_suffix(".pub").write_text(pub_str + "\n")


def _required(var: str) -> str:
    raise EnvironmentError(f"Required env var '{var}' is not set.")
