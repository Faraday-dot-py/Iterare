"""TIDEClient — JupyterHub + Jupyter Server API client for TIDE.

Authentication: JupyterHub token (TIDE_API_KEY in .env).

Two API surfaces:
  JupyterHub API  — https://<hub>/hub/api/       — server management
  Jupyter Server  — https://<hub>/user/<name>/api/ — kernels, files, execution
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(os.getenv("ITERARE_ROOT", Path(__file__).resolve().parents[3])) / ".env")

HUB_BASE = "https://csu-tide-jupyterhub.nrp-nautilus.io"


class TIDEClient:
    def __init__(
        self,
        token: str | None = None,
        username: str | None = None,
        hub_base: str = HUB_BASE,
    ):
        self.token = token or os.getenv("TIDE_API_KEY") or _required("TIDE_API_KEY")
        self.username = username or os.getenv("TIDE_USERNAME") or _required("TIDE_USERNAME")
        self.hub_base = hub_base.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"token {self.token}"})

    # ── URL helpers ───────────────────────────────────────────────────────────

    @property
    def hub_api(self) -> str:
        return f"{self.hub_base}/hub/api"

    @property
    def server_api(self) -> str:
        return f"{self.hub_base}/user/{self.username}/api"

    @property
    def server_ws(self) -> str:
        base = self.hub_base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{base}/user/{self.username}/api"

    # ── JupyterHub server management ──────────────────────────────────────────

    def server_status(self) -> dict:
        """Return info about the user's JupyterHub server."""
        r = self._session.get(f"{self.hub_api}/users/{self.username}")
        r.raise_for_status()
        data = r.json()
        servers = data.get("servers", {})
        default = servers.get("", {})
        return {
            "ready": default.get("ready", False),
            "stopped": default.get("stopped", True),
            "pending": default.get("pending"),
            "started": default.get("started"),
            "last_activity": default.get("last_activity"),
            "profile": default.get("user_options", {}),
            "url": default.get("url"),
        }

    def start_server(self, wait: bool = True, timeout: int = 120) -> dict:
        """Start the JupyterHub server if it is not running."""
        import time
        status = self.server_status()
        if status["ready"]:
            return status

        r = self._session.post(f"{self.hub_api}/users/{self.username}/server")
        if r.status_code not in (200, 201, 202):
            r.raise_for_status()

        if not wait:
            return self.server_status()

        elapsed = 0
        while elapsed < timeout:
            time.sleep(3)
            elapsed += 3
            status = self.server_status()
            if status["ready"]:
                return status
        raise TimeoutError(f"Server did not start within {timeout}s.")

    def stop_server(self) -> None:
        """Stop the JupyterHub server."""
        r = self._session.delete(f"{self.hub_api}/users/{self.username}/server")
        if r.status_code not in (200, 202, 204):
            r.raise_for_status()

    # ── Kernel management ─────────────────────────────────────────────────────

    def create_kernel(self, kernel_name: str = "python3") -> str:
        """Create a new kernel and return its ID."""
        r = self._session.post(
            f"{self.server_api}/kernels",
            json={"name": kernel_name},
        )
        r.raise_for_status()
        return r.json()["id"]

    def list_kernels(self) -> list[dict]:
        """Return all running kernels."""
        r = self._session.get(f"{self.server_api}/kernels")
        r.raise_for_status()
        return r.json()

    def delete_kernel(self, kernel_id: str) -> None:
        """Delete a kernel."""
        r = self._session.delete(f"{self.server_api}/kernels/{kernel_id}")
        if r.status_code not in (200, 204):
            r.raise_for_status()

    # ── Contents (file) API ───────────────────────────────────────────────────

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a local file to the Jupyter server."""
        import base64
        content = Path(local_path).read_bytes()
        payload = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(content).decode(),
        }
        r = self._session.put(
            f"{self.server_api}/contents/{remote_path.lstrip('/')}",
            json=payload,
        )
        r.raise_for_status()

    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a file from the Jupyter server."""
        import base64
        r = self._session.get(
            f"{self.server_api}/contents/{remote_path.lstrip('/')}",
            params={"format": "base64"},
        )
        r.raise_for_status()
        data = r.json()
        content = base64.b64decode(data["content"])
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(content)

    def list_files(self, remote_path: str = "") -> list[dict]:
        """List files/directories at a remote path."""
        r = self._session.get(
            f"{self.server_api}/contents/{remote_path.lstrip('/')}",
        )
        r.raise_for_status()
        data = r.json()
        if data.get("type") == "directory":
            return [
                {"name": item["name"], "type": item["type"], "size": item.get("size")}
                for item in data.get("content", [])
            ]
        return [{"name": data["name"], "type": data["type"], "size": data.get("size")}]

    def delete_file(self, remote_path: str) -> None:
        """Delete a file on the Jupyter server."""
        r = self._session.delete(
            f"{self.server_api}/contents/{remote_path.lstrip('/')}",
        )
        if r.status_code not in (200, 204):
            r.raise_for_status()

    def verify_connection(self) -> dict:
        """Quick health check — returns server status."""
        return self.server_status()


def _required(var: str) -> str:
    raise EnvironmentError(
        f"Required env var '{var}' is not set. Add it to .env or export it."
    )
