"""TIDEClient — authenticated Kubernetes client for the NRP Nautilus / TIDE cluster.

Auth priority:
1. kubeconfig (~/.kube/config or KUBECONFIG env var)
2. TIDE_API_KEY + TIDE_API_SERVER environment variables

The TIDE_API_KEY can be either:
- A Kubernetes service account token (long JWT string)
- A short hex token issued by the NRP portal (32 chars)
"""

import os
from pathlib import Path

import kubernetes
from kubernetes import client as k8s_client
from kubernetes.client import ApiClient, Configuration


NRP_DEFAULT_SERVER = "https://nautilus.optiputer.net"


class TIDEClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_server: str | None = None,
        namespace: str | None = None,
        kubeconfig: str | None = None,
    ):
        self.namespace = namespace or os.getenv("TIDE_NAMESPACE") or self._required("TIDE_NAMESPACE")
        self._api_client = self._build_client(api_key, api_server, kubeconfig)

    # ── Public API objects ───────────────────────────────────────────────────

    @property
    def batch(self) -> k8s_client.BatchV1Api:
        return k8s_client.BatchV1Api(self._api_client)

    @property
    def core(self) -> k8s_client.CoreV1Api:
        return k8s_client.CoreV1Api(self._api_client)

    # ── Auth ─────────────────────────────────────────────────────────────────

    def _build_client(
        self,
        api_key: str | None,
        api_server: str | None,
        kubeconfig: str | None,
    ) -> ApiClient:
        # Try kubeconfig first
        kubeconfig_path = kubeconfig or os.getenv("KUBECONFIG") or str(Path.home() / ".kube" / "config")
        if Path(kubeconfig_path).exists():
            kubernetes.config.load_kube_config(config_file=kubeconfig_path)
            return ApiClient()

        # Fall back to API key
        key = api_key or os.getenv("TIDE_API_KEY") or self._required("TIDE_API_KEY")
        server = api_server or os.getenv("TIDE_API_SERVER") or NRP_DEFAULT_SERVER

        cfg = Configuration()
        cfg.host = server
        cfg.verify_ssl = True

        # K8s service account tokens are long JWTs; short hex keys use a
        # different prefix but we try Bearer for both since NRP accepts it.
        cfg.api_key["authorization"] = key
        cfg.api_key_prefix["authorization"] = "Bearer"

        return ApiClient(configuration=cfg)

    @staticmethod
    def _required(var: str) -> str:
        raise EnvironmentError(
            f"Required env var '{var}' is not set. "
            f"Add it to .env or export it in your shell."
        )

    def verify_connection(self) -> dict:
        """Attempt a lightweight API call and return server version info."""
        version_api = k8s_client.VersionApi(self._api_client)
        v = version_api.get_code()
        return {"git_version": v.git_version, "platform": v.platform}
