"""SSH-based script and code execution on a RunPod instance via paramiko.

Uploads a script via SFTP then executes it, streaming stdout/stderr in real time.
"""

import io
import select
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import paramiko


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    elapsed_seconds: float = 0.0

    @property
    def output(self) -> str:
        return self.stdout

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def error(self) -> Optional[str]:
        if not self.success:
            return self.stderr or f"Exit code {self.exit_code}"
        return None


def run_ssh(
    host: str,
    port: int,
    key_path: Path,
    command: str,
    timeout: int = 86400,
    on_output: Optional[Callable[[str], None]] = None,
    connect_timeout: int = 30,
    connect_retries: int = 10,
    retry_delay: int = 5,
) -> ExecutionResult:
    """
    Open an SSH connection, run command, stream output, return result.
    Retries the connection on failure (pod may still be booting).
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    for attempt in range(connect_retries):
        try:
            client.connect(
                hostname=host,
                port=port,
                username="root",
                key_filename=str(key_path),
                timeout=connect_timeout,
                banner_timeout=connect_timeout,
                auth_timeout=connect_timeout,
            )
            break
        except (paramiko.ssh_exception.NoValidConnectionsError,
                paramiko.ssh_exception.SSHException,
                OSError) as exc:
            if attempt == connect_retries - 1:
                raise RuntimeError(f"SSH connection failed after {connect_retries} attempts: {exc}") from exc
            time.sleep(retry_delay)

    result = ExecutionResult()
    start = time.time()

    try:
        transport = client.get_transport()
        channel = transport.open_session()
        channel.settimeout(timeout)
        channel.exec_command(command)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        while True:
            if time.time() - start > timeout:
                channel.close()
                result.exit_code = -1
                result.stderr = f"Timed out after {timeout}s"
                break

            readable, _, _ = select.select([channel], [], [], 1.0)
            if readable:
                if channel.recv_ready():
                    chunk = channel.recv(4096).decode("utf-8", errors="replace")
                    stdout_buf.write(chunk)
                    if on_output and chunk:
                        on_output(chunk)
                if channel.recv_stderr_ready():
                    chunk = channel.recv_stderr(4096).decode("utf-8", errors="replace")
                    stderr_buf.write(chunk)

            if channel.exit_status_ready():
                # Drain remaining output
                while channel.recv_ready():
                    chunk = channel.recv(4096).decode("utf-8", errors="replace")
                    stdout_buf.write(chunk)
                    if on_output and chunk:
                        on_output(chunk)
                while channel.recv_stderr_ready():
                    chunk = channel.recv_stderr(4096).decode("utf-8", errors="replace")
                    stderr_buf.write(chunk)
                result.exit_code = channel.recv_exit_status()
                break

        result.stdout = stdout_buf.getvalue()
        result.stderr = stderr_buf.getvalue()
    finally:
        client.close()

    result.elapsed_seconds = round(time.time() - start, 2)
    return result


def upload_file(
    host: str,
    port: int,
    key_path: Path,
    local_path: str,
    remote_path: str,
) -> None:
    """Upload a single file to the pod via SFTP."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username="root",
        key_filename=str(key_path),
        timeout=30,
    )
    try:
        sftp = client.open_sftp()
        # Ensure remote directory exists
        remote_dir = str(Path(remote_path).parent)
        if remote_dir and remote_dir != ".":
            try:
                sftp.makedirs(remote_dir)
            except Exception:
                _sftp_makedirs(sftp, remote_dir)
        sftp.put(local_path, remote_path)
        sftp.close()
    finally:
        client.close()


def download_file(
    host: str,
    port: int,
    key_path: Path,
    remote_path: str,
    local_path: str,
) -> None:
    """Download a single file from the pod via SFTP."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username="root",
        key_filename=str(key_path),
        timeout=30,
    )
    try:
        sftp = client.open_sftp()
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, local_path)
        sftp.close()
    finally:
        client.close()


def _sftp_makedirs(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    """Recursively create remote directories."""
    parts = Path(remote_dir).parts
    current = ""
    for part in parts:
        current = str(Path(current) / part) if current else part
        try:
            sftp.mkdir(current)
        except IOError:
            pass  # already exists
