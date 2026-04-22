"""Resume TIDE submission — skips parse/encrypt/data upload (already done)."""
import os, sys
from pathlib import Path

ITERARE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ITERARE_ROOT / "code"))

from dotenv import load_dotenv
load_dotenv(ITERARE_ROOT / ".env")

from tools.tide.client import TIDEClient
from tools.tide.jobs import run_script

WORKER_SCRIPT = ITERARE_ROOT / "tasks/profile-001/worker_tide.py"
CODE_MODULES  = ITERARE_ROOT / "code/tools/message_profile"
RESULTS_LOCAL = ITERARE_ROOT / "tasks/profile-001/results_tide"
REMOTE_BASE   = "profile_tide"
REMOTE_CODE   = f"{REMOTE_BASE}/code"
REMOTE_RESULT = f"{REMOTE_BASE}/results.tar.gz"

import tarfile, tempfile


def _ensure_dir_chain(client, remote_dir):
    parts = remote_dir.strip("/").split("/")
    for i in range(1, len(parts) + 1):
        path = "/".join(parts[:i])
        try:
            payload = {"type": "directory", "format": "json", "content": []}
            client._session.put(f"{client.server_api}/contents/{path}", json=payload)
        except Exception:
            pass


def upload_code_modules(client):
    for d in [REMOTE_CODE, f"{REMOTE_CODE}/tools", f"{REMOTE_CODE}/tools/message_profile"]:
        _ensure_dir_chain(client, d)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write("")
        tmp_path = tmp.name
    for rel in ["tools/__init__.py", "tools/message_profile/__init__.py"]:
        client.upload_file(tmp_path, f"{REMOTE_CODE}/{rel}")
    os.unlink(tmp_path)
    for src in CODE_MODULES.glob("*.py"):
        rp = f"{REMOTE_CODE}/tools/message_profile/{src.name}"
        print(f"  {src.name}")
        client.upload_file(str(src), rp)


def download_results(client):
    RESULTS_LOCAL.mkdir(parents=True, exist_ok=True)
    tar_local = RESULTS_LOCAL / "results.tar.gz"
    print("Downloading results.tar.gz ...")
    client.download_file(REMOTE_RESULT, str(tar_local))
    print(f"  {tar_local.stat().st_size / 1e6:.1f} MB")
    with tarfile.open(str(tar_local), "r:gz") as tar:
        tar.extractall(str(RESULTS_LOCAL))
    print(f"  Extracted to {RESULTS_LOCAL}/")


client = TIDEClient()
client.start_server(wait=True)
print("Server ready")

print("Uploading code modules ...")
upload_code_modules(client)
print("Done uploading modules")

fernet_key = os.getenv("PROFILE_FERNET_KEY", "")
if not fernet_key:
    raise RuntimeError("PROFILE_FERNET_KEY not in .env")

print("Submitting worker ...")
result = run_script(
    client,
    str(WORKER_SCRIPT),
    timeout=14400,
    remote_dir=REMOTE_BASE,
    on_output=lambda text: print(text, end="", flush=True),
    cleanup=False,
    env_inject={
        "FERNET_KEY":         fernet_key,
        "PUSHBULLET_API_KEY": os.getenv("PUSHBULLET_API_KEY", ""),
    },
)

print(f"\nStatus: {result.status} | Elapsed: {result.elapsed_seconds:.0f}s")
if result.status == "failed":
    print("Error:", result.error)
    sys.exit(1)

download_results(client)
print("Done. Results at:", RESULTS_LOCAL)
