"""
profile-001 TIDE submission script — runs locally.

Steps:
  1. Parse + anonymize messages locally (no raw text leaves this machine)
  2. gzip + Fernet-encrypt the anonymized JSON
  3. Upload encrypted data + code modules to TIDE
  4. Submit worker_tide.py with FERNET_KEY injected via env_inject (never written to disk)
  5. Download results.tar.gz after completion

Encryption key stored in .env as PROFILE_FERNET_KEY.
Generate once; subsequent runs reuse. Delete from .env to rotate.
"""

import os
import sys
import gzip
import json
import tarfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# ── Resolve roots ──────────────────────────────────────────────────────────────
ITERARE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ITERARE_ROOT / "code"))

from dotenv import load_dotenv
load_dotenv(ITERARE_ROOT / ".env")

from tools.message_profile.parsers import load_all
from tools.message_profile.anonymize import Anonymizer
from tools.tide.client import TIDEClient
from tools.tide.jobs import run_script

# ── Paths ──────────────────────────────────────────────────────────────────────
INSTAGRAM_MESSAGES_ROOT = Path.home() / "Research/MyMessages/instagram/your_instagram_activity/messages"
DISCORD_MESSAGES_ROOT   = Path.home() / "Research/MyMessages/discord/Messages"
INSTAGRAM_USER_NAME     = "Adam Webb"

RESULTS_LOCAL = ITERARE_ROOT / "tasks/profile-001/results_tide"
WORKER_SCRIPT = ITERARE_ROOT / "tasks/profile-001/worker_tide.py"
CODE_MODULES  = ITERARE_ROOT / "code/tools/message_profile"

REMOTE_BASE   = "profile_tide"
REMOTE_DATA   = f"{REMOTE_BASE}/messages.enc"
REMOTE_CODE   = f"{REMOTE_BASE}/code"
REMOTE_RESULT = f"{REMOTE_BASE}/results.tar.gz"


def get_or_generate_key() -> str:
    """Return PROFILE_FERNET_KEY from .env, or generate + append to .env."""
    key = os.getenv("PROFILE_FERNET_KEY", "")
    if key:
        return key
    from cryptography.fernet import Fernet
    key = Fernet.generate_key().decode()
    env_path = ITERARE_ROOT / ".env"
    with open(env_path, "a") as fh:
        fh.write(f"\nPROFILE_FERNET_KEY={key}\n")
    print(f"  Generated new Fernet key → appended to .env")
    return key


def encrypt_messages(messages: list[dict], fernet_key: str) -> bytes:
    from cryptography.fernet import Fernet
    f = Fernet(fernet_key.encode())
    serialized = json.dumps(messages, default=str).encode()
    compressed = gzip.compress(serialized, compresslevel=6)
    encrypted  = f.encrypt(compressed)
    return encrypted


def _ensure_dir_chain(client: TIDEClient, remote_dir: str) -> None:
    """Create all intermediate directories in remote_dir."""
    parts = remote_dir.strip("/").split("/")
    for i in range(1, len(parts) + 1):
        path = "/".join(parts[:i])
        try:
            payload = {"type": "directory", "format": "json", "content": []}
            client._session.put(f"{client.server_api}/contents/{path}", json=payload)
        except Exception:
            pass


def upload_code_modules(client: TIDEClient) -> None:
    """Upload code/tools/message_profile/ and stub __init__.py files."""
    # Create directory chain first
    for d in [
        f"{REMOTE_CODE}",
        f"{REMOTE_CODE}/tools",
        f"{REMOTE_CODE}/tools/message_profile",
    ]:
        _ensure_dir_chain(client, d)

    # Package stub __init__.py files
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write("")
        tmp_path = tmp.name
    for rel in ["tools/__init__.py", "tools/message_profile/__init__.py"]:
        client.upload_file(tmp_path, f"{REMOTE_CODE}/{rel}")
    os.unlink(tmp_path)

    # Actual modules
    for src in CODE_MODULES.glob("*.py"):
        remote_path = f"{REMOTE_CODE}/tools/message_profile/{src.name}"
        print(f"  Uploading {src.name} → {remote_path}")
        client.upload_file(str(src), remote_path)


def download_results(client: TIDEClient) -> None:
    RESULTS_LOCAL.mkdir(parents=True, exist_ok=True)
    tar_local = RESULTS_LOCAL / "results.tar.gz"
    print(f"\nDownloading results.tar.gz ...")
    client.download_file(REMOTE_RESULT, str(tar_local))
    print(f"  Saved to {tar_local} ({tar_local.stat().st_size / 1e6:.1f} MB)")

    with tarfile.open(str(tar_local), "r:gz") as tar:
        tar.extractall(str(RESULTS_LOCAL))
    print(f"  Extracted to {RESULTS_LOCAL}/")


def main():
    print("=" * 60)
    print("profile-001 TIDE Submission")
    print("=" * 60)

    # ── 1. Parse + anonymize ──────────────────────────────────────────────────
    print("\n[1/5] Parsing messages locally ...")
    messages = load_all(
        str(INSTAGRAM_MESSAGES_ROOT),
        str(DISCORD_MESSAGES_ROOT),
        instagram_user_name=INSTAGRAM_USER_NAME,
    )
    print(f"  {len(messages):,} messages loaded")

    print("[2/5] Anonymizing ...")
    anon = Anonymizer(user_name=INSTAGRAM_USER_NAME)
    anon_messages = anon.anonymize(messages)
    del messages
    print(f"  {len(anon.pseudonym_map())} unique names pseudonymized")

    # Print pseudonym map (stdout only, not written to disk)
    pmap = anon.pseudonym_map()
    print("\nPseudonym map (stdout only):")
    for real, pseudo in sorted(pmap.items(), key=lambda x: x[1]):
        print(f"  {pseudo:12s} → {real}")

    # ── 3. Encrypt ────────────────────────────────────────────────────────────
    print("\n[3/5] Encrypting messages ...")
    fernet_key = get_or_generate_key()
    # Serialize timestamps to ISO strings for JSON
    serial = []
    for m in anon_messages:
        row = dict(m)
        if isinstance(row.get("timestamp"), datetime):
            row["timestamp"] = row["timestamp"].isoformat()
        serial.append(row)
    encrypted = encrypt_messages(serial, fernet_key)
    print(f"  Encrypted payload: {len(encrypted) / 1e6:.1f} MB")
    del anon_messages, serial

    # ── 4. Connect + upload ───────────────────────────────────────────────────
    print("\n[4/5] Connecting to TIDE ...")
    client = TIDEClient()
    status = client.start_server(wait=True)
    print(f"  Server ready: {status['ready']}")

    print("  Uploading encrypted data ...")
    with tempfile.NamedTemporaryFile(suffix=".enc", delete=False) as tmp:
        tmp.write(encrypted)
        tmp_path = tmp.name
    del encrypted
    client.upload_file(tmp_path, REMOTE_DATA)
    os.unlink(tmp_path)
    print(f"  Uploaded → {REMOTE_DATA}")

    print("  Uploading code modules ...")
    upload_code_modules(client)

    # ── 5. Submit ─────────────────────────────────────────────────────────────
    print("\n[5/5] Submitting worker (this will stream output) ...")
    print("-" * 60)

    result = run_script(
        client,
        str(WORKER_SCRIPT),
        timeout=14400,  # 4 hours
        remote_dir=REMOTE_BASE,
        on_output=lambda text: print(text, end="", flush=True),
        cleanup=False,  # keep worker on TIDE for debugging
        env_inject={
            "FERNET_KEY":         fernet_key,
            "PUSHBULLET_API_KEY": os.getenv("PUSHBULLET_API_KEY", ""),
        },
    )

    print("-" * 60)
    print(f"\nStatus: {result.status} | Elapsed: {result.elapsed_seconds:.0f}s")

    if result.status == "failed":
        print(f"Error:\n{result.error}")
        sys.exit(1)

    # ── Download results ──────────────────────────────────────────────────────
    download_results(client)
    print("\nDone. Results at:", RESULTS_LOCAL)


if __name__ == "__main__":
    main()
