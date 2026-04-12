"""Download Exp41-44 results from TIDE."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from tide.client import TIDEClient

DOWNLOADS = [
    ("steer001_asst_keyword.json",        "manager-41/worker-1/asst_keyword_results.json"),
    ("steer001_seeds7_12.json",           "manager-42/worker-1/seeds7_12_results.json"),
    ("steer001_seeds13_18.json",          "manager-43/worker-1/seeds13_18_results.json"),
    ("steer001_large_noise_restart.json", "manager-44/worker-1/large_noise_restart_results.json"),
]

base = Path(__file__).parent

client = TIDEClient()
client.start_server(wait=True)

for remote, local_rel in DOWNLOADS:
    local = base / local_rel
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(remote, str(local))
        print(f"OK  {remote} -> {local_rel}")
    except Exception as e:
        print(f"ERR {remote}: {e}")
