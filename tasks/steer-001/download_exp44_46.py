"""Download Exp44-46 results from TIDE."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from tide.client import TIDEClient

DOWNLOADS = [
    ("steer001_large_noise_restart.json", "manager-44/worker-1/large_noise_restart_results.json"),
    ("steer001_hf_topk200.json",          "manager-46/worker-1/hf_topk200_results.json"),
    ("steer001_seeds19_24.json",          "manager-45/worker-1/seeds19_24_results.json"),
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
