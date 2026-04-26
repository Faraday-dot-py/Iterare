"""Download Exp33-35 results from TIDE."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from tide.client import TIDEClient

DOWNLOADS = [
    ("steer001_scale64.json",                "manager-33/worker-1/scale_prefix_64_results.json"),
    ("steer001_multiseed16_seeds3_6.json",   "manager-34/worker-1/multiseed_len16_seeds3_6_results.json"),
    ("steer001_seed2_len64.json",            "manager-35/worker-1/seed2_len64_results.json"),
]

base = Path(__file__).parent

client = TIDEClient()
client.start_server(wait=True)

for remote, local_rel in DOWNLOADS:
    local = base / local_rel
    try:
        client.download_file(remote, str(local))
        print(f"OK  {remote} -> {local_rel}")
    except Exception as e:
        print(f"ERR {remote}: {e}")
