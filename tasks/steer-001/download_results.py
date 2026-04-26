"""Download Exp16-20 results from TIDE."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from tide.client import TIDEClient

DOWNLOADS = [
    ("steer001_voronoi_margin.json",   "manager-16/worker-1/voronoi_margin_results.json"),
    ("steer001_multiseed_topk50.json", "manager-17/worker-1/multiseed_topk50_results.json"),
    ("steer001_topk_escalation.json",  "manager-18/worker-1/topk_escalation_results.json"),
    ("steer001_long_prefix.json",      "manager-19/worker-1/long_prefix_results.json"),
    ("steer001_multiseed_fp32.json",   "manager-20/worker-1/multiseed_fp32_results.json"),
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
