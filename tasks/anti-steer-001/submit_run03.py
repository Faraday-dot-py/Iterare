"""Submit anti-steer-001 run03 to TIDE.

Run03: Qwen2.5-3B-Instruct, 30 tasks, 5 ideas, all conditions + alpha sweep.
Estimated time: ~30min on L40.

Usage: python3 tasks/anti-steer-001/submit_run03.py [--gpu 0]
"""

import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

WORKER = Path(__file__).parent / "manager-1/worker-1/tide_run03.py"
OUT_DIR = Path(__file__).parent / "manager-1/worker-1"
REMOTE_OUT = "anti_steer001_run03.json"

GPU = "0"
if "--gpu" in sys.argv:
    GPU = sys.argv[sys.argv.index("--gpu") + 1]

PB_KEY  = os.getenv("PUSHBULLET_API_KEY", "")
GPU_PIN = f"import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '{GPU}'\n"
ENV_INJ = f"import os as _os; _os.environ['PUSHBULLET_API_KEY'] = {PB_KEY!r}\n"


def main():
    if not tide_available():
        print("ERROR: TIDE not configured (set TIDE_API_KEY and TIDE_USERNAME)")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        pinned = Path(tmp) / "tide_run03.py"
        pinned.write_text(ENV_INJ + GPU_PIN + WORKER.read_text())

        print(f"[run03] Submitting to TIDE GPU {GPU}...", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=86400,
            on_output=lambda t: print(t, end="", flush=True),
        )

    print(f"\n[run03] Status: {result.status}  elapsed={result.elapsed_seconds:.0f}s")
    if result.error:
        print(f"[run03] ERROR: {result.error[:500]}")

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = OUT_DIR / "run03_results.json"
            client.download_file(REMOTE_OUT, str(local_path))
            print(f"[run03] Saved to {local_path}")
        except Exception as e:
            print(f"[run03] Download failed: {e}")


if __name__ == "__main__":
    main()
