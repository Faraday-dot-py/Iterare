"""Submit Exp 25 (PREFIX_LEN=32 scaling) to TIDE.

Tests if 4× the original prefix length (len=8→32) continues the CE improvement trend.
Timing estimate: ~5.5h (soft ~2400s + HotFlip 35 steps × ~498s each).

Usage: python3 submit_exp25.py [--gpu 0|1]  (default: gpu 1)
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP25_SCRIPT  = Path(__file__).parent / "manager-25/worker-1/scale_prefix_32.py"
EXP25_OUT_DIR = Path(__file__).parent / "manager-25/worker-1"

GPU = "1"
if "--gpu" in sys.argv:
    idx = sys.argv.index("--gpu")
    GPU = sys.argv[idx + 1]

GPU_PIN = f"import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '{GPU}'\n"
PB_KEY  = os.getenv("PUSHBULLET_API_KEY", "")
ENV_INJ = f"import os as _os; _os.environ['PUSHBULLET_API_KEY'] = {PB_KEY!r}\n"


def main():
    if not tide_available():
        print("ERROR: TIDE not configured")
        sys.exit(1)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pinned = Path(tmp) / f"scale_prefix_32_gpu{GPU}.py"
        pinned.write_text(ENV_INJ + GPU_PIN + EXP25_SCRIPT.read_text())

        print(f"[Exp25] Submitting PREFIX_LEN=32 scaling run on GPU {GPU}...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp25] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=86400,   # 24h client timeout — job itself ~5.5h
            on_output=on_output,
        )

    print(f"[Exp25] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp25] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP25_OUT_DIR / "scale_prefix_32_results.json"
            client.download_file("steer001_scale32.json", str(local_path))
            print(f"[Exp25] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp25] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
