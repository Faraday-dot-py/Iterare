"""Submit Exp 28 (PREFIX_LEN=48 scaling) to TIDE.

Tests if the super-linear scaling trend from len=32 continues at len=48.
Timing estimate: ~3.7h (soft ~3600s + 30 HotFlip steps × ~320s each).

Usage: python3 submit_exp28.py [--gpu 0|1]  (default: gpu 0)
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP28_SCRIPT  = Path(__file__).parent / "manager-28/worker-1/scale_prefix_48.py"
EXP28_OUT_DIR = Path(__file__).parent / "manager-28/worker-1"

GPU = "0"
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
        pinned = Path(tmp) / f"scale_prefix_48_gpu{GPU}.py"
        pinned.write_text(ENV_INJ + GPU_PIN + EXP28_SCRIPT.read_text())

        print(f"[Exp28] Submitting PREFIX_LEN=48 scaling run on GPU {GPU}...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp28] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=86400,
            on_output=on_output,
        )

    print(f"[Exp28] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp28] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP28_OUT_DIR / "scale_prefix_48_results.json"
            client.download_file("steer001_scale48.json", str(local_path))
            print(f"[Exp28] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp28] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
