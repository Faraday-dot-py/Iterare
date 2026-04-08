"""Submit Exp 33 (PREFIX_LEN=64, seed=42) to TIDE.

Probes the next scaling step — the super-linear trend (Δ per +8 tokens:
0.007 → 0.010 → 0.037) is still accelerating at len=32. Tests len=64
to see if acceleration continues or saturates. Reduced HF budget (15 steps)
to fit within timing constraints.
Timing estimate: ~3.2h (5000s soft + 15×430s HF).

Usage: python3 submit_exp33.py [--gpu 0|1]  (default: gpu 1)
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP33_SCRIPT  = Path(__file__).parent / "manager-33/worker-1/scale_prefix_64.py"
EXP33_OUT_DIR = Path(__file__).parent / "manager-33/worker-1"

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
        pinned = Path(tmp) / f"scale_prefix_64_gpu{GPU}.py"
        pinned.write_text(ENV_INJ + GPU_PIN + EXP33_SCRIPT.read_text())

        print(f"[Exp33] Submitting PREFIX_LEN=64 scaling run on GPU {GPU}...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp33] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=86400,
            on_output=on_output,
        )

    print(f"[Exp33] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp33] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP33_OUT_DIR / "scale_prefix_64_results.json"
            client.download_file("steer001_scale64.json", str(local_path))
            print(f"[Exp33] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp33] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
