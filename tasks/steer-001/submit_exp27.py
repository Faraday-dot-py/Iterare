"""Submit Exp 27 (SGDR warm-restart cosine annealing at PREFIX_LEN=16) to TIDE.

Tests whether 3× warm-restart cosine cycles (vs single cycle in Exp19) allow
ST optimization to find better discrete attractors.
Timing estimate: ~5.9h (soft ~1200s + HotFlip 80 steps × ~249s each).

Usage: python3 submit_exp27.py [--gpu 0|1]  (default: gpu 1)
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP27_SCRIPT  = Path(__file__).parent / "manager-27/worker-1/sgdr_warmrestart.py"
EXP27_OUT_DIR = Path(__file__).parent / "manager-27/worker-1"

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
        pinned = Path(tmp) / f"sgdr_warmrestart_gpu{GPU}.py"
        pinned.write_text(ENV_INJ + GPU_PIN + EXP27_SCRIPT.read_text())

        print(f"[Exp27] Submitting SGDR warm-restart at PREFIX_LEN=16 on GPU {GPU}...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp27] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=86400,   # 24h client timeout — job itself ~5.9h
            on_output=on_output,
        )

    print(f"[Exp27] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp27] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP27_OUT_DIR / "sgdr_warmrestart_results.json"
            client.download_file("steer001_sgdr.json", str(local_path))
            print(f"[Exp27] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp27] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
