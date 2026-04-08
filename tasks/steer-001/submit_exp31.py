"""Submit Exp 31 (seed=2, PREFIX_LEN=48) to TIDE.

Tests whether the seed=2 advantage (Δ=0.075 at len=16) is additive with
the length scaling (Δ=0.037 per +8 tokens at len=32). If additive, could
reach CE < 0.56 at len=48.
Timing estimate: ~3.2h (3600s soft + 25×320s HF).

Usage: python3 submit_exp31.py [--gpu 0|1]  (default: gpu 1)
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP31_SCRIPT  = Path(__file__).parent / "manager-31/worker-1/seed2_len48.py"
EXP31_OUT_DIR = Path(__file__).parent / "manager-31/worker-1"

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
        pinned = Path(tmp) / f"seed2_len48_gpu{GPU}.py"
        pinned.write_text(ENV_INJ + GPU_PIN + EXP31_SCRIPT.read_text())

        print(f"[Exp31] Submitting seed=2 PREFIX_LEN=48 on GPU {GPU}...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp31] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=86400,
            on_output=on_output,
        )

    print(f"[Exp31] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp31] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP31_OUT_DIR / "seed2_len48_results.json"
            client.download_file("steer001_seed2_len48.json", str(local_path))
            print(f"[Exp31] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp31] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
