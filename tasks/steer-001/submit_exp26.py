"""Submit Exp 26 (multi-seed at PREFIX_LEN=16: seeds 0,1,2) to TIDE.

Tests whether seed=42 is privileged at len=16 or if other seeds benefit
equally from the extra discrete capacity.
Timing estimate: ~6h (3 seeds × ~7200s each: 1000s soft + 25×249s HotFlip).

Usage: python3 submit_exp26.py [--gpu 0|1]  (default: gpu 0)
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP26_SCRIPT  = Path(__file__).parent / "manager-26/worker-1/multiseed_len16.py"
EXP26_OUT_DIR = Path(__file__).parent / "manager-26/worker-1"

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
        pinned = Path(tmp) / f"multiseed_len16_gpu{GPU}.py"
        pinned.write_text(ENV_INJ + GPU_PIN + EXP26_SCRIPT.read_text())

        print(f"[Exp26] Submitting multi-seed len=16 (seeds 0,1,2) on GPU {GPU}...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp26] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=86400,   # 24h client timeout — job itself ~6h
            on_output=on_output,
        )

    print(f"[Exp26] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp26] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP26_OUT_DIR / "multiseed_len16_results.json"
            client.download_file("steer001_multiseed16.json", str(local_path))
            print(f"[Exp26] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp26] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
