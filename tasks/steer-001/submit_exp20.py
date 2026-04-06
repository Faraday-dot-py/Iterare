"""Submit Exp 20 (multi-seed ST fp32 sims, seeds 5-9) to TIDE.

Usage: python3 submit_exp20.py [--gpu 0|1]  (default: gpu 1)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP20_SCRIPT = Path(__file__).parent / "manager-20/worker-1/st_multiseed_fp32.py"
EXP20_OUT_DIR = Path(__file__).parent / "manager-20/worker-1"

GPU = "1"
if "--gpu" in sys.argv:
    idx = sys.argv.index("--gpu")
    GPU = sys.argv[idx + 1]

GPU_PIN = f"import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '{GPU}'\n"


def main():
    if not tide_available():
        print("ERROR: TIDE not configured")
        sys.exit(1)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pinned = Path(tmp) / f"st_multiseed_fp32_gpu{GPU}.py"
        pinned.write_text(GPU_PIN + EXP20_SCRIPT.read_text())

        print(f"[Exp20] Submitting multi-seed fp32 ST (seeds 5-9) on GPU {GPU}...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp20] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=14400,
            on_output=on_output,
        )

    print(f"[Exp20] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp20] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP20_OUT_DIR / "multiseed_fp32_results.json"
            client.download_file("steer001_multiseed_fp32.json", str(local_path))
            print(f"[Exp20] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp20] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
