"""Submit Exp 8 (multi-seed random restarts) to TIDE on GPU 0."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP8_SCRIPT = Path(__file__).parent / "manager-8/worker-1/multi_seed.py"
EXP8_OUT_DIR = Path(__file__).parent / "manager-8/worker-1"

GPU_PIN = "import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n"


def main():
    if not tide_available():
        print("ERROR: TIDE not configured")
        sys.exit(1)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pinned = Path(tmp) / "multi_seed_gpu0.py"
        pinned.write_text(GPU_PIN + EXP8_SCRIPT.read_text())

        print("[Exp8] Submitting on GPU 0...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp8] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=14400,
            on_output=on_output,
        )

    print(f"[Exp8] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp8] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP8_OUT_DIR / "multi_seed_results.json"
            client.download_file("steer001_multiseed.json", str(local_path))
            print(f"[Exp8] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp8] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
