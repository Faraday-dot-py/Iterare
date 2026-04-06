"""Submit Exp 9 (gradient checkpointing baseline) to TIDE on GPU 0."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP9_SCRIPT = Path(__file__).parent / "manager-9/worker-1/ckpt_baseline.py"
EXP9_OUT_DIR = Path(__file__).parent / "manager-9/worker-1"

GPU_PIN = "import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n"


def main():
    if not tide_available():
        print("ERROR: TIDE not configured")
        sys.exit(1)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pinned = Path(tmp) / "ckpt_baseline_gpu0.py"
        pinned.write_text(GPU_PIN + EXP9_SCRIPT.read_text())

        print("[Exp9] Submitting on GPU 0...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp9] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=14400,
            on_output=on_output,
        )

    print(f"[Exp9] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp9] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP9_OUT_DIR / "ckpt_baseline_results.json"
            client.download_file("steer001_ckpt_baseline.json", str(local_path))
            print(f"[Exp9] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp9] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
