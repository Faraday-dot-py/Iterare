"""Resubmit fixed Exp3 (layer ablation) on GPU 0."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP3_SCRIPT = Path(__file__).parent / "manager-3/worker-1/layerablation.py"
EXP3_OUT_DIR = Path(__file__).parent / "manager-3/worker-1"

GPU_PIN = "import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n"


def main():
    if not tide_available():
        print("ERROR: TIDE not configured")
        sys.exit(1)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pinned = Path(tmp) / "layerablation_gpu0.py"
        pinned.write_text(GPU_PIN + EXP3_SCRIPT.read_text())

        print("[Exp3] Submitting on GPU 0...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp3] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=12000,
            on_output=on_output,
        )

    print(f"[Exp3] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp3] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP3_OUT_DIR / "layerablation_results.json"
            client.download_file("steer001_layerablation.json", str(local_path))
            print(f"[Exp3] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp3] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
