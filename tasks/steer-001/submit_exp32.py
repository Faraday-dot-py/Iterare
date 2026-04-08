"""Submit Exp 32 (seed=2, PREFIX_LEN=32, 60 HotFlip steps) to TIDE.

Best seed (2) at the SOTA prefix length (32) with a full HotFlip budget.
Provides a clean comparison to Exp29 (seed=42, len=32, 80 HF steps) to
isolate the seed=2 advantage at fixed length and near-equal HF budget.
Timing estimate: ~4.1h (700s soft + 60×237s HF).

Usage: python3 submit_exp32.py [--gpu 0|1]  (default: gpu 0)
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP32_SCRIPT  = Path(__file__).parent / "manager-32/worker-1/seed2_len32_full.py"
EXP32_OUT_DIR = Path(__file__).parent / "manager-32/worker-1"

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
        pinned = Path(tmp) / f"seed2_len32_full_gpu{GPU}.py"
        pinned.write_text(ENV_INJ + GPU_PIN + EXP32_SCRIPT.read_text())

        print(f"[Exp32] Submitting seed=2 PREFIX_LEN=32 60HF-steps on GPU {GPU}...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp32] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=86400,
            on_output=on_output,
        )

    print(f"[Exp32] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp32] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP32_OUT_DIR / "seed2_len32_full_results.json"
            client.download_file("steer001_seed2_len32_full.json", str(local_path))
            print(f"[Exp32] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp32] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
