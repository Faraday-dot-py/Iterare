"""Submit Exp 30 (multi-seed len=32, seeds 1 and 2) to TIDE.

Exp26 showed seeds 1/2 dramatically outperform seed=42 at len=16 (0.6044 vs 0.679).
This tests whether that advantage holds at len=32 (current SOTA territory).
Timing estimate: ~5.1h (2 seeds × ~900s soft + 35×237s HF each).

Usage: python3 submit_exp30.py [--gpu 0|1]  (default: gpu 0)
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP30_SCRIPT  = Path(__file__).parent / "manager-30/worker-1/multiseed_len32.py"
EXP30_OUT_DIR = Path(__file__).parent / "manager-30/worker-1"

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
        pinned = Path(tmp) / f"multiseed_len32_gpu{GPU}.py"
        pinned.write_text(ENV_INJ + GPU_PIN + EXP30_SCRIPT.read_text())

        print(f"[Exp30] Submitting multi-seed len=32 (seeds 1,2) on GPU {GPU}...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp30] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=86400,
            on_output=on_output,
        )

    print(f"[Exp30] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp30] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP30_OUT_DIR / "multiseed_len32_results.json"
            client.download_file("steer001_multiseed32.json", str(local_path))
            print(f"[Exp30] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp30] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
