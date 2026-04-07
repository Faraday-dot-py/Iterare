"""Submit Exp 21 (Voronoi margin retry: λ=0.5 and λ=2.0) to TIDE.

Usage: python3 submit_exp21.py [--gpu 0|1]  (default: gpu 0)
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP21_SCRIPT  = Path(__file__).parent / "manager-21/worker-1/voronoi_margin_retry.py"
EXP21_OUT_DIR = Path(__file__).parent / "manager-21/worker-1"

GPU = "0"
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
        pinned = Path(tmp) / f"voronoi_margin_retry_gpu{GPU}.py"
        pinned.write_text(GPU_PIN + EXP21_SCRIPT.read_text())

        print(f"[Exp21] Submitting Voronoi margin retry (λ=0.5, λ=2.0) on GPU {GPU}...",
              flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp21] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=14400,
            on_output=on_output,
            env_inject={"PUSHBULLET_API_KEY": os.getenv("PUSHBULLET_API_KEY", "")},
        )

    print(f"[Exp21] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp21] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP21_OUT_DIR / "voronoi_margin_retry_results.json"
            client.download_file("steer001_voronoi_margin_retry.json", str(local_path))
            print(f"[Exp21] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp21] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
