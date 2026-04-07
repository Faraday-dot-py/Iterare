"""Submit Exp 23 (held-out suffix generalization eval) to TIDE.

Evaluates Exp19 SOTA prefix, Exp16 λ=0 prefix, and Exp11 baseline on
12 training suffixes + 20 held-out suffixes to measure generalization.

Usage: python3 submit_exp23.py [--gpu 0|1]  (default: gpu 0)
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP23_SCRIPT  = Path(__file__).parent / "manager-23/worker-1/holdout_eval.py"
EXP23_OUT_DIR = Path(__file__).parent / "manager-23/worker-1"

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
        pinned = Path(tmp) / f"holdout_eval_gpu{GPU}.py"
        pinned.write_text(GPU_PIN + EXP23_SCRIPT.read_text())

        print(f"[Exp23] Submitting holdout eval (3 prefixes × 32 suffixes) on GPU {GPU}...",
              flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp23] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=3600,   # eval only, should finish in ~20min
            on_output=on_output,
            env_inject={"PUSHBULLET_API_KEY": os.getenv("PUSHBULLET_API_KEY", "")},
        )

    print(f"[Exp23] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp23] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP23_OUT_DIR / "holdout_eval_results.json"
            client.download_file("steer001_holdout_eval.json", str(local_path))
            print(f"[Exp23] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp23] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
