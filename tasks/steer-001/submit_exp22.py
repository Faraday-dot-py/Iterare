"""Submit Exp 22 (VQ commitment loss: β=0.1, 0.5, 2.0) to TIDE.

Usage: python3 submit_exp22.py [--gpu 0|1]  (default: gpu 1)
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP22_SCRIPT  = Path(__file__).parent / "manager-22/worker-1/vq_commitment.py"
EXP22_OUT_DIR = Path(__file__).parent / "manager-22/worker-1"

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
        pinned = Path(tmp) / f"vq_commitment_gpu{GPU}.py"
        pinned.write_text(GPU_PIN + EXP22_SCRIPT.read_text())

        print(f"[Exp22] Submitting VQ commitment loss (β=0.1, 0.5, 2.0) on GPU {GPU}...",
              flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[Exp22] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=14400,
            on_output=on_output,
            env_inject={"PUSHBULLET_API_KEY": os.getenv("PUSHBULLET_API_KEY", "")},
        )

    print(f"[Exp22] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[Exp22] ERROR: {result.error[:500]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = EXP22_OUT_DIR / "vq_commitment_results.json"
            client.download_file("steer001_vq_commitment.json", str(local_path))
            print(f"[Exp22] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[Exp22] Download failed: {e}", flush=True)


if __name__ == "__main__":
    main()
