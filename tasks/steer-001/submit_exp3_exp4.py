"""
Submit Exp 3 (layer ablation) and Exp 4 (naturalness penalty) to TIDE in parallel.
Exp 3 runs on GPU 0, Exp 4 runs on GPU 1.
Results are downloaded and saved to their respective worker directories.
"""

import sys
import os
import json
import threading
from pathlib import Path

# Add iterare to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP3_SCRIPT = Path(__file__).parent / "manager-3/worker-1/layerablation.py"
EXP4_SCRIPT = Path(__file__).parent / "manager-4/worker-1/naturalness.py"

EXP3_OUT_DIR = Path(__file__).parent / "manager-3/worker-1"
EXP4_OUT_DIR = Path(__file__).parent / "manager-4/worker-1"

GPU_PIN_PREFIX = "import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu}'\n"


def make_pinned_script(src: Path, gpu: int, tmp_dir: Path) -> Path:
    """Return a temp script with CUDA_VISIBLE_DEVICES pinned."""
    code = GPU_PIN_PREFIX.format(gpu=gpu) + src.read_text()
    out = tmp_dir / f"{src.stem}_gpu{gpu}.py"
    out.write_text(code)
    return out


def run_exp(name, script_path, gpu, out_dir, remote_result_path, local_result_name, results):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pinned = make_pinned_script(script_path, gpu, Path(tmp))
        print(f"[{name}] Submitting on GPU {gpu}...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[{name}] {line}", flush=True)

        result = run_on_tide(
            script_path=str(pinned),
            timeout=12000,  # ~3.3h
            on_output=on_output,
        )
        results[name] = result
        print(f"[{name}] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
        if result.error:
            print(f"[{name}] ERROR: {result.error}", flush=True)

    # Download result file from TIDE
    if result.status == "complete":
        print(f"[{name}] Downloading {remote_result_path}...", flush=True)
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = out_dir / local_result_name
            client.download_file(remote_result_path, str(local_path))
            print(f"[{name}] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[{name}] Download failed: {e}", flush=True)


def main():
    if not tide_available():
        print("ERROR: TIDE not configured (TIDE_API_KEY / TIDE_USERNAME missing)")
        sys.exit(1)

    results = {}

    t3 = threading.Thread(
        target=run_exp,
        args=(
            "Exp3-LayerAblation",
            EXP3_SCRIPT,
            0,
            EXP3_OUT_DIR,
            "steer001_layerablation.json",  # remote path relative to /home/jovyan/
            "layerablation_results.json",
            results,
        ),
    )
    t4 = threading.Thread(
        target=run_exp,
        args=(
            "Exp4-Naturalness",
            EXP4_SCRIPT,
            1,
            EXP4_OUT_DIR,
            "steer001_naturalness.json",    # remote path relative to /home/jovyan/
            "naturalness_results.json",
            results,
        ),
    )

    t3.start()
    t4.start()
    t3.join()
    t4.join()

    print("\n=== SUMMARY ===")
    for name, r in results.items():
        print(f"  {name}: {r.status} ({r.elapsed_seconds:.0f}s)")
        if r.error:
            print(f"    Error: {r.error[:200]}")


if __name__ == "__main__":
    main()
