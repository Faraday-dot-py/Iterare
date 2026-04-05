"""
Submit Exp 5 (prefix length) and Exp 6 (iterative projection) to TIDE in parallel.
Exp 5 runs on GPU 0, Exp 6 runs on GPU 1.
Run this after Exp3/Exp4 complete (GPUs freed).
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP5_SCRIPT = Path(__file__).parent / "manager-5/worker-1/prefixlen.py"
EXP6_SCRIPT = Path(__file__).parent / "manager-6/worker-1/iterative_projection.py"
EXP5_OUT_DIR = Path(__file__).parent / "manager-5/worker-1"
EXP6_OUT_DIR = Path(__file__).parent / "manager-6/worker-1"

GPU_PIN_PREFIX = "import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu}'\n"


def make_pinned(src, gpu, tmp_dir):
    code = GPU_PIN_PREFIX.format(gpu=gpu) + src.read_text()
    out = Path(tmp_dir) / f"{src.stem}_gpu{gpu}.py"
    out.write_text(code)
    return out


def run_exp(name, script_path, gpu, out_dir, remote_result, local_result, results):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pinned = make_pinned(script_path, gpu, tmp)
        print(f"[{name}] Submitting on GPU {gpu}...", flush=True)

        def on_output(text):
            for line in text.splitlines():
                print(f"[{name}] {line}", flush=True)

        result = run_on_tide(script_path=str(pinned), timeout=14400, on_output=on_output)
        results[name] = result

    print(f"[{name}] Done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
    if result.error:
        print(f"[{name}] ERROR: {result.error[:300]}", flush=True)

    if result.status == "complete":
        try:
            client = TIDEClient()
            client.start_server(wait=True)
            local_path = out_dir / local_result
            client.download_file(remote_result, str(local_path))
            print(f"[{name}] Saved to {local_path}", flush=True)
        except Exception as e:
            print(f"[{name}] Download failed: {e}", flush=True)


def main():
    if not tide_available():
        print("ERROR: TIDE not configured")
        sys.exit(1)

    results = {}

    t5 = threading.Thread(target=run_exp, args=(
        "Exp5-PrefixLen", EXP5_SCRIPT, 0, EXP5_OUT_DIR,
        "steer001_prefixlen.json", "prefixlen_results.json", results))
    t6 = threading.Thread(target=run_exp, args=(
        "Exp6-Iterative", EXP6_SCRIPT, 1, EXP6_OUT_DIR,
        "steer001_iterative.json", "iterative_results.json", results))

    t5.start()
    t6.start()
    t5.join()
    t6.join()

    print("\n=== SUMMARY ===")
    for name, r in results.items():
        print(f"  {name}: {r.status} ({r.elapsed_seconds:.0f}s)")


if __name__ == "__main__":
    main()
