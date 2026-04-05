"""
Submit Exp 5 (prefix length, OOM-fixed) and Exp 7 (Gumbel-Softmax) to TIDE in parallel.
Also tries to download any existing Exp 6 results (it may have completed in a prior session).

Exp 5 → GPU 0, Exp 7 → GPU 1.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

EXP5_SCRIPT = Path(__file__).parent / "manager-5/worker-1/prefixlen.py"
EXP7_SCRIPT = Path(__file__).parent / "manager-7/worker-1/gumbel_softmax.py"
EXP5_OUT_DIR = Path(__file__).parent / "manager-5/worker-1"
EXP6_OUT_DIR = Path(__file__).parent / "manager-6/worker-1"
EXP7_OUT_DIR = Path(__file__).parent / "manager-7/worker-1"

GPU_PIN_PREFIX = "import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu}'\n"


def make_pinned(src, gpu, tmp_dir):
    code = GPU_PIN_PREFIX.format(gpu=gpu) + src.read_text()
    out = Path(tmp_dir) / f"{src.stem}_gpu{gpu}.py"
    out.write_text(code)
    return out


def try_download_exp6():
    """Try to grab Exp6 results if they exist from a prior run."""
    local_path = EXP6_OUT_DIR / "iterative_results.json"
    if local_path.exists():
        print("[Exp6] Results already downloaded locally, skipping.")
        return
    try:
        client = TIDEClient()
        client.start_server(wait=True)
        client.download_file("/home/jovyan/steer001_iterative.json", str(local_path))
        print(f"[Exp6] Downloaded existing results to {local_path}")
    except Exception as e:
        print(f"[Exp6] No existing results available ({e})")


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
        print(f"[{name}] ERROR: {result.error[:500]}", flush=True)

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

    # Try to recover Exp6 from a prior run
    try_download_exp6()

    results = {}

    t5 = threading.Thread(target=run_exp, args=(
        "Exp5-PrefixLen", EXP5_SCRIPT, 0, EXP5_OUT_DIR,
        "steer001_prefixlen.json", "prefixlen_results.json", results))
    t7 = threading.Thread(target=run_exp, args=(
        "Exp7-Gumbel", EXP7_SCRIPT, 1, EXP7_OUT_DIR,
        "steer001_gumbel.json", "gumbel_results.json", results))

    t5.start()
    t7.start()
    t5.join()
    t7.join()

    print("\n=== SUMMARY ===")
    for name, r in results.items():
        print(f"  {name}: {r.status} ({r.elapsed_seconds:.0f}s)")


if __name__ == "__main__":
    main()
