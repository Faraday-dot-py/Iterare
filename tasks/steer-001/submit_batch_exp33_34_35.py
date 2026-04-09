"""Batch runner: submit Exp33-35 to TIDE across both GPUs.

GPU 0: Exp34 (multi-seed len=16, seeds 3-6) — ~4h
GPU 1: Exp33 (seed=42, len=64, 15HF) → Exp35 (seed=2, len=64, 20HF) — ~5.8h total

Exp32 (seed=2, len=32, 60HF) is skipped — Exp30 seed=2 already converged at
0.6402 with 35 HF steps, so 60 steps would not improve the result.

Usage: python3 submit_batch_exp33_34_35.py
"""

import os, sys, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

BASE = Path(__file__).parent

EXPERIMENTS = {
    "gpu0": [
        {
            "name": "Exp34",
            "script": BASE / "manager-34/worker-1/multiseed_len16_seeds3_6.py",
            "out_dir": BASE / "manager-34/worker-1",
            "remote_file": "steer001_multiseed16_seeds3_6.json",
            "local_file": "multiseed_len16_seeds3_6_results.json",
            "gpu": "0",
        },
    ],
    "gpu1": [
        {
            "name": "Exp33",
            "script": BASE / "manager-33/worker-1/scale_prefix_64.py",
            "out_dir": BASE / "manager-33/worker-1",
            "remote_file": "steer001_scale64.json",
            "local_file": "scale_prefix_64_results.json",
            "gpu": "1",
        },
        {
            "name": "Exp35",
            "script": BASE / "manager-35/worker-1/seed2_len64.py",
            "out_dir": BASE / "manager-35/worker-1",
            "remote_file": "steer001_seed2_len64.json",
            "local_file": "seed2_len64_results.json",
            "gpu": "1",
        },
    ],
}

PB_KEY = os.getenv("PUSHBULLET_API_KEY", "")


def run_queue(queue_name, experiments):
    """Run a list of experiments sequentially on a single GPU."""
    import tempfile
    for exp in experiments:
        name = exp["name"]
        gpu = exp["gpu"]
        print(f"[{queue_name}] Starting {name} on GPU {gpu}...", flush=True)

        gpu_pin = f"import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu}'\n"
        env_inj = f"import os as _os; _os.environ['PUSHBULLET_API_KEY'] = {PB_KEY!r}\n"

        with tempfile.TemporaryDirectory() as tmp:
            pinned = Path(tmp) / f"{exp['script'].stem}_gpu{gpu}.py"
            pinned.write_text(env_inj + gpu_pin + exp["script"].read_text())

            def on_output(text, _name=name):
                for line in text.splitlines():
                    print(f"[{_name}] {line}", flush=True)

            try:
                result = run_on_tide(
                    script_path=str(pinned),
                    timeout=86400,
                    on_output=on_output,
                )
            except Exception as e:
                print(f"[{queue_name}] {name} submission error: {e}", flush=True)
                continue

        print(f"[{queue_name}] {name} done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
        if result.error:
            print(f"[{queue_name}] {name} ERROR: {result.error[:500]}", flush=True)

        if result.status == "complete":
            try:
                client = TIDEClient()
                client.start_server(wait=True)
                local_path = exp["out_dir"] / exp["local_file"]
                client.download_file(exp["remote_file"], str(local_path))
                print(f"[{queue_name}] {name} saved to {local_path}", flush=True)
            except Exception as e:
                print(f"[{queue_name}] {name} download failed: {e}", flush=True)


def main():
    if not tide_available():
        print("ERROR: TIDE not configured (TIDE_API_KEY / TIDE_USERNAME not set)")
        sys.exit(1)

    print("=== Batch runner: Exp33-35 ===", flush=True)
    print("GPU 0: Exp34 (multi-seed len=16, seeds 3-6, ~4h)", flush=True)
    print("GPU 1: Exp33 (seed=42 len=64, ~3.2h) → Exp35 (seed=2 len=64, ~2.6h)", flush=True)
    print("Both queues running in parallel.", flush=True)

    t0 = threading.Thread(target=run_queue, args=("GPU0", EXPERIMENTS["gpu0"]), daemon=True)
    t1 = threading.Thread(target=run_queue, args=("GPU1", EXPERIMENTS["gpu1"]), daemon=True)

    t0.start()
    t1.start()
    t0.join()
    t1.join()

    print("=== All queues complete ===", flush=True)


if __name__ == "__main__":
    main()
