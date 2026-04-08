"""Batch runner: submit Exp28-33 to TIDE across both GPUs.

GPU 0 queue (sequential): Exp28 (len=48 seed=42) → Exp30 (multiseed len=32) → Exp32 (seed=2 len=32 60HF)
GPU 1 queue (sequential): Exp29 (len=32 80HF seed=42) → Exp31 (seed=2 len=48) → Exp33 (len=64 seed=42)

Both GPU queues run in parallel via threads.

Usage: python3 submit_batch_exp28_33.py
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
            "name": "Exp28",
            "script": BASE / "manager-28/worker-1/scale_prefix_48.py",
            "out_dir": BASE / "manager-28/worker-1",
            "remote_file": "steer001_scale48.json",
            "local_file": "scale_prefix_48_results.json",
            "gpu": "0",
        },
        {
            "name": "Exp30",
            "script": BASE / "manager-30/worker-1/multiseed_len32.py",
            "out_dir": BASE / "manager-30/worker-1",
            "remote_file": "steer001_multiseed32.json",
            "local_file": "multiseed_len32_results.json",
            "gpu": "0",
        },
        {
            "name": "Exp32",
            "script": BASE / "manager-32/worker-1/seed2_len32_full.py",
            "out_dir": BASE / "manager-32/worker-1",
            "remote_file": "steer001_seed2_len32_full.json",
            "local_file": "seed2_len32_full_results.json",
            "gpu": "0",
        },
    ],
    "gpu1": [
        {
            "name": "Exp29",
            "script": BASE / "manager-29/worker-1/scale_prefix_32_full.py",
            "out_dir": BASE / "manager-29/worker-1",
            "remote_file": "steer001_scale32_full.json",
            "local_file": "scale_prefix_32_full_results.json",
            "gpu": "1",
        },
        {
            "name": "Exp31",
            "script": BASE / "manager-31/worker-1/seed2_len48.py",
            "out_dir": BASE / "manager-31/worker-1",
            "remote_file": "steer001_seed2_len48.json",
            "local_file": "seed2_len48_results.json",
            "gpu": "1",
        },
        {
            "name": "Exp33",
            "script": BASE / "manager-33/worker-1/scale_prefix_64.py",
            "out_dir": BASE / "manager-33/worker-1",
            "remote_file": "steer001_scale64.json",
            "local_file": "scale_prefix_64_results.json",
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

    print("=== Batch runner: Exp28-33 ===", flush=True)
    print("GPU 0: Exp28 → Exp30 → Exp32", flush=True)
    print("GPU 1: Exp29 → Exp31 → Exp33", flush=True)
    print("Both queues running in parallel.", flush=True)

    t0_gpu0 = threading.Thread(
        target=run_queue, args=("GPU0", EXPERIMENTS["gpu0"]), daemon=True
    )
    t1_gpu1 = threading.Thread(
        target=run_queue, args=("GPU1", EXPERIMENTS["gpu1"]), daemon=True
    )

    t0_gpu0.start()
    t1_gpu1.start()

    t0_gpu0.join()
    t1_gpu1.join()

    print("=== All queues complete ===", flush=True)


if __name__ == "__main__":
    main()
