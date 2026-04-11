"""Batch runner: Exp42-44 across both GPUs.

GPU 0: Exp42 — seeds 7-12 at len=16    (~8.6h)
GPU 1: Exp43 — seeds 13-18 at len=16   (~8.6h)
       Exp44 — large-noise warm restart  (~7.2h, runs after Exp43)

Total wallclock: ~8.6h (GPU 0) / ~15.8h (GPU 1)

Usage: python3 submit_batch_exp42_44.py
"""

import os, sys, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

BASE   = Path(__file__).parent
PB_KEY = os.getenv("PUSHBULLET_API_KEY", "")

EXPERIMENTS = {
    "gpu0": [
        {
            "name":        "Exp42",
            "script":      BASE / "manager-42/worker-1/seeds7_12.py",
            "out_dir":     BASE / "manager-42/worker-1",
            "remote_file": "steer001_seeds7_12.json",
            "local_file":  "seeds7_12_results.json",
            "gpu":         "0",
        },
    ],
    "gpu1": [
        {
            "name":        "Exp43",
            "script":      BASE / "manager-43/worker-1/seeds13_18.py",
            "out_dir":     BASE / "manager-43/worker-1",
            "remote_file": "steer001_seeds13_18.json",
            "local_file":  "seeds13_18_results.json",
            "gpu":         "1",
        },
        {
            "name":        "Exp44",
            "script":      BASE / "manager-44/worker-1/large_noise_restart.py",
            "out_dir":     BASE / "manager-44/worker-1",
            "remote_file": "steer001_large_noise_restart.json",
            "local_file":  "large_noise_restart_results.json",
            "gpu":         "1",
        },
    ],
}


def run_queue(queue_name, experiments):
    import tempfile
    for exp in experiments:
        name = exp["name"]
        gpu  = exp["gpu"]
        print(f"[{queue_name}] Starting {name} on GPU {gpu}...", flush=True)

        env_inj = f"import os as _os; _os.environ['PUSHBULLET_API_KEY'] = {PB_KEY!r}\n"
        gpu_pin = f"import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu}'\n"

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

        print(f"[{queue_name}] {name} done: {result.status} ({result.elapsed_seconds:.0f}s)",
              flush=True)
        if result.error:
            print(f"[{queue_name}] {name} ERROR: {result.error[:500]}", flush=True)

        if result.status == "complete":
            try:
                client = TIDEClient()
                client.start_server(wait=True)
                local_path = exp["out_dir"] / exp["local_file"]
                local_path.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(exp["remote_file"], str(local_path))
                print(f"[{queue_name}] {name} saved to {local_path}", flush=True)
            except Exception as e:
                print(f"[{queue_name}] {name} download failed: {e}", flush=True)


def main():
    if not tide_available():
        print("ERROR: TIDE not configured")
        sys.exit(1)

    print("=== Batch runner: Exp42-44 ===", flush=True)
    print("GPU 0: Exp42 (seeds 7-12)", flush=True)
    print("GPU 1: Exp43 (seeds 13-18) → Exp44 (large-noise restart)", flush=True)

    t0 = threading.Thread(target=run_queue, args=("GPU0", EXPERIMENTS["gpu0"]), daemon=True)
    t1 = threading.Thread(target=run_queue, args=("GPU1", EXPERIMENTS["gpu1"]), daemon=True)

    t0.start()
    t1.start()
    t0.join()
    t1.join()

    print("=== All queues complete ===", flush=True)


if __name__ == "__main__":
    main()
