"""Batch runner: Exp48-51 across both GPUs.

GPU 0 queue (sequential):
  Exp51 — Best seeds 13,14,17,18,23 with topk=200   (~2.2h, fast)
  Exp48 — Seeds 0-6 re-run with topk=200             (~6h)

GPU 1 queue (sequential):
  Exp49 — SA from new SOTA (CE=0.5993), topk=200     (~1.8h)
  Exp50 — Warm restart × 5 from new SOTA, topk=200   (~1h)

All experiments exploit the Exp46 finding: topk=200 exposes better HotFlip
candidates than topk=50, which found the current new SOTA=0.5993.

Usage: python3 submit_batch_exp48_51.py
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
            "name":        "Exp51",
            "script":      BASE / "manager-51/worker-1/best_seeds_topk200.py",
            "out_dir":     BASE / "manager-51/worker-1",
            "remote_file": "steer001_best_seeds_topk200.json",
            "local_file":  "best_seeds_topk200_results.json",
            "gpu":         "0",
        },
        {
            "name":        "Exp48",
            "script":      BASE / "manager-48/worker-1/seeds0_6_topk200.py",
            "out_dir":     BASE / "manager-48/worker-1",
            "remote_file": "steer001_seeds0_6_topk200.json",
            "local_file":  "seeds0_6_topk200_results.json",
            "gpu":         "0",
        },
    ],
    "gpu1": [
        {
            "name":        "Exp49",
            "script":      BASE / "manager-49/worker-1/sa_from_new_sota.py",
            "out_dir":     BASE / "manager-49/worker-1",
            "remote_file": "steer001_sa_new_sota.json",
            "local_file":  "sa_new_sota_results.json",
            "gpu":         "1",
        },
        {
            "name":        "Exp50",
            "script":      BASE / "manager-50/worker-1/warm_restart_new_sota.py",
            "out_dir":     BASE / "manager-50/worker-1",
            "remote_file": "steer001_warm_restart_new_sota.json",
            "local_file":  "warm_restart_new_sota_results.json",
            "gpu":         "1",
        },
    ],
}


def run_queue(queue_name, experiments, client):
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

    client = TIDEClient()
    client.start_server(wait=True)

    print("=== Batch runner: Exp48-51 ===", flush=True)
    print("GPU 0: Exp51 (best seeds topk=200) → Exp48 (seeds 0-6 topk=200)", flush=True)
    print("GPU 1: Exp49 (SA from new SOTA) → Exp50 (warm restart new SOTA)", flush=True)

    t0 = threading.Thread(
        target=run_queue, args=("GPU0", EXPERIMENTS["gpu0"], client), daemon=True)
    t1 = threading.Thread(
        target=run_queue, args=("GPU1", EXPERIMENTS["gpu1"], client), daemon=True)

    t0.start()
    t1.start()
    t0.join()
    t1.join()

    print("=== All queues complete ===", flush=True)


if __name__ == "__main__":
    main()
