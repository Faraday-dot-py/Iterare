"""Batch runner: Exp47 (GPU 0) and Exp44 re-run (GPU 1).

GPU 0: Exp47 — HF topk=500 from new SOTA CE=0.5993   (~88 min)
GPU 1: Exp44 — large-noise warm restart (re-run, no prior output found)

Exp44 re-run: the original batch (submit_batch_exp42_44.py) appears to have
lost Exp44 — no result or checkpoint found on TIDE. This re-runs it directly.

Usage: python3 submit_batch_exp44_47.py
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
            "name":        "Exp47",
            "script":      BASE / "manager-47/worker-1/hf_topk500.py",
            "out_dir":     BASE / "manager-47/worker-1",
            "remote_file": "steer001_hf_topk500.json",
            "local_file":  "hf_topk500_results.json",
            "gpu":         "0",
        },
    ],
    "gpu1": [
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

    print("=== Batch runner: Exp47 (GPU 0) + Exp44 re-run (GPU 1) ===", flush=True)

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
