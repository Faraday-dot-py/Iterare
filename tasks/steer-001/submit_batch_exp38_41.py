"""Batch runner: submit Exp38-41 in pairs across both GPUs.

Round 1 (after Exp36/37 free GPUs):
  GPU 0: Exp38 — GCG discrete search, seeds {0,2}     (~4.4h)
  GPU 1: Exp39 — Assistant-turn prefix, seeds {0,1,2}  (~3.7h)

Round 2 (after Round 1 completes, sequential on each GPU):
  GPU 0: Exp40 — Keyword objective, seeds {0,1,2}      (~3.7h)
  GPU 1: Exp41 — Asst-turn + keyword combined, seeds {0,1,2}  (~3.7h)

Usage: python3 submit_batch_exp38_41.py
"""

import os, sys, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

BASE   = Path(__file__).parent
PB_KEY = os.getenv("PUSHBULLET_API_KEY", "")

# Each GPU queue runs its list sequentially
EXPERIMENTS = {
    "gpu0": [
        {
            "name":        "Exp38",
            "script":      BASE / "manager-38/worker-1/gcg_search.py",
            "out_dir":     BASE / "manager-38/worker-1",
            "remote_file": "steer001_gcg_search.json",
            "local_file":  "gcg_search_results.json",
            "gpu":         "0",
        },
        {
            "name":        "Exp40",
            "script":      BASE / "manager-40/worker-1/keyword_objective.py",
            "out_dir":     BASE / "manager-40/worker-1",
            "remote_file": "steer001_keyword_objective.json",
            "local_file":  "keyword_objective_results.json",
            "gpu":         "0",
        },
    ],
    "gpu1": [
        {
            "name":        "Exp39",
            "script":      BASE / "manager-39/worker-1/asst_prefix.py",
            "out_dir":     BASE / "manager-39/worker-1",
            "remote_file": "steer001_asst_prefix.json",
            "local_file":  "asst_prefix_results.json",
            "gpu":         "1",
        },
        {
            "name":        "Exp41",
            "script":      BASE / "manager-41/worker-1/asst_keyword.py",
            "out_dir":     BASE / "manager-41/worker-1",
            "remote_file": "steer001_asst_keyword.json",
            "local_file":  "asst_keyword_results.json",
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

        print(f"[{queue_name}] {name} done: {result.status} ({result.elapsed_seconds:.0f}s)", flush=True)
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

    print("=== Batch runner: Exp38-41 ===", flush=True)
    print("GPU 0: Exp38 (GCG) → Exp40 (keyword obj)", flush=True)
    print("GPU 1: Exp39 (asst prefix) → Exp41 (asst+keyword)", flush=True)

    t0 = threading.Thread(target=run_queue, args=("GPU0", EXPERIMENTS["gpu0"]), daemon=True)
    t1 = threading.Thread(target=run_queue, args=("GPU1", EXPERIMENTS["gpu1"]), daemon=True)

    t0.start()
    t1.start()
    t0.join()
    t1.join()

    print("=== All queues complete ===", flush=True)


if __name__ == "__main__":
    main()
