"""Combined batch runner: Exp36-41 across both GPUs.

GPU 0 queue (sequential):
  Exp36 — SA escape from SOTA              (~1h)
  Exp38 — GCG discrete search              (~4.4h)
  Exp40 — Keyword probability objective    (~3.7h)

GPU 1 queue (sequential):
  Exp37 — Warm-restart soft opt × 3        (~4.3h)
  Exp39 — Assistant-turn prefix placement  (~3.7h)
  Exp41 — Asst-turn + keyword combined     (~3.7h)

Total wallclock: ~9.1h (GPU 0) / ~11.7h (GPU 1)

Usage: python3 submit_batch_exp36_41.py
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
            "name":        "Exp36",
            "script":      BASE / "manager-36/worker-1/sa_escape.py",
            "out_dir":     BASE / "manager-36/worker-1",
            "remote_file": "steer001_sa_escape.json",
            "local_file":  "sa_escape_results.json",
            "gpu":         "0",
        },
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
            "name":        "Exp37",
            "script":      BASE / "manager-37/worker-1/warm_restart.py",
            "out_dir":     BASE / "manager-37/worker-1",
            "remote_file": "steer001_warm_restart.json",
            "local_file":  "warm_restart_results.json",
            "gpu":         "1",
        },
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

    print("=== Batch runner: Exp36-41 ===", flush=True)
    print("GPU 0: Exp36 (SA) → Exp38 (GCG) → Exp40 (keyword obj)", flush=True)
    print("GPU 1: Exp37 (warm restart) → Exp39 (asst prefix) → Exp41 (asst+keyword)", flush=True)

    t0 = threading.Thread(target=run_queue, args=("GPU0", EXPERIMENTS["gpu0"]), daemon=True)
    t1 = threading.Thread(target=run_queue, args=("GPU1", EXPERIMENTS["gpu1"]), daemon=True)

    t0.start()
    t1.start()
    t0.join()
    t1.join()

    print("=== All queues complete ===", flush=True)


if __name__ == "__main__":
    main()
