"""Batch runner: Exp45-46 on GPU 0 while Exp44 runs on GPU 1.

GPU 0 queue (sequential):
  Exp46 — HotFlip topk=200 + beam=2 from SOTA   (~37 min, very fast)
  Exp45 — seeds 19-24 at len=16                  (~8.6h)

Run Exp46 first since it's fast and directly tests the SOTA minimum.
Exp45 seeds 19-24 fills remaining GPU time.

Usage: python3 submit_batch_exp45_46.py
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from iterare.tools.tide_runner import run_on_tide, tide_available
from tide.client import TIDEClient

BASE   = Path(__file__).parent
PB_KEY = os.getenv("PUSHBULLET_API_KEY", "")

# GPU 0 queue: Exp46 first (fast), then Exp45
EXPERIMENTS = [
    {
        "name":        "Exp46",
        "script":      BASE / "manager-46/worker-1/hf_topk200.py",
        "out_dir":     BASE / "manager-46/worker-1",
        "remote_file": "steer001_hf_topk200.json",
        "local_file":  "hf_topk200_results.json",
        "gpu":         "0",
    },
    {
        "name":        "Exp45",
        "script":      BASE / "manager-45/worker-1/seeds19_24.py",
        "out_dir":     BASE / "manager-45/worker-1",
        "remote_file": "steer001_seeds19_24.json",
        "local_file":  "seeds19_24_results.json",
        "gpu":         "0",
    },
]


def main():
    import tempfile

    if not tide_available():
        print("ERROR: TIDE not configured")
        sys.exit(1)

    print("=== Batch runner: Exp45-46 (GPU 0) ===", flush=True)
    print("GPU 0: Exp46 (HF topk=200+beam) → Exp45 (seeds 19-24)", flush=True)

    client = TIDEClient()
    client.start_server(wait=True)

    for exp in EXPERIMENTS:
        name = exp["name"]
        gpu  = exp["gpu"]
        print(f"\n[GPU{gpu}] Starting {name}...", flush=True)

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
                print(f"[GPU{gpu}] {name} submission error: {e}", flush=True)
                continue

        print(f"[GPU{gpu}] {name} done: {result.status} ({result.elapsed_seconds:.0f}s)",
              flush=True)
        if result.error:
            print(f"[GPU{gpu}] {name} ERROR: {result.error[:500]}", flush=True)

        if result.status == "complete":
            try:
                local_path = exp["out_dir"] / exp["local_file"]
                local_path.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(exp["remote_file"], str(local_path))
                print(f"[GPU{gpu}] {name} saved to {local_path}", flush=True)
            except Exception as e:
                print(f"[GPU{gpu}] {name} download failed: {e}", flush=True)

    print("\n=== All experiments complete ===", flush=True)


if __name__ == "__main__":
    main()
