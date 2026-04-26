"""Submit environment probe to TIDE."""
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "tools"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from iterare.tools.tide_runner import run_on_tide, tide_available

WORKER = Path(__file__).parent / "manager-1/worker-1/env_probe.py"

if not tide_available():
    print("ERROR: TIDE not configured")
    sys.exit(1)

result = run_on_tide(
    script_path=str(WORKER),
    timeout=120,
    on_output=lambda t: print(t, end="", flush=True),
)
print(f"\nStatus: {result.status}  elapsed={result.elapsed_seconds:.0f}s")
if result.error:
    print(f"ERROR: {result.error[:2000]}")
