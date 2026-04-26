"""Quick environment probe for TIDE — reports available packages and GPU."""
import sys, subprocess

print(f"Python: {sys.version}")
print(f"Prefix: {sys.prefix}\n")

pkgs = ["torch", "transformers", "numpy", "sentence_transformers", "accelerate", "huggingface_hub"]
for p in pkgs:
    try:
        m = __import__(p)
        ver = getattr(m, "__version__", "?")
        print(f"  {p}: {ver}")
    except ImportError:
        print(f"  {p}: MISSING")

try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                        "--format=csv,noheaders"], capture_output=True, text=True)
    print(f"\nGPU: {r.stdout.strip()}")
except Exception as e:
    print(f"\nGPU probe failed: {e}")

try:
    r = subprocess.run(["pip", "list", "--format=columns"], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if any(x in l.lower() for x in
             ["torch", "transform", "numpy", "accelerate", "hugging"])]
    print(f"\nRelevant pip packages:\n" + "\n".join(lines))
except Exception as e:
    print(f"pip list failed: {e}")
