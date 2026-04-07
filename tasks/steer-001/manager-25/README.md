# Manager 25: Exp 25 — PREFIX_LEN=32 Scaling

**Task:** steer-001 | **Status:** Queued

## Summary

Aggressive scaling test: extend Exp19's method to PREFIX_LEN=32 (2× Exp19, 4× baseline).
Runs in parallel with Exp24 on GPU 1. Tests whether the scaling trend continues
or hits diminishing returns beyond 24 tokens.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `scale_prefix_32.py` | `scale_prefix_32_results.json` (pending) |
