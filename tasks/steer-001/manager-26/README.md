# Manager 26: Exp 26 — Multi-seed at PREFIX_LEN=16

**Task:** steer-001 | **Status:** Queued

## Summary

Tests whether seed=42 is still privileged at PREFIX_LEN=16. Runs seeds 0, 1, 2
with the same ST+cosine+best-prefix method. At len=8, seeds 0-4 underperformed
seed=42 (best 0.694 vs 0.679). This experiment checks if that gap persists at len=16.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `multiseed_len16.py` | `multiseed_len16_results.json` (pending) |
