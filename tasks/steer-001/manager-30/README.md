# Manager 30: Exp 30 — Multi-seed PREFIX_LEN=32 (seeds 1, 2)

**Task:** steer-001 | **Status:** Queued

## Summary

Exp26 revealed that seeds 1 and 2 dramatically outperform seed=42 at PREFIX_LEN=16
(seed=2: CE=0.6044, seed=1: CE=0.6236 vs seed=42 Exp19: CE=0.679).
This tests whether that advantage carries to PREFIX_LEN=32 — the current SOTA territory.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `multiseed_len32.py` | `multiseed_len32_results.json` (pending) |
