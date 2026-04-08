# Manager 28: Exp 28 — PREFIX_LEN=48 Scaling

**Task:** steer-001 | **Status:** Queued

## Summary

Continues the super-linear scaling trend: len=24→32 improved CE by 0.037 (Δ=0.037),
far exceeding the len=16→24 improvement (Δ=0.010). Tests whether len=48 continues
this acceleration. Uses reduced HotFlip budget (30 steps) to fit in time.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `scale_prefix_48.py` | `scale_prefix_48_results.json` (pending) |
