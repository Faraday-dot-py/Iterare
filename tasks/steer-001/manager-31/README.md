# Manager 31: Exp 31 — seed=2 at PREFIX_LEN=48

**Task:** steer-001 | **Status:** Queued

## Summary

Tests whether the seed=2 advantage (Δ=0.075 at len=16) is additive with the
length scaling advantage (Δ=0.037 per +8 tokens at len=32). If additive, could
push CE below 0.56 at PREFIX_LEN=48. Paired with Exp28 (seed=42, len=48) for comparison.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `seed2_len48.py` | `seed2_len48_results.json` (pending) |
