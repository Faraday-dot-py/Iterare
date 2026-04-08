# Manager 33: Exp 33 — PREFIX_LEN=64 scaling

**Task:** steer-001 | **Status:** Queued

## Summary

Probes the next scaling step after len=32. The per-token CE improvement is accelerating
(Δ: 0.007 → 0.010 → 0.037 for +8 tokens each). Tests len=64 to see if acceleration
continues or saturates. Paired with Exp28 (len=48) to map the scaling curve.
Reduced HF budget (15 steps) to fit timing constraints at this prefix length.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `scale_prefix_64.py` | `scale_prefix_64_results.json` (pending) |
