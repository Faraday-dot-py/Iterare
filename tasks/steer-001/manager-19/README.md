# Manager 19: Exp 19 — ST + Cosine Annealing + Best-Prefix, PREFIX_LEN=16

**Task:** steer-001 | **Status:** Complete

## Summary

ST+cosine anneal+best-prefix with PREFIX_LEN=16 (doubled from baseline).
Result: **HotFlip CE = 0.6794 — NEW SOTA**, beating Exp16's 0.686.
More discrete tokens give HotFlip more degrees of freedom to optimize.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `st_long_prefix.py` | `long_prefix_results.json` |
