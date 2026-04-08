# Manager 24: Exp 24 — PREFIX_LEN=24 Scaling

**Task:** steer-001 | **Status:** Complete

## Summary

Direct scaling test: extend Exp19's method (ST+cosine+best-prefix) to PREFIX_LEN=24.
Exp19 showed len=8→16 improved CE 0.686→0.679. Hypothesis: len=24 continues the trend.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `scale_prefix_24.py` | `scale_prefix_24_results.json` ✓ |
