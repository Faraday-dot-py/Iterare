# Manager 23: Exp 23 — Held-out Suffix Generalization Evaluation

**Task:** steer-001 | **Status:** Complete

## Summary

Evaluates three prefix checkpoints (Exp19 SOTA, Exp16 λ=0, Exp11 baseline) on
12 in-sample suffixes and 20 held-out suffixes to measure generalization.
Addresses the research report's concern that CE on training suffixes may overfit.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `holdout_eval.py` | `holdout_eval_results.json` ✓ |
