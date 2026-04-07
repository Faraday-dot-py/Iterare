# Manager 20: Exp 20 — Multi-Seed fp32-Sims ST + TOPK=50 (Seeds 5-9)

**Task:** steer-001 | **Status:** Complete

## Summary

fp32-sims ST + HotFlip TOPK=50 on seeds 5-9.
Best: **0.7178** (seed=5). No improvement over seeds 0-4; seeds 5-9 are worse initializations.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `st_multiseed_fp32.py` | `multiseed_fp32_results.json` |
