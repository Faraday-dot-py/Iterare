# Manager 17: Exp 17 — Multi-Seed ST+Anneal+Best-Prefix + TOPK=50

**Task:** steer-001 | **Status:** Complete

## Summary

Five independent ST+cosine-anneal+best-prefix seeds with HotFlip TOPK=50.
Best result: **0.6942** (seed=0), no improvement over Exp16's 0.686 SOTA.
High inter-seed variance (range 0.078) confirms the method is noisy.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `st_multiseed_topk50.py` | `multiseed_topk50_results.json` |
