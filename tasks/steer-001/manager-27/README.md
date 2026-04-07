# Manager 27: Exp 27 — SGDR Warm-Restart Cosine Annealing

**Task:** steer-001 | **Status:** Queued

## Summary

Tests SGDR (warm-restart cosine annealing) as a drop-in replacement for Exp19's
single-cycle cosine schedule. Same total budget (300 soft steps), but split into
3 cycles of 100 steps each with LR reset to LR_MAX at cycle boundaries.
Hypothesis: periodic restarts allow the optimizer to escape local Voronoi basins.

## Worker

| Worker | Script | Results |
|--------|--------|---------|
| worker-1 | `sgdr_warmrestart.py` | `sgdr_warmrestart_results.json` (pending) |
