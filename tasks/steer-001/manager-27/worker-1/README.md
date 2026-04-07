# Exp 27: SGDR Warm-Restart Cosine Annealing at PREFIX_LEN=16, seed=42

**Status:** Queued | **Results:** Pending

## Method

SGDR (Loshchilov & Hutter 2016) applied to ST soft optimization:
- 3 cosine cycles × 100 steps = 300 total soft steps (identical budget to Exp19)
- Each cycle: LR_MAX=0.01 → LR_MIN=0.001 (same range as Exp19)
- Adam optimizer state reset at each cycle boundary (ensures clean restart)
- Global best-prefix tracking across all cycles
- HotFlip: 80 steps, TOPK=50 (same as Exp19 for fair comparison)

Key difference from Exp19: instead of one slow LR decay, we take three "attempts"
at finding a good trajectory. The LR jump at each cycle boundary (0.001→0.01)
allows the soft prefix to move 10× further in a single step, potentially
escaping a Voronoi cell pattern that the first cycle settled into.

## Research Question

Does the ST optimizer benefit from periodic LR restarts?

The single-cycle schedule (Exp19) decays LR from 0.01→0.001 linearly in log space.
By step 200/300, LR=0.0015 — small moves only. If the optimizer is in a suboptimal
basin at step 100, it cannot escape. SGDR restores LR=0.01 at step 100 and 200,
allowing larger moves.

Expected outcomes:
- **SGDR better (CE < 0.679)**: Warm restarts allow meaningful basin-hopping.
  → Future soft opt should use SGDR or longer warm-restart cycles.
- **SGDR similar (CE ≈ 0.679 ± 0.003)**: LR schedule is not the bottleneck.
  → The quality of discrete attractors found is determined earlier in training.
- **SGDR worse (CE > 0.682)**: The smooth decay is important; restarts disrupt
  a well-converging trajectory.

## Timing Estimate

~5.9h: soft ~1200s + HotFlip 80 × 249s ≈ 21120s total

## Comparison

| Exp | LR Schedule | Soft Steps | HotFlip CE |
|-----|------------|-----------|-----------|
| 19 | Single cosine decay | 300 | **0.679** ← SOTA |
| **27** | **SGDR 3×100 restarts** | **300** | **—** |
