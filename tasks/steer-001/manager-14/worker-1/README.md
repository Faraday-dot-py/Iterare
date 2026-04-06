# Exp 14: Alternating ST + HotFlip Optimization

**Result: HotFlip CE = 0.701 — worse than Exp11 (0.689). Approach failed.**

## Method

Alternating rounds of:
1. ST soft optimization (warm-started from previous HotFlip result)
2. HotFlip discrete refinement

Round 0: 300 ST steps from random init → project → 80 HotFlip steps
Rounds 1–4: 100 ST steps warm-started from HotFlip result → project → 80 HotFlip steps

Config: BATCH_SIZE=12, HF_TOPK=30, seed=42, 5 rounds total

## Results

| Round | Proj CE | HotFlip CE | New best? |
|-------|---------|------------|-----------|
| 0 | 0.8440 | 0.7015 | yes |
| 1 | 0.7015 | 0.7015 | (fixed pt) |
| 2 | 0.7015 | 0.7015 | (fixed pt) |
| 3 | 0.7015 | 0.7015 | (fixed pt) |
| 4 | 0.7015 | 0.7015 | (fixed pt) |

Best prefix: ` equipping catsClassic narrationAYE snippet text cats`
Total time: 17234s (~4.8h)

## Why it failed

When ST is warm-started from a discrete HotFlip solution, the soft prefix begins
*already at* a discrete token embedding — projection CE immediately equals the HotFlip
CE. The ST gradient at this point is zero (already at the discrete minimum for that
Voronoi cell), so Adam makes no progress. The warm-start creates a fixed point.

The approach would require ST to escape the current Voronoi cell and find a better
one, but there is no gradient signal to do so since the current tokens are already
an exact projection.

## Comparison

| Method | Proj CE | HotFlip CE |
|--------|---------|------------|
| Exp 11 (ST, lucky) | 0.762 | **0.689** ← SOTA |
| Exp 14 (Alternating) | 0.844 | 0.701 |

## Key finding

Alternating ST+HotFlip does not improve over plain ST. The warm-start strategy
produces a fixed point after round 0.
