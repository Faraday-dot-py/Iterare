# Exp 15: ST + Cosine LR Annealing + Best-Prefix Tracking

**Result: HotFlip CE = 0.73776 (WORSE than Exp11's 0.68935)**

## Method

ST estimator with cosine LR annealing (LR_MAX=0.01 → LR_MIN=0.001 over 300 steps)
and best-prefix tracking (project from lowest-ST-CE snapshot, not final step).
seed=42, BATCH_SIZE=12, HF_TOPK=30.

## Results

| Stage | CE |
|-------|-----|
| Best ST-CE | 0.87847 |
| Final ST-CE | 0.89166 |
| Proj CE (best snapshot) | 0.94974 |
| Proj CE (final step) | 0.95718 |
| Best-prefix benefit | +0.00744 |
| HotFlip CE | **0.73776** |

Final prefix: `" primarily manage cats'];?> continuare narrationtone CAT"`

## Comparison

| Method | Proj CE | HotFlip CE |
|--------|---------|------------|
| Exp 11 (constant LR, no best-tracking) | 0.762 | **0.689** |
| Exp 15 (cosine LR, best-tracking) | 0.950 | 0.738 |

## Key finding

Cosine LR annealing with seed=42 finds a **worse** Voronoi cell (proj=0.868 vs 0.762)
despite best-prefix tracking. The lower-LR late phase settled into a different local
minimum. Best-prefix tracking gave only +0.007 CE benefit.

Conclusion: cosine annealing is **not reliably better** than constant LR for ST optimization.
